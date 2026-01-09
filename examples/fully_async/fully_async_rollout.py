from __future__ import annotations
import asyncio
import atexit
import logging
import os
import threading
import time
from dataclasses import dataclass, field

from miles.rollout.base_types import RolloutFnTrainOutput
from miles.rollout.sglang_rollout import GenerateState, generate_and_rm_group
from miles.utils.async_utils import get_async_loop, run
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

_global_worker = None
_worker_lock = threading.Lock()


@dataclass(frozen=True, order=True, slots=True)
class Priority:
    head_version: int
    timestamp: float
    group_id: int


@dataclass(order=True, slots=True)
class PQItem:
    # PriorityQueue will compare PQItem by this field (because order=True)
    priority: Priority
    # Payload should not participate in ordering
    samples: list[Sample] = field(compare=False)


def _get_sample_head_version(sample: Sample) -> int:
    if not sample.weight_versions:
        raise RuntimeError(
            "Expected SGLang to return meta_info['weight_version'] for staleness enforcement."
        )
    return int(sample.weight_versions[0])


def _get_group_head_version(group: list[Sample]) -> int:
    if not group:
        raise RuntimeError("Unexpected empty group.")
    return min(_get_sample_head_version(s) for s in group)


def _get_current_train_version(args, rollout_id: int) -> int:
    update_interval = max(1, args.update_weights_interval)
    return 1 + max(0, rollout_id - args.start_rollout_id) // update_interval


def get_global_worker(args, data_buffer) -> AsyncRolloutWorker:
    global _global_worker
    with _worker_lock:
        if _global_worker is None or not _global_worker.is_running():
            _global_worker = AsyncRolloutWorker(args, data_buffer)
            _global_worker.start()
        return _global_worker


def stop_global_worker():
    global _global_worker
    with _worker_lock:
        if _global_worker is not None:
            _global_worker.stop()
            _global_worker = None


class AsyncRolloutWorker:
    """Long-lived producer coroutine on Miles' global asyncio loop."""

    def __init__(self, args, data_buffer):
        self.args = args
        self.data_buffer = data_buffer
        self.running = True

        self.staleness_cap_batches = int(os.environ.get("MAX_STALENESS_BATCHES", "4"))
        batch_size = max(1, args.rollout_batch_size)
        default_queue_cap = (self.staleness_cap_batches + 1) * batch_size

        self.max_inflight_groups = max(1, int(os.environ.get("MAX_INFLIGHT_GROUPS", str(batch_size))))
        self.queue_cap_groups = max(1, int(os.environ.get("QUEUE_CAP_GROUPS", str(default_queue_cap))))

        self.output_queue: asyncio.PriorityQueue[PQItem] = asyncio.PriorityQueue(maxsize=self.queue_cap_groups)
        self._inflight_sem = asyncio.Semaphore(self.max_inflight_groups)
        self._producer_task = None
        self._state = GenerateState(args)

        self._rollout_group_failures = 0

    def is_running(self) -> bool:
        return self._producer_task is not None and not self._producer_task.done()

    async def _producer_loop(self) -> None:
        logger.info(
            "AsyncRolloutWorker started (η=%s, inflight=%s, queue=%s)",
            self.staleness_cap_batches,
            self.max_inflight_groups,
            self.queue_cap_groups,
        )
        group_id = 0
        while self.running:
            samples = self.data_buffer.get_samples(1)
            if not samples:
                await asyncio.sleep(0.01)
                continue

            await self._inflight_sem.acquire()
            asyncio.create_task(self._process_group(samples[0], group_id))
            group_id += 1

    def start(self):
        if self._producer_task is None or self._producer_task.done():
            loop = get_async_loop().loop
            self._producer_task = asyncio.run_coroutine_threadsafe(self._producer_loop(), loop)

    def stop(self):
        self.running = False
        if self._producer_task is not None:
            self._producer_task.cancel()

    async def get_completed_group(self) -> tuple[PQItem, float]:
        """
        Returns (PQItem, blocked_seconds).
        Logs every 1s while blocked.
        """
        start = time.monotonic()
        while True:
            try:
                item = await asyncio.wait_for(self.output_queue.get(), timeout=1.0)
                return item, (time.monotonic() - start)
            except asyncio.TimeoutError:
                logger.info("Blocked %.1fs waiting for completed group", time.monotonic() - start)

    def get_queue_size(self) -> int:
        return self.output_queue.qsize()

    def get_inflight_groups(self) -> int:
        return self.max_inflight_groups - self._inflight_sem._value

    def get_rollout_group_failures_total(self) -> int:
        return self._rollout_group_failures

    async def _process_group(self, group, group_id: int) -> None:
        try:
            result = await generate_and_rm_group(
                self.args,
                group,
                sampling_params=self._state.sampling_params.copy(),
                evaluation=False,
            )
            prio = Priority(
                head_version=_get_group_head_version(result),
                timestamp=time.monotonic(),
                group_id=group_id,
            )
            await self.output_queue.put(PQItem(priority=prio, samples=result))
        except Exception:
            logger.exception("Rollout for prompt group %s failed", group_id)
            self._rollout_group_failures += 1
        finally:
            self._inflight_sem.release()


async def generate_rollout_async(args, rollout_id: int, data_buffer) -> tuple[list[list[Sample]], dict]:
    if not args.rollout_global_dataset:
        raise ValueError("fully_async rollout requires --rollout-global-dataset")

    worker = get_global_worker(args, data_buffer)
    target_size = args.rollout_batch_size

    prompt_groups: list[list[Sample]] = []
    start_time = time.monotonic()

    interrupted_groups_requeued = 0
    current_version = _get_current_train_version(args, rollout_id)
    max_staleness = worker.staleness_cap_batches
    stale_groups_dropped = 0
    accepted_staleness: list[int] = []
    interrupted_group_requeue_failures = 0

    total_blocked_secs = 0.0
    while len(prompt_groups) < target_size:
        pq_item, blocked_secs = await worker.get_completed_group()
        total_blocked_secs += blocked_secs
        group = pq_item.samples

        if any(s.status == Sample.Status.ABORTED for s in group):
            interrupted_groups_requeued += 1
            try:
                data_buffer.add_samples([group])
            except Exception:
                logger.exception(
                    "Failed to requeue in-flight aborted prompt group %s",
                    pq_item.priority.group_id,
                )
                interrupted_group_requeue_failures += 1
            continue

        staleness = current_version - pq_item.priority.head_version
        if staleness > max_staleness:
            stale_groups_dropped += 1
            continue
        accepted_staleness.append(staleness)
        prompt_groups.append(group)

    logger.info("Rollout consumer completed in %.2fs", time.monotonic() - start_time)

    prompt_groups = sorted(prompt_groups, key=lambda g: g[0].index)
    total_consumed = len(prompt_groups) + stale_groups_dropped + interrupted_groups_requeued
    return prompt_groups, {
        "async/queue_size_groups": worker.get_queue_size(),
        "async/inflight_groups": worker.get_inflight_groups(),
        "async/groups_consumed": total_consumed,
        "async/interrupted_groups_requeued": interrupted_groups_requeued,
        "async/interrupted_group_requeue_failures": interrupted_group_requeue_failures,
        "async/offpolicy_dropped_stale_groups": stale_groups_dropped,
        "async/offpolicy_drop_rate": stale_groups_dropped / total_consumed if total_consumed else 0,
        "async/offpolicy_accepted_staleness_mean": (
            sum(accepted_staleness) / len(accepted_staleness) if accepted_staleness else 0
        ),
        "async/offpolicy_accepted_staleness_max": max(accepted_staleness) if accepted_staleness else 0,
        "async/rollout_consumer_wait_secs": total_blocked_secs,
        "async/rollout_group_failures_total": worker.get_rollout_group_failures_total(),
    }


def generate_rollout_fully_async(args, rollout_id, data_buffer, evaluation=False):
    if evaluation:
        raise ValueError("Evaluation mode not supported")
    completed, metrics = run(generate_rollout_async(args, rollout_id, data_buffer))
    return RolloutFnTrainOutput(samples=completed, metrics=metrics)


atexit.register(stop_global_worker)
