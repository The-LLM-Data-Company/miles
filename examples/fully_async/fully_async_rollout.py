import asyncio
import atexit
import os
import threading
import time

from miles.rollout.base_types import RolloutFnTrainOutput
from miles.rollout.sglang_rollout import GenerateState, generate_and_rm_group
from miles.utils.async_utils import get_async_loop, run
from miles.utils.types import Sample

_global_worker = None
_worker_lock = threading.Lock()


def _get_sample_head_version(sample: Sample) -> int:
    if not sample.weight_versions:
        raise RuntimeError("Expected SGLang to return meta_info['weight_version'] for staleness enforcement.")
    return int(sample.weight_versions[0])


def _get_group_head_version(group: list[Sample]) -> int:
    if not group:
        raise RuntimeError("Unexpected empty group.")
    return min(_get_sample_head_version(s) for s in group)


def _derive_current_train_version(args, rollout_id: int) -> int:
    update_interval = max(1, args.update_weights_interval)
    return 1 + max(0, rollout_id - args.start_rollout_id) // update_interval


def get_global_worker(args, data_buffer):
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
        b = max(1, args.rollout_batch_size)
        default_queue_cap = (self.staleness_cap_batches + 1) * b

        self.max_inflight_groups = max(1, int(os.environ.get("MAX_INFLIGHT_GROUPS", str(b))))
        self.queue_cap_groups = max(1, int(os.environ.get("QUEUE_CAP_GROUPS", str(default_queue_cap))))

        self.output_queue: asyncio.PriorityQueue = asyncio.PriorityQueue(maxsize=self.queue_cap_groups)
        self._inflight_sem = asyncio.Semaphore(self.max_inflight_groups)
        self._producer_task = None
        self._state = GenerateState(args)

    def is_running(self) -> bool:
        return self._producer_task is not None and not self._producer_task.done()

    async def _producer_loop(self) -> None:
        print(f"AsyncRolloutWorker started (η={self.staleness_cap_batches}, "
              f"inflight={self.max_inflight_groups}, queue={self.queue_cap_groups})")
        gid = 0
        while self.running:
            await self._inflight_sem.acquire()
            samples = self.data_buffer.get_samples(1)
            if not samples:
                self._inflight_sem.release()
                await asyncio.sleep(0.01)
                continue
            asyncio.create_task(self._process_group(samples[0], gid))
            gid += 1

    def start(self):
        if self._producer_task is None or self._producer_task.done():
            loop = get_async_loop().loop
            self._producer_task = asyncio.run_coroutine_threadsafe(self._producer_loop(), loop)

    def stop(self):
        self.running = False
        if self._producer_task is not None:
            self._producer_task.cancel()

    def try_get_completed_group(self) -> tuple | None:
        try:
            _prio, item = self.output_queue.get_nowait()
            return item
        except asyncio.QueueEmpty:
            return None

    def get_queue_size(self) -> int:
        return self.output_queue.qsize()

    def get_inflight_groups(self) -> int:
        return self.max_inflight_groups - self._inflight_sem._value

    async def _process_group(self, group, gid: int) -> None:
        try:
            result = await generate_and_rm_group(
                self.args, group, sampling_params=self._state.sampling_params.copy(), evaluation=False
            )
            item = ((_get_group_head_version(result), time.time(), gid), (gid, result))
            await self.output_queue.put(item)
        except Exception as e:
            print(f"Task {gid} failed: {e}")
        finally:
            self._inflight_sem.release()


async def generate_rollout_async(args, rollout_id: int, data_buffer) -> tuple[list[list[Sample]], dict]:
    if not args.rollout_global_dataset:
        raise ValueError("fully_async rollout requires --rollout-global-dataset")

    worker = get_global_worker(args, data_buffer)
    target_size = args.rollout_batch_size

    data = []
    start_time = time.time()
    last_progress_time = start_time
    aborted_groups_requeued = 0
    current_version = _derive_current_train_version(args, rollout_id)
    max_staleness = worker.staleness_cap_batches
    stale_groups_dropped = 0
    accepted_staleness = []

    while len(data) < target_size:
        item = worker.try_get_completed_group()
        if item is None:
            if time.time() - last_progress_time > 30.0:
                print(f"Warning: no progress for 30s (queue={worker.get_queue_size()}, got={len(data)}/{target_size})")
                last_progress_time = time.time()
            await asyncio.sleep(0.01)
            continue

        group_id, group = item
        last_progress_time = time.time()

        if any(s.status == Sample.Status.ABORTED for s in group):
            aborted_groups_requeued += 1
            try:
                data_buffer.add_samples([group])
            except Exception:
                pass
            continue

        head_version = _get_group_head_version(group)
        staleness = current_version - head_version
        if staleness > max_staleness:
            stale_groups_dropped += 1
            continue
        accepted_staleness.append(staleness)
        data.append(group)

    print(f"Rollout completed in {time.time() - start_time:.2f}s")

    data = sorted(data, key=lambda g: g[0].index)
    total_consumed = len(data) + stale_groups_dropped + aborted_groups_requeued
    return data, {
        "async/queue_size_groups": worker.get_queue_size(),
        "async/inflight_groups": worker.get_inflight_groups(),
        "async/groups_consumed": total_consumed,
        "async/aborted_groups_requeued": aborted_groups_requeued,
        "async/offpolicy_dropped_stale_groups": stale_groups_dropped,
        "async/offpolicy_drop_rate": stale_groups_dropped / total_consumed if total_consumed else 0,
        "async/offpolicy_accepted_staleness_mean": sum(accepted_staleness) / len(accepted_staleness) if accepted_staleness else 0,
        "async/offpolicy_accepted_staleness_max": max(accepted_staleness) if accepted_staleness else 0,
    }


def generate_rollout_fully_async(args, rollout_id, data_buffer, evaluation=False):
    if evaluation:
        raise ValueError("Evaluation mode not supported")
    completed, metrics = run(generate_rollout_async(args, rollout_id, data_buffer))
    return RolloutFnTrainOutput(samples=completed, metrics=metrics)


atexit.register(stop_global_worker)
