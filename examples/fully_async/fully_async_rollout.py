import asyncio
import atexit
import os
import queue
import threading
import time

# Import core functions from sglang_rollout directly to avoid code duplication
from miles.rollout.base_types import RolloutFnTrainOutput
from miles.rollout.sglang_rollout import GenerateState, generate_and_rm_group
from miles.utils.async_utils import run
from miles.utils.types import Sample

# Global worker manager
_global_worker = None
_worker_lock = threading.Lock()


def _get_sample_head_version(sample: Sample) -> int:
    # Version awareness: use the first SGLang-reported weight_version (head).
    assert sample.weight_versions, "Expected SGLang to return meta_info['weight_version'] (head version)."
    return int(sample.weight_versions[0])


def _get_group_head_version(group: list[Sample]) -> int:
    # Group head version for off-policyness drop (min across samples).
    return min(_get_sample_head_version(sample) for sample in group)


def _derive_current_train_version(args, rollout_id: int) -> int:
    """Map rollout_id -> trainer weight version under `train_async.py` update cadence."""
    update_interval_raw = getattr(args, "update_weights_interval", 1)
    try:
        update_interval = max(1, int(1 if update_interval_raw is None else update_interval_raw))
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid args.update_weights_interval={update_interval_raw!r}") from e

    start_rollout_id_raw = getattr(args, "start_rollout_id", 0)
    try:
        start_rollout_id = int(0 if start_rollout_id_raw is None else start_rollout_id_raw)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid args.start_rollout_id={start_rollout_id_raw!r}") from e
    return 1 + max(0, rollout_id - start_rollout_id) // update_interval


def get_global_worker(args, data_buffer):
    """Get or create global worker"""
    global _global_worker
    with _worker_lock:
        if _global_worker is None or not _global_worker.worker_thread.is_alive():
            print("Creating new global async worker...")
            _global_worker = AsyncRolloutWorker(args, data_buffer, concurrency=args.sglang_server_concurrency)
            _global_worker.start()
        return _global_worker


def stop_global_worker():
    """Stop global worker"""
    global _global_worker
    with _worker_lock:
        if _global_worker is not None:
            _global_worker.stop()
            _global_worker = None


class AsyncRolloutWorker:
    """
    Simplified asynchronous rollout worker, using threads instead of processes
    Supports continuous running, independent of rollout function lifecycle
    """

    def __init__(self, args, data_buffer, concurrency=10):
        self.args = args
        self.data_buffer = data_buffer  # Directly save data_buffer reference
        self.concurrency = concurrency
        self.running = True
        # Staleness cap (η): max off-policyness / staleness in units of trainer weight versions.
        # (We also use it to size default queue caps in groups: (η + 1) * rollout_batch_size.)
        self.staleness_cap_batches = int(os.environ.get("MAX_STALENESS_BATCHES", "4"))
        b = max(1, int(getattr(self.args, "rollout_batch_size", 1)))
        default_queue_cap_groups = max(1, (self.staleness_cap_batches + 1) * b)

        # Optional knobs: allow overriding inflight and queue caps.
        self.max_inflight_groups = max(1, int(os.environ.get("MAX_INFLIGHT_GROUPS", str(b))))
        self.queue_cap_groups = max(1, int(os.environ.get("QUEUE_CAP_GROUPS", str(default_queue_cap_groups))))

        self.output_queue = queue.Queue(maxsize=self.queue_cap_groups)  # Continuous output queue
        self.worker_thread = None
        self.state = GenerateState(args)

        self._metrics_lock = threading.Lock()
        self._inflight_groups = 0
        self._producer_blocked_s = 0.0
        self._producer_block_events = 0
        self._producer_blocked_start = None

    async def continuous_worker_loop(self):
        """Continuous work loop - constantly get data from data_buffer and process"""
        print(
            "Continuous async rollout worker started "
            f"({self.staleness_cap_batches=} {self.max_inflight_groups=} {self.queue_cap_groups=})"
        )

        active_tasks = set()
        max_concurrent_tasks = self.max_inflight_groups
        group_id_counter = 0

        while self.running:
            try:
                # Clean up completed tasks
                if active_tasks:
                    done_tasks = {task for task in active_tasks if task.done()}
                    for task in done_tasks:
                        try:
                            task.result()  # Results are already handled in callbacks
                        except Exception as e:
                            print(f"Task failed with exception: {e}")
                    active_tasks -= done_tasks
                with self._metrics_lock:
                    self._inflight_groups = len(active_tasks)

                # Backpressure: if the completed output queue is full, stop submitting until it is drained.
                if self.running and len(active_tasks) < max_concurrent_tasks and self.output_queue.full():
                    with self._metrics_lock:
                        if self._producer_blocked_start is None:
                            self._producer_blocked_start = time.time()
                            self._producer_block_events += 1
                    await asyncio.sleep(0.1)
                    continue

                with self._metrics_lock:
                    if self._producer_blocked_start is not None:
                        self._producer_blocked_s += time.time() - self._producer_blocked_start
                        self._producer_blocked_start = None

                while (
                    len(active_tasks) < max_concurrent_tasks
                    and self.running
                    and not self.output_queue.full()
                ):
                    samples = self.data_buffer.get_samples(1)

                    for group in samples:
                        group_id = group_id_counter
                        group_id_counter += 1

                        # Create new async task
                        task = asyncio.create_task(
                            generate_and_rm_group(
                                self.args,
                                group,
                                sampling_params=self.state.sampling_params.copy(),
                                evaluation=False,
                            )
                        )

                        # Add completion callback
                        def make_callback(gid):
                            def task_done_callback(done_task):
                                try:
                                    result = done_task.result()
                                except Exception as e:
                                    # Keep the event loop healthy even if a task fails.
                                    print(f"Task failed with exception: {e}")
                                    return

                                # IMPORTANT: never block the asyncio event loop thread in a callback.
                                # `queue.Queue.put()` can block when the queue is full and stall all async progress.
                                try:
                                    self.output_queue.put_nowait((gid, result))
                                except queue.Full:
                                    # Offload the blocking put to a daemon thread so the event loop can keep running.
                                    threading.Thread(
                                        target=self.output_queue.put,
                                        args=((gid, result),),
                                        daemon=True,
                                    ).start()

                            return task_done_callback

                        task.add_done_callback(make_callback(group_id))
                        active_tasks.add(task)
                        break

                # Brief sleep to avoid busy waiting
                await asyncio.sleep(1)

            except Exception as e:
                print(f"Error in continuous worker loop: {e}")
                await asyncio.sleep(1)

        if active_tasks:
            print(f"Waiting for {len(active_tasks)} continuous tasks to complete...")
            await asyncio.wait(active_tasks)

        with self._metrics_lock:
            if self._producer_blocked_start is not None:
                self._producer_blocked_s += time.time() - self._producer_blocked_start
                self._producer_blocked_start = None

        print("Continuous async rollout worker stopped")

    def worker_thread_func(self):
        """Worker function running in independent thread"""
        asyncio.run(self.continuous_worker_loop())

    def start(self):
        """Start continuous work mode"""
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.worker_thread = threading.Thread(target=self.worker_thread_func, daemon=True)
            self.worker_thread.start()
            print("Started continuous async worker thread")

    def stop(self):
        """Stop worker thread"""
        self.running = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5)
        print("Stopped async worker thread")

    def get_completed_groups(self) -> list[tuple]:
        """Get completed sample groups"""
        completed = []
        while True:
            try:
                result = self.output_queue.get_nowait()
                completed.append(result)
            except queue.Empty:
                break
        return completed

    def try_get_completed_group(self) -> tuple | None:
        """Try to get a single completed group without draining the entire queue."""
        try:
            return self.output_queue.get_nowait()
        except queue.Empty:
            return None

    def get_queue_size(self) -> int:
        """Get current output queue size"""
        return self.output_queue.qsize()

    def get_inflight_groups(self) -> int:
        with self._metrics_lock:
            return self._inflight_groups

    def get_producer_blocked_s(self) -> float:
        with self._metrics_lock:
            blocked_s = self._producer_blocked_s
            if self._producer_blocked_start is not None:
                blocked_s += time.time() - self._producer_blocked_start
            return blocked_s

    def get_producer_block_events(self) -> int:
        with self._metrics_lock:
            return self._producer_block_events


async def generate_rollout_async(args, rollout_id: int, data_buffer) -> tuple[list[list[Sample]], dict]:
    """
    Simplified asynchronous rollout generation - using global continuous worker
    """
    assert args.rollout_global_dataset

    # Get global worker, which will run continuously
    worker = get_global_worker(args, data_buffer)

    # Simplified: directly use rollout_batch_size as target
    target_data_size = args.rollout_batch_size

    data = []
    do_print = True

    print(f"Starting async rollout generation for {target_data_size} groups")
    print(f"Global worker queue size: {worker.get_queue_size()}")

    # Main loop: collect results from global worker's output queue
    start_time = time.time()
    last_progress_time = start_time
    no_progress_timeout = 30.0  # Warn if no progress for 30 seconds

    start_blocked_s = worker.get_producer_blocked_s()
    start_blocked_events = worker.get_producer_block_events()
    aborted_groups_requeued = 0
    aborted_samples_requeued = 0
    current_version = _derive_current_train_version(args, rollout_id)
    max_head_offpolicyness = worker.staleness_cap_batches  # η, in trainer weight versions
    stale_groups_dropped = 0
    accepted_staleness_values = []
    dropped_staleness_values = []

    while len(data) < target_data_size:
        processed_any = False
        # Consume completed groups incrementally until we have enough accepted groups.
        # Do NOT drain the entire queue; extra completed groups should remain queued for the next call.
        while len(data) < target_data_size:
            item = worker.try_get_completed_group()
            if item is None:
                break
            group_id, group = item
            last_progress_time = time.time()
            processed_any = True

            # If any sample in the group was aborted, return the whole group to the data buffer
            # and do not forward it to the training engine.
            try:
                any_aborted = any([sample.status == Sample.Status.ABORTED for sample in group])
            except Exception:
                any_aborted = False

            if any_aborted:
                aborted_groups_requeued += 1
                try:
                    aborted_samples_requeued += sum(1 for sample in group if sample.status == Sample.Status.ABORTED)
                except Exception:
                    pass
                try:
                    # add back to buffer so it can be retried or handled by buffer policy
                    data_buffer.add_samples([group])
                    print(f"Returned aborted group {group_id} to data buffer", flush=True)
                except Exception as e:
                    print(f"Failed to return aborted group {group_id} to buffer: {e}", flush=True)
                # don't count as processed for training
                continue

            # Version-aware offpolicyness drop (dequeue-time).
            head_version = _get_group_head_version(group)
            staleness = current_version - head_version
            if staleness > max_head_offpolicyness:
                stale_groups_dropped += 1
                dropped_staleness_values.append(staleness)
                continue
            accepted_staleness_values.append(staleness)

            if do_print:
                print(
                    f"First rollout sample: {[group[0].prompt + group[0].response]}, "
                    f"label: {group[0].label}, reward: {group[0].reward}",
                    flush=True,
                )
                do_print = False

            # Simplified: directly add samples, no filters used
            data.append(group)

        # Check progress
        current_time = time.time()
        if current_time - last_progress_time > no_progress_timeout:
            print(
                f"Warning: No progress for {no_progress_timeout}s. "
                f"Queue size: {worker.get_queue_size()}, "
                f"Collected: {len(data)}/{target_data_size}"
            )
            last_progress_time = current_time

        # If no results were processed, brief sleep to avoid busy waiting
        if not processed_any:
            await asyncio.sleep(0.01)

    duration = time.time() - start_time
    print(f"Rollout completed in {duration:.2f}s! Global worker queue size: {worker.get_queue_size()}")

    if data:
        print(
            f"Finish rollout: {[data[-1][0].prompt + data[-1][0].response]}, "
            f"label: {data[-1][0].label}, reward: {data[-1][0].reward}",
            flush=True,
        )

    data = sorted(data, key=lambda group: group[0].index)
    metrics = {
        "async/queue_size_groups": worker.get_queue_size(),
        "async/inflight_groups": worker.get_inflight_groups(),
        "async/staleness_cap_batches": worker.staleness_cap_batches,
        "async/aborted_groups_requeued": aborted_groups_requeued,
        "async/aborted_samples_requeued": aborted_samples_requeued,
        "async/producer_blocked_s": worker.get_producer_blocked_s() - start_blocked_s,
        "async/producer_block_events": worker.get_producer_block_events() - start_blocked_events,
        # Version-aware offpolicyness (based on train_async cadence).
        "async/offpolicy_current_version": current_version,
        "async/offpolicy_max_head_offpolicyness": max_head_offpolicyness,
        "async/offpolicy_dropped_stale_groups": stale_groups_dropped,
        "async/offpolicy_dropped_staleness_max": (
            max(dropped_staleness_values) if dropped_staleness_values else 0
        ),
        "async/offpolicy_accepted_staleness_max": (
            max(accepted_staleness_values) if accepted_staleness_values else 0
        ),
    }
    return data, metrics


def generate_rollout_fully_async(args, rollout_id, data_buffer, evaluation=False):
    if evaluation:
        raise ValueError("Evaluation mode not supported in simple async rollout")

    completed_samples, metrics = run(generate_rollout_async(args, rollout_id, data_buffer))
    return RolloutFnTrainOutput(samples=completed_samples, metrics=metrics)


# Register exit cleanup function

atexit.register(stop_global_worker)
