import atexit
import threading
from typing import Any, Optional

from miles.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from miles.rollout.inflight_actor import InflightRolloutGenerator

# NOTE: Keep PipelineRL-specific rollout behavior self-contained here (similar to
# `examples/fully_async/fully_async_rollout.py`), so the core RolloutManager logic
# doesn't need PipelineRL conditionals.

_global_inflight: Optional[InflightRolloutGenerator] = None
_global_lock = threading.Lock()


def _get_global_inflight(args, data_buffer: Any) -> InflightRolloutGenerator:
    global _global_inflight
    with _global_lock:
        if _global_inflight is None:
            _global_inflight = InflightRolloutGenerator(args, data_buffer.get_samples)
        # Idempotent; safe to call repeatedly.
        _global_inflight.start()
        return _global_inflight


def stop_global_inflight() -> None:
    global _global_inflight
    with _global_lock:
        if _global_inflight is not None:
            _global_inflight.stop()
            _global_inflight = None


def generate_rollout(
    args, rollout_id: int, data_buffer: Any, evaluation: bool = False
) -> RolloutFnTrainOutput | RolloutFnEvalOutput:
    """PipelineRL rollout entrypoint.

    - Training mode (`evaluation=False`): consume groups from a long-lived inflight
      generator that continuously maintains a fixed number of in-flight requests.
    - Eval mode: not supported here; set `--eval-function-path` explicitly.
    """
    if evaluation:
        raise ValueError(
            "PipelineRL rollout function does not support evaluation. "
            "Set --eval-function-path to a non-PipelineRL rollout implementation."
        )

    if not getattr(args, "pipeline_rl", False):
        raise ValueError(
            "miles.rollout.pipelinerl_rollout.generate_rollout requires --pipeline-rl"
        )
    if getattr(args, "debug_train_only", False):
        raise ValueError("PipelineRL rollout generation is disabled when debug_train_only is set")

    inflight = _get_global_inflight(args, data_buffer)
    groups = inflight.get_next_groups(args.rollout_batch_size)

    # Flatten and deterministically order samples (matches the previous RolloutManager
    # PipelineRL branch behavior).
    samples = []
    for group in groups:
        if not group:
            continue
        if isinstance(group[0], list):
            samples += sum(group, [])
        else:
            samples += group
    try:
        samples = sorted(samples, key=lambda s: s.index)
    except Exception:
        pass

    return RolloutFnTrainOutput(samples=samples)


# Ensure cleanup on process exit (mirrors `examples/fully_async/fully_async_rollout.py`).
atexit.register(stop_global_inflight)


