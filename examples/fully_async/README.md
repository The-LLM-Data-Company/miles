## Fully Asynchronous Rollout Example

This example shows a simple way to make rollout generation **fully asynchronous**: a single global worker is created once and then keeps running in the background, continuously pulling prompts and launching generation tasks. Training only needs to fetch already finished results. This removes the per‑step wait that happens in the normal synchronous style.

This version adds **AReaL-style staleness control** via backpressure and dequeue-time version gating. When a weight update interrupts in-flight generation, interrupted groups are re-queued and resumed (KV cache is recomputed on resume).

### Files
* `fully_async_rollout.py`: global async worker + `generate_rollout_fully_async` entry (returns `RolloutFnTrainOutput` with metrics).
* `run-qwen3-4b-fully_async.sh`: example launch script with Qwen3‑4B.

### Prerequisite
First set up model & environment following the Qwen3-4B example.

### Quick Start
```bash
cd miles
bash examples/fully_async/run-qwen3-4b-fully_async.sh
```
You should see log lines like:
```
Creating new global async worker...
Continuous async rollout worker started
```

### How It Works (Very Short)
* First call: create `AsyncRolloutWorker` (thread + asyncio loop).
* Loop keeps up to `MAX_INFLIGHT_GROUPS` tasks in flight (default: `--rollout-batch-size`) using `generate_and_rm_group`.
* Completed groups are pushed into a queue; caller pops until it has enough samples (leftover completions remain queued for the next step).
* Worker is stopped automatically at process exit.
* **AReaL additions**:
  * Backpressure: producer uses two hard caps: `inflight_groups <= MAX_INFLIGHT_GROUPS` and `queue_size_groups < QUEUE_CAP_GROUPS`.
  * Interrupt recovery: if a group is interrupted (`ABORTED`), it is pushed back into the data buffer for retry (KV recomputed on resume).
  * Off-policyness guardrail: at dequeue time, drop groups whose **head** `weight_version` is more than `η` versions behind the current training version.

### Limitations
* No evaluation mode.
* Ordering is best effort (sorted at the end by index).
* Minimal error handling.

### Config Differences
To enable the AReaL async pattern there are only two changes compared to a normal run:

1. Use the async training driver: `train_async.py` (not `train.py`).
2. Set the rollout function path:
	```bash
	--rollout-function-path fully_async_rollout.generate_rollout_fully_async
	```

Why is it still "fully" async although `train_async.py` itself schedules rollouts step‑by‑step?

Because the real generation work is done by a **persistent background worker** created in `generate_rollout_fully_async`. Each call from `train_async.py` only drains already completed samples from the worker's output queue; the worker has been continuously generating since the first call. Thus rollout production (model inference) and training consume happen in parallel with minimal waiting.

### Knobs
* `MAX_STALENESS_BATCHES` (η): max head off-policyness / staleness in units of trainer weight versions. Default: `4`.
* `MAX_INFLIGHT_GROUPS`: max concurrent in-flight groups. Default: `--rollout-batch-size`.
* `QUEUE_CAP_GROUPS`: max completed queue depth in groups. Default: `(η + 1) * --rollout-batch-size`.
* `--use-tis`: recommended for off-policy tolerance.
