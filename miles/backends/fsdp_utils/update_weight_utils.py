import abc
import logging
import socket
from datetime import timedelta
from argparse import Namespace
from collections.abc import Sequence

import ray
import torch
import torch.distributed as dist
from ray.actor import ActorHandle
from torch.distributed.tensor import DTensor, Replicate

try:
    from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions  # type: ignore[import]
except ImportError:
    from sglang.srt.patch_torch import monkey_patch_torch_reductions  # type: ignore[import]

from sglang.srt.utils import MultiprocessingSerializer

from miles.utils.distributed_utils import init_process_group


try:
    from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket  # type: ignore[import]
except ImportError:
    from sglang.srt.model_executor.model_runner import FlattenedTensorBucket  # type: ignore[import]


logger = logging.getLogger(__name__)

_PIPELINE_RL_UPDATE_GROUP_TIMEOUT_S = 120


def _raise_if_any_engine_failed(results: list[dict], *, weight_version: int) -> None:
    """Fail-closed if any rollout engine reports update failure."""
    failures = []
    for i, r in enumerate(results):
        try:
            success = bool(r.get("success", True))
            message = str(r.get("message", ""))
        except Exception:
            success = False
            message = f"unparseable result: {r!r}"
        if not success:
            failures.append((i, message))
    if failures:
        msg = " | ".join([f"engine[{i}]={m}" for i, m in failures])
        raise RuntimeError(
            f"PipelineRL weight update failed: weight_version={weight_version}. {msg}"
        )


class UpdateWeight(abc.ABC):
    def __init__(self, args: Namespace, model: torch.nn.Module) -> None:
        self.args = args
        self.model = model
        self.weight_version = 0

    @abc.abstractmethod
    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle | None,
    ) -> None:
        pass

    def update_weights(self) -> None:
        self.weight_version += 1
        self._do_update_weights()

    def _do_update_weights(self) -> None:
        """Core weight update logic. Override in subclasses to wrap with pause/continue."""
        bucket = []
        bucket_size = 0
        for name, param in self.model.state_dict().items():
            param_size = param.numel() * param.element_size()
            if bucket and bucket_size + param_size >= self.args.update_weight_buffer_size:
                self.wait_and_update_bucket_weights(bucket)
                del bucket
                bucket = []
                bucket_size = 0

            param = param.cuda()
            if isinstance(param, DTensor):
                # async version of param.full_tensor
                param = param.redistribute(
                    placements=[Replicate()] * param.device_mesh.ndim,
                    async_op=True,
                ).to_local()
            bucket.append((name, param))
            bucket_size += param_size

        if bucket:
            self.wait_and_update_bucket_weights(bucket)
            del bucket
            bucket = []
            bucket_size = 0

    def wait_and_update_bucket_weights(self, bucket):
        bucket = [(name, param.wait()) if hasattr(param, "wait") else (name, param) for name, param in bucket]
        self.update_bucket_weights(bucket)

    @abc.abstractmethod
    def update_bucket_weights(self, named_tensors) -> None:
        pass


class UpdateWeightFromTensor(UpdateWeight):
    """Push model weights to rollout engines using tensors.

    Streams parameters in size-bounded buckets; optionally groups tensors by dtype
    and flattens per dtype, gathers per-rank blobs to the source, and issues one
    RPC per dtype per bucket (or one per bucket if not flattened).
    """

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle | None,
    ) -> None:
        """Attach rollout engines and create per-engine IPC (Gloo) groups.

        Sets the gather source rank, engine handle, and `tp_rank` within the
        engine's local group.
        """
        self.rollout_engines = rollout_engines

        # Here we assume the gpu id of rollout engines and train actors are the same.
        for i, engine in enumerate(self.rollout_engines):
            start_rank = i * self.args.rollout_num_gpus_per_engine
            end_rank = (i + 1) * self.args.rollout_num_gpus_per_engine
            group_ranks = list(range(start_rank, end_rank))
            new_group = dist.new_group(
                ranks=group_ranks,
                backend="gloo",
            )
            if dist.get_rank() in group_ranks:
                self._ipc_gather_src = start_rank
                self._ipc_gather_group = new_group
                self._ipc_engine = engine
                # Calculate TP rank within this SGLang engine group
                self.tp_rank = dist.get_rank() - start_rank

    def update_bucket_weights(self, named_tensors) -> None:
        monkey_patch_torch_reductions()
        # Use flattened bucket approach similar to Megatron
        logger.info("Using flattened tensor bucket")
        # Group tensors by dtype (same as Megatron)
        named_tensors_by_dtypes = {}
        for name, tensor in named_tensors:
            dtype = tensor.dtype
            if dtype not in named_tensors_by_dtypes:
                named_tensors_by_dtypes[dtype] = []
            named_tensors_by_dtypes[dtype].append((name, tensor))

        # Create flattened bucket for each dtype group
        serialized_tensors = []
        for _dtype, named_tensors in named_tensors_by_dtypes.items():
            flattened_tensor_bucket = FlattenedTensorBucket(named_tensors=named_tensors)
            metadata = flattened_tensor_bucket.get_metadata()
            flattened_tensor_data = {
                "flattened_tensor": flattened_tensor_bucket.get_flattened_tensor(),
                "metadata": metadata,
            }
            serialized_tensors.append(MultiprocessingSerializer.serialize(flattened_tensor_data, output_str=True))

        if self._ipc_gather_src == dist.get_rank():
            # On rank 0, prepare a list to hold the gathered batches from all ranks.
            gathered_serialized_batches = [None for _ in range(dist.get_world_size(self._ipc_gather_group))]
        else:
            gathered_serialized_batches = None

        # Gather the serialized batches from all ranks to rank 0.
        dist.gather_object(
            obj=serialized_tensors,
            object_gather_list=gathered_serialized_batches,
            dst=self._ipc_gather_src,
            group=self._ipc_gather_group,
        )

        if dist.get_rank() == self._ipc_gather_src:
            # Handle flattened bucket format (same as Megatron approach)
            # Each rank may have multiple dtype buckets
            # TODO: here we assume all ranks have the same number of dtypes
            num_dtypes = len(gathered_serialized_batches[0])
            assert num_dtypes > 0
            for i in range(num_dtypes):
                kwargs = {
                    "serialized_named_tensors": [tensors[i] for tensors in gathered_serialized_batches],
                    "load_format": "flattened_bucket",
                    "flush_cache": False,
                    "weight_version": str(self.weight_version),
                }
                ref = self._ipc_engine.update_weights_from_tensor.remote(**kwargs)
                ray.get(ref)

        if dist.get_rank() == self._ipc_gather_src:
            ref = self._ipc_engine.flush_cache.remote()
            ray.get(ref)


class UpdateWeightFromDistributed(UpdateWeight):
    """Broadcast weights via a temporary NCCL group to rollout engines."""

    def __init__(self, args: Namespace, model: torch.nn.Module) -> None:
        super().__init__(args=args, model=model)

    def update_weights(self) -> None:
        """Override to add PipelineRL pause/verify/continue for distributed engines."""
        self.weight_version += 1

        if dist.get_rank() == 0 and getattr(self.args, "pipeline_rl", False):
            ray.get([engine.pause_generation.remote(mode="in_place_safe") for engine in self.rollout_engines])

        dist.barrier()
        err: Exception | None = None
        try:
            self._do_update_weights()
        except Exception as e:
            err = e

        # Prevent deadlocks: make all ranks observe failure and exit instead of
        # hanging at the post-update barrier.
        err_flag = torch.tensor([1 if err is not None else 0], device=torch.cuda.current_device())
        dist.all_reduce(err_flag, op=dist.ReduceOp.MAX)
        dist.barrier()

        if err_flag.item() != 0:
            if err is not None:
                raise err
            raise RuntimeError(
                f"PipelineRL weight update failed on another rank (weight_version={self.weight_version}); "
                "see logs on rank 0."
            )

        if dist.get_rank() == 0 and getattr(self.args, "pipeline_rl", False):
            ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])
        dist.barrier()

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle | None,
    ) -> None:
        """On rank 0, initialize a temporary NCCL group for parameter broadcast."""
        self.rollout_engines = rollout_engines
        self.rollout_engine_lock = rollout_engine_lock

        # For TP:
        #   1. AllGather parameters to rank 0
        #   2. Broadcast parameters from rank 0 to all sglang engines
        self._is_src_rank = dist.get_rank() == 0
        if self._is_src_rank:
            self._group_name = "miles"
            master_address = ray._private.services.get_node_ip_address()
            with socket.socket() as sock:
                sock.bind(("", 0))
                master_port = sock.getsockname()[1]
            ## TODO: why +1?
            world_size = self.args.rollout_num_gpus + 1

            refs = [
                engine.init_weights_update_group.remote(
                    master_address,
                    master_port,
                    i * self.args.rollout_num_gpus_per_engine + 1,
                    world_size,
                    self._group_name,
                    backend="nccl",
                )
                for i, engine in enumerate(self.rollout_engines)
            ]
            self._model_update_groups = init_process_group(
                backend="nccl",
                init_method=f"tcp://{master_address}:{master_port}",
                world_size=world_size,
                rank=0,
                group_name=self._group_name,
                # Fail fast instead of hanging forever if some rank never enters the collective.
                timeout=timedelta(seconds=_PIPELINE_RL_UPDATE_GROUP_TIMEOUT_S)
                if getattr(self.args, "pipeline_rl", False)
                else None,
            )
            ray.get(refs)

    def update_bucket_weights(self, named_tensors) -> None:
        """Send names/dtypes/shapes metadata to engines, then broadcast tensors.

        Ensures tensors are contiguous; when `world_size == 1`, converts DTensors
        to full tensors prior to `dist.broadcast`.
        """
        if not self._is_src_rank or not named_tensors:
            return

        handles = []
        if getattr(self.args, "pipeline_rl", False):
            named_tensors_by_dtype = {}
            for name, tensor in named_tensors:
                named_tensors_by_dtype.setdefault(tensor.dtype, []).append((name, tensor))

            for _, dtype_named_tensors in named_tensors_by_dtype.items():
                try:
                    refs = [
                        engine.update_weights_from_distributed.remote(
                            names=[name for name, _ in dtype_named_tensors],
                            dtypes=[param.dtype for _, param in dtype_named_tensors],
                            shapes=[param.shape for _, param in dtype_named_tensors],
                            group_name=self._group_name,
                            weight_version=str(self.weight_version),
                            load_format="flattened_bucket",
                        )
                        for engine in self.rollout_engines
                    ]
                    bucket = FlattenedTensorBucket(named_tensors=dtype_named_tensors)
                    flattened_tensor = bucket.get_flattened_tensor()
                    dist.broadcast(flattened_tensor, 0, group=self._model_update_groups)
                    results = ray.get(refs)
                    _raise_if_any_engine_failed(results, weight_version=self.weight_version)
                except Exception as e:
                    # Fail closed: if update fails or hangs, do not silently continue.
                    logger.exception(
                        f"PipelineRL update_weights_from_distributed failed (weight_version={self.weight_version})"
                    )
                    raise e
        else:
            refs = [
                engine.update_weights_from_distributed.remote(
                    names=[name for name, _ in named_tensors],
                    dtypes=[param.dtype for _, param in named_tensors],
                    shapes=[param.shape for _, param in named_tensors],
                    group_name=self._group_name,
                    weight_version=str(self.weight_version),
                )
                for engine in self.rollout_engines
            ]

            for _name, param in named_tensors:
                torch.cuda.empty_cache()
                # Ensure tensor is contiguous and on the right device
                param_data = param.data.contiguous()

                # avoid `DTensor._op_dispatcher.dispatch` has `assert compute_mesh is not None` error
                if dist.get_world_size() == 1 and isinstance(param_data, DTensor):
                    param_data = param_data.full_tensor()

                # Synchronous broadcast to avoid memory buildup
                handles.append(dist.broadcast(param_data, 0, group=self._model_update_groups, async_op=True))

            for handle in handles:
                handle.wait()
            ray.get(refs)
