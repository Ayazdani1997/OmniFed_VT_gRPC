# Copyright (c) 2025, Oak Ridge National Laboratory.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
import time
import warnings
from enum import Enum
from typing import Optional

import rich.repr
import torch
import torch.distributed as dist
from torch import nn

from ..utils import print
from .base import AggregationOp, BaseCommunicator, AggregationMetric
from .utils import get_msg_info
from .utils import get_class_from_str
from .compression import (
    Compression,
    layerwise_decompress,
)
from .compression.sparsification import _sparse_compression_
from .compression.quantization import _quantized_compression_
from .compression.lowrank_approximation import _lora_compression_


class InitMethod(str, Enum):
    """Initialization methods for PyTorch distributed process groups."""

    TCP = "tcp"  # TCP-based initialization for network communication
    FILE = "file"  # File-based initialization for shared filesystem


# ======================================================================================


@rich.repr.auto
class TorchDistCommunicator(BaseCommunicator):
    """
    Communication backend using PyTorch distributed collective operations.

    Implements broadcast and aggregation via efficient collective
    primitives (broadcast, all-reduce).
    Uses process groups for coordination with support for multiple backends
    (gloo, nccl) and initialization methods.
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        master_addr: str = "127.0.0.1",
        master_port: int = 29500,
        init_method: InitMethod = InitMethod.TCP,
        backend: str = "gloo",
        sharedfile: str = "sharedfile",
        timeout: int = 60,
        max_retries: int = 5,
        compressor: Optional[str] = None,
    ) -> None:
        """
        Initialize PyTorch distributed communicator.

        Args:
            rank: Process rank in distributed group
            world_size: Total number of processes in distributed group
            master_addr: Master node address for coordination
            master_port: Master node port for coordination
            init_method: TCP or file-based process group initialization
            backend: Communication backend ('gloo' for CPU, 'nccl' for GPU)
            sharedfile: Shared file path for file-based initialization
            timeout: Process group initialization timeout (seconds)
            max_retries: Maximum initialization retry attempts
        """
        super().__init__(rank, world_size, master_addr, master_port)
        print(
            f"rank={rank}/{world_size} | backend={backend} | addr={master_addr}:{master_port}"
        )

        # Core distributed parameters
        # self.rank: int = rank
        # self.world_size: int = world_size
        # self.master_addr: str = master_addr
        # self.master_port: int = master_port

        # ---
        self.init_method: str = init_method
        self.backend: str = backend
        self.sharedfile: str = sharedfile
        self.timeout: datetime.timedelta = datetime.timedelta(seconds=timeout)
        self.max_retries: int = max_retries
        if compressor is not None:
            compressor_class = get_class_from_str(compressor)
            self.compressor: Compression = compressor_class()
            self.compressor_name = self.compressor.__class__.__name__
            print(f"going to use {self.compressor_name} compression scheme")
        else:
            self.compressor_name = None

        # Backend validation with automatic fallback
        if self.backend == "nccl" and not torch.cuda.is_available():
            print("NCCL→gloo fallback (no CUDA available)")
            self.backend = "gloo"

    def _setup(self):
        """
        Initialize PyTorch distributed process group.

        Creates the distributed process group using the specified initialization
        method and backend.

        All ranks must call this before any collective operations.
        """

        print(
            f"rank={self.rank}/{self.world_size} | {self.init_method} | backend={self.backend}"
        )

        match self.init_method:
            case InitMethod.TCP:
                addr = f"tcp://{self.master_addr}:{self.master_port}"
                dist.init_process_group(
                    backend=self.backend,
                    init_method=addr,
                    rank=self.rank,
                    world_size=self.world_size,
                    timeout=self.timeout,
                )
            case InitMethod.FILE:
                addr = f"file://{self.sharedfile}"
                dist.init_process_group(
                    backend=self.backend,
                    init_method=addr,
                    rank=self.rank,
                    world_size=self.world_size,
                    timeout=self.timeout,
                )
            case _:
                raise ValueError(
                    f"Unknown init_method: {self.init_method}. Supported: {[m.value for m in InitMethod]}"
                )

    def broadcast(
        self,
        msg: BaseCommunicator.MsgT,
        src: int = 0,
        # TODO: Consider adding control arguments like utils.scale_params:
        # requires_grad: Optional[bool] = None,
        # include_buffers: bool = True,
        # filter_fn: Optional[Callable[[str, torch.Tensor], bool]] = None
    ) -> BaseCommunicator.MsgT:
        """
        Broadcast message from source rank to all other ranks.

        Uses PyTorch's distributed broadcast collective for efficient
        one-to-many communication within the process group.

        Args:
            msg: Model, tensor dict, or tensor to broadcast
            src: Source rank ID (default: 0)

        Returns:
            Message with broadcasted values
        """
        print(f"{get_msg_info(msg)} | src={src}")

        if isinstance(msg, nn.Module):
            # Broadcast all trainable parameters
            for _, p in msg.named_parameters():
                # Only broadcast trainable parameters (frozen params stay local)
                if p.requires_grad:
                    dist.broadcast(p.data, src=src)
            # Broadcast all buffers (batch norm stats, etc.) - only floating-point buffers
            for name, buffer in msg.named_buffers():
                if buffer is None:  # type: ignore
                    warnings.warn(f"Buffer '{name}' is None, skipping broadcast")
                    continue
                if not buffer.dtype.is_floating_point:
                    continue  # Skip integer buffers like num_batches_tracked
                dist.broadcast(buffer.data, src=src)
        elif isinstance(msg, dict):
            # Broadcast each tensor in dictionary
            for tensor in msg.values():
                dist.broadcast(tensor, src=src)
        else:
            # Broadcast single tensor
            dist.broadcast(msg, src=src)
        return msg

    def aggregate(
        self,
        msg: BaseCommunicator.MsgT,
        reduction: AggregationOp,
        reduction_type: Optional[AggregationMetric] = AggregationMetric.PARAMETER,
        # TODO: Consider adding control arguments like utils.scale_params:
        # requires_grad: Optional[bool] = None,
        # include_buffers: bool = True,
        # filter_fn: Optional[Callable[[str, torch.Tensor], bool]] = None
    ) -> BaseCommunicator.MsgT:
        print(
            f"{get_msg_info(msg)} | reduction={reduction} | reduction_type={reduction_type}"
        )

        # Map reduction type to PyTorch operation
        reduction_ops = {
            AggregationOp.SUM: dist.ReduceOp.SUM,
            AggregationOp.MEAN: dist.ReduceOp.AVG,
            AggregationOp.MAX: dist.ReduceOp.MAX,
        }

        if reduction not in reduction_ops:
            raise ValueError(f"Unsupported reduction type: {reduction}")

        reduction_op = reduction_ops[reduction]

        reduction_types = {
            AggregationMetric.GRADIENT: "grad",
            AggregationMetric.PARAMETER: "param",
        }

        if reduction_type not in reduction_types:
            raise ValueError(f"Unsupported reduction_type metric: {reduction_type}")

        aggregate_metric_op = reduction_types[reduction_type]

        if self.compressor_name is not None:
            print(
                f"compressor_name {self.compressor_name} and _sparse_compression_ {_sparse_compression_} "
                f"and compressor_class {self.compressor_name}"
            )

        if isinstance(msg, nn.Module):
            if self.compressor_name in _sparse_compression_:
                print(f"using sparse compression {self.compressor_name}")
                agg_model = self._sparse_aggregate_(
                    msg=msg, aggregation_metric=aggregate_metric_op
                )
                print("@@@@@@@@@@@@@@@@@@@@@  aggregated sparse model!!!!!")
            elif self.compressor_name in _quantized_compression_:
                print(f"using quantized compression {self.compressor_name}")
                agg_model = self._quantize_aggregate_(
                    msg=msg, reduction_op=reduction_op
                )
                print("#####################  aggregated quantized model!!!!!")
            elif self.compressor_name in _lora_compression_:
                print(f"using lora compression {self.compressor_name}")
                agg_model = self._lora_aggregate_(msg=msg, reduction_op=reduction_op)
                print("$$$$$$$$$$$$$$$$$$$$$  aggregated low-rank model!!!!!")
            else:
                print(f"using default aggregation operation!!")
                agg_model = self._default_aggregate_(msg=msg, reduction_op=reduction_op)

            return agg_model
        else:
            msg = self._default_aggregate_(msg=msg, reduction_op=reduction_op)
            return msg

    def _default_aggregate_(
        self,
        msg: BaseCommunicator.MsgT,
        reduction_op: dist.ReduceOp,
        aggregation_metric: str = "param",
    ):
        """
        Aggregate message across all ranks using PyTorch all-reduce collective.

        Performs efficient element-wise reduction across all process ranks.

        Args:
            msg: Model, tensor dict, or tensor to aggregate
            reduction: SUM, MEAN, or MAX reduction operation

        Returns:
            Message with aggregated values distributed to all ranks
        """
        if isinstance(msg, nn.Module):
            # Aggregate all trainable parameters
            for _, p in msg.named_parameters():
                if p.requires_grad:
                    if aggregation_metric == "grad":
                        dist.all_reduce(p.grad, op=reduction_op)
                    elif aggregation_metric == "param":
                        dist.all_reduce(p.data, op=reduction_op)
            # Aggregate all buffers (batch norm stats, etc.) - only floating-point buffers
            for name, buffer in msg.named_buffers():
                if buffer is None:  # type: ignore
                    warnings.warn(f"Buffer '{name}' is None, skipping aggregation")
                    continue
                if not buffer.dtype.is_floating_point:
                    continue  # Skip integer buffers like num_batches_tracked

                if aggregation_metric == "grad":
                    dist.all_reduce(buffer.grad, op=reduction_op)
                elif aggregation_metric == "param":
                    dist.all_reduce(buffer.data, op=reduction_op)
        elif isinstance(msg, dict):
            # Aggregate each tensor in dictionary
            for tensor in msg.values():
                dist.all_reduce(tensor, op=reduction_op)
        else:
            # Aggregate single tensor
            dist.all_reduce(msg, op=reduction_op)

        return msg

    def _sparse_aggregate_(
        self, msg: BaseCommunicator.MsgT, aggregation_metric: str = "grad"
    ):
        if isinstance(msg, nn.Module):
            layerwise_vals, layerwise_ixs = [], []
            with torch.no_grad():
                for name, param in msg.named_parameters():
                    if aggregation_metric == "grad":
                        compressed_tnsr, _ = self.compressor.compress(
                            tensor=param.grad, name=name
                        )
                    elif aggregation_metric == "param":
                        compressed_tnsr, _ = self.compressor.compress(
                            tensor=param.data, name=name
                        )
                    else:
                        raise ValueError(
                            f"Unsupported aggregation_metric: {aggregation_metric} in sparse compression"
                        )

                    layerwise_vals.append(compressed_tnsr[0])
                    layerwise_ixs.append(compressed_tnsr[1])

            for param, update_val, update_ix in zip(
                msg.parameters(), layerwise_vals, layerwise_ixs
            ):
                # update_val = update_val.to(device)
                # update_ix = update_ix.to(device)
                tensor_sizes = [torch.LongTensor([0]) for _ in range(self.world_size)]
                tensor_size = update_val.numel()
                dist.all_gather(tensor_sizes, torch.LongTensor([tensor_size]))

                tensor_list = []
                ix_list = []
                size_list = [int(size.item()) for size in tensor_sizes]
                max_size = max(size_list)
                if max_size > 0:
                    for _ in size_list:
                        tensor_list.append(
                            torch.zeros(
                                size=(max_size,),
                                dtype=torch.float32,
                                # device=device
                            )
                        )
                        ix_list.append(
                            torch.zeros(
                                size=(max_size,),
                                dtype=torch.long,
                                # device=device
                            )
                        )

                    if tensor_size != max_size:
                        g_padding = torch.zeros(
                            size=(max_size - tensor_size,),
                            dtype=torch.float32,
                            # device=device,
                        )
                        ix_padding = torch.zeros(
                            size=(max_size - tensor_size,),
                            dtype=torch.long,
                            # device=device,
                        )
                        update_val = torch.cat((update_val, g_padding), dim=0)
                        update_ix = torch.cat((update_ix, ix_padding), dim=0)

                    dist.all_gather(tensor_list, update_val)
                    dist.all_gather(ix_list, update_ix)

                    if aggregation_metric == "grad":
                        param.grad = layerwise_decompress(
                            collected_vals=tensor_list,
                            collected_ix=ix_list,
                            tensor_shape=param.shape,
                            client_count=self.world_size,
                            # device=device,
                        )
                    else:
                        param.data = layerwise_decompress(
                            collected_vals=tensor_list,
                            collected_ix=ix_list,
                            tensor_shape=param.shape,
                            client_count=self.world_size,
                            # device=device,
                        )
            return msg
        else:
            raise ValueError(
                f"Error in compression op. msg in communicator's aggregate function "
                f"must be of type nn.Module"
            )

    def _quantize_aggregate_(
        self,
        msg: BaseCommunicator.MsgT,
        reduction_op: dist.ReduceOp,
        aggregation_metric: str = "grad",
    ):
        if isinstance(msg, nn.Module):
            if aggregation_metric == "grad":
                # QSGD: Quantize and aggregate gradients
                updates = [p.grad for p in msg.parameters() if p.requires_grad]
            else:
                # if quantization metric is model parameters
                updates = [p.data for p in msg.parameters() if p.requires_grad]

            # Apply QSGD quantization to each gradient
            quantized_grads, norms = self.compressor.compress(updates)
            for param_idx, param in enumerate(msg.parameters()):
                if param.requires_grad:
                    if aggregation_metric == "grad":
                        # Replace gradient with quantized version
                        param.grad.data = quantized_grads[param_idx]
                    elif aggregation_metric == "param":
                        param.data = quantized_grads[param_idx]

            agg_model = self._default_aggregate_(msg=msg, reduction_op=reduction_op)

            return agg_model
        else:
            raise ValueError(
                f"Error in compression op. msg in communicator's aggregate function "
                f"must be of type nn.Module"
            )

    def _lora_aggregate_(
        self,
        msg: BaseCommunicator.MsgT,
        reduction_op: dist.ReduceOp,
        aggregation_metric: str = "grad",
    ):
        if isinstance(msg, nn.Module):
            with torch.no_grad():
                for name, param in msg.named_parameters():
                    if aggregation_metric == "grad":
                        original_update = param.grad.clone()
                        param_P, matrix, param_og_shape, was_compressed = (
                            self.compressor.compress(tensor=param.grad, param_name=name)
                        )
                    else:
                        original_update = param.data.clone()
                        param_P, matrix, param_og_shape, was_compressed = (
                            self.compressor.compress(tensor=param.data, param_name=name)
                        )

                    param_P = dist.all_reduce(tensor=param_P, op=reduction_op)

                    if was_compressed:
                        param_Q = self.compressor._update_Q(P=param_P, matrix=matrix)
                        decompressed_update = self.compressor.decompress(
                            P=param_P,
                            Q=param_Q,
                            original_shape=param_og_shape,
                            was_compressed=was_compressed,
                        )

                    else:
                        decompressed_update = param_P

                    self.compressor.update_error_feedback(
                        original_update=original_update,
                        compressed_update=decompressed_update,
                        param_name=name,
                    )

                    if aggregation_metric == "grad":
                        param.grad.copy_(decompressed_update)
                    else:
                        param.data.copy_(decompressed_update)

            return msg
        else:
            raise ValueError(
                f"Error in compression op. msg in communicator's aggregate function "
                f"must be of type nn.Module"
            )

    def close(self):
        """
        Destroy PyTorch distributed process group and clean up resources.

        Should be called when distributed communication is no longer needed.
        All ranks must call this to properly clean up the process group.
        """
        print()
        dist.destroy_process_group()
