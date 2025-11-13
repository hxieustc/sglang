"""
Adaptive AllReduce layer that selects optimal backend based on tuned configurations.

This module provides a unified interface for different allreduce backends:
1. FlashInfer fusion: flashinfer.comm.trtllm_allreduce_fusion
   Source: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/flashinfer_comm_fusion.py

2. Torch Symmetric Memory: torch.ops.symm_mem.multimem_all_reduce_
   Source: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/distributed/device_communicators/torch_symm_mem.py

3. Custom AllReduce: sgl_kernel custom allreduce
   Source: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/distributed/device_communicators/custom_all_reduce.py

4. Standard NCCL: torch.distributed.all_reduce
   Source: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/distributed/parallel_state.py
"""

import logging
from typing import Optional, Tuple

import torch

from sglang.srt.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.layers.allreduce.config import (
    AllReduceBackendConfig,
    select_allreduce_config,
)
from sglang.srt.utils import is_flashinfer_available

logger = logging.getLogger(__name__)

# Try to import FlashInfer
_flashinfer_comm = None
if is_flashinfer_available():
    try:
        import flashinfer.comm as _flashinfer_comm
    except ImportError:
        _flashinfer_comm = None

# Try to import torch symmetric memory
_torch_symm_mem_available = False
try:
    from sglang.srt.distributed.device_communicators.torch_symm_mem import (
        TorchSymmMemCommunicator,
    )

    _torch_symm_mem_available = True
except ImportError:
    _torch_symm_mem_available = False

# Try to import custom allreduce
_custom_allreduce_available = False
try:
    from sglang.srt.distributed.device_communicators.custom_all_reduce import (
        CustomAllreduce,
    )

    _custom_allreduce_available = True
except ImportError:
    _custom_allreduce_available = False


class AdaptiveAllReduceLayer:
    """
    Adaptive AllReduce layer that selects optimal backend based on batch size and configuration.

    This layer can be used in model forward pass to automatically select the best allreduce
    implementation based on runtime conditions and tuned configurations.
    """

    def __init__(
        self,
        hidden_size: int,
        enable_adaptive_allreduce: bool = False,
        flashinfer_workspace_tensor: Optional[torch.Tensor] = None,
        torch_symm_mem_communicator: Optional["TorchSymmMemCommunicator"] = None,
        custom_allreduce: Optional["CustomAllreduce"] = None,
    ):
        """
        Initialize adaptive allreduce layer.

        Args:
            hidden_size: Hidden dimension
            enable_adaptive_allreduce: Whether to enable adaptive backend selection
            flashinfer_workspace_tensor: FlashInfer workspace (if using flashinfer)
            torch_symm_mem_communicator: Torch symmetric memory communicator
            custom_allreduce: Custom allreduce communicator
        """
        self.hidden_size = hidden_size
        self.enable_adaptive_allreduce = enable_adaptive_allreduce
        self.world_size = get_tensor_model_parallel_world_size()

        # Store communicators
        self.flashinfer_workspace_tensor = flashinfer_workspace_tensor
        self.torch_symm_mem_communicator = torch_symm_mem_communicator
        self.custom_allreduce = custom_allreduce

        # Check availability
        self.flashinfer_available = (
            _flashinfer_comm is not None and flashinfer_workspace_tensor is not None
        )
        self.torch_symm_mem_available = (
            _torch_symm_mem_available
            and torch_symm_mem_communicator is not None
            and not torch_symm_mem_communicator.disabled
        )
        self.custom_allreduce_available = (
            _custom_allreduce_available
            and custom_allreduce is not None
            and not custom_allreduce.disabled
        )

        logger.info(
            f"AdaptiveAllReduceLayer initialized: "
            f"flashinfer={self.flashinfer_available}, "
            f"torch_symm_mem={self.torch_symm_mem_available}, "
            f"custom_allreduce={self.custom_allreduce_available}, "
            f"adaptive={enable_adaptive_allreduce}"
        )

    def select_backend_and_allreduce(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        layernorm: Optional[torch.nn.Module] = None,
        enable_cross_layer_fusion: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Select optimal backend and perform allreduce + layernorm.

        Args:
            hidden_states: Input hidden states
            residual: Residual tensor (optional)
            layernorm: Layer normalization module (optional)
            enable_cross_layer_fusion: If True and flashinfer_fusion is selected,
                                      set flag for cross-layer fusion instead of executing immediately

        Returns:
            Tuple of (output_hidden_states, output_residual)
        """
        # Single GPU case
        if self.world_size <= 1:
            if residual is not None and layernorm is not None:
                return layernorm(hidden_states, residual)
            elif layernorm is not None:
                return layernorm(hidden_states), None
            return hidden_states, residual

        batch_size = hidden_states.shape[0]

        # Select backend based on configuration
        if self.enable_adaptive_allreduce:
            config = select_allreduce_config(
                batch_size=batch_size,
                hidden_size=self.hidden_size,
                tp_size=self.world_size,
            )
            logger.debug(f"Selected backend for bs={batch_size}: {config.backend_name}")
        else:
            # Fall back to default NCCL
            config = AllReduceBackendConfig(
                backend_type=AllReduceBackendConfig.BACKEND_NCCL,
                backend_name="nccl",
            )

        # Execute based on selected backend
        if config.backend_type == AllReduceBackendConfig.BACKEND_FLASHINFER_FUSION:
            # If cross-layer fusion is enabled and conditions are met, set flag instead of executing
            if (
                enable_cross_layer_fusion
                and self.flashinfer_available
                and config.use_residual_rmsnorm_fusion
                and residual is not None
                and layernorm is not None
            ):
                # Set flag for cross-layer fusion (allreduce in current layer, rmsnorm in next layer)
                hidden_states._sglang_needs_allreduce_fusion = True
                return hidden_states, residual
            else:
                # Execute fusion immediately
                return self._flashinfer_fusion_allreduce(
                    hidden_states, residual, layernorm, config
                )
        elif config.backend_type == AllReduceBackendConfig.BACKEND_TORCH_SYMM_MEM:
            return self._torch_symm_mem_allreduce(
                hidden_states, residual, layernorm, config
            )
        elif config.backend_type == AllReduceBackendConfig.BACKEND_CUSTOM_ALLREDUCE:
            return self._custom_allreduce_allreduce(
                hidden_states, residual, layernorm, config
            )
        else:  # NCCL
            return self._nccl_allreduce(hidden_states, residual, layernorm, config)

    def _flashinfer_fusion_allreduce(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        layernorm: Optional[torch.nn.Module],
        config: AllReduceBackendConfig,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        FlashInfer fused allreduce + residual + rmsnorm.

        Source: python/sglang/srt/layers/flashinfer_comm_fusion.py
        Kernel: flashinfer.comm.trtllm_allreduce_fusion
        """
        if not self.flashinfer_available:
            logger.warning("FlashInfer not available, falling back to NCCL")
            return self._nccl_allreduce(hidden_states, residual, layernorm, config)

        if (
            not config.use_residual_rmsnorm_fusion
            or residual is None
            or layernorm is None
        ):
            # Can't use fusion without residual and layernorm
            return self._nccl_allreduce(hidden_states, residual, layernorm, config)

        try:
            import torch.distributed as dist

            batch_size = hidden_states.shape[0]

            # Create output tensors
            norm_out = torch.empty_like(hidden_states)
            residual_out = torch.empty_like(residual)

            # Call FlashInfer fused kernel
            _flashinfer_comm.trtllm_allreduce_fusion(
                allreduce_in=hidden_states,
                world_size=self.world_size,
                world_rank=dist.get_rank(),
                token_num=batch_size,
                hidden_dim=self.hidden_size,
                workspace_ptrs=self.flashinfer_workspace_tensor,
                launch_with_pdl=True,
                use_oneshot=True,
                trigger_completion_at_end=False,
                fp32_acc=True,
                pattern_code=_flashinfer_comm.AllReduceFusionPattern.kARResidualRMSNorm,
                allreduce_out=None,
                residual_in=residual,
                residual_out=residual_out,
                norm_out=norm_out,
                quant_out=None,
                scale_out=None,
                rms_gamma=layernorm.weight,
                rms_eps=layernorm.variance_epsilon,
                scale_factor=None,
                layout_code=None,
            )

            return norm_out, residual_out
        except Exception as e:
            logger.warning(f"FlashInfer fusion failed: {e}, falling back to NCCL")
            return self._nccl_allreduce(hidden_states, residual, layernorm, config)

    def _torch_symm_mem_allreduce(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        layernorm: Optional[torch.nn.Module],
        config: AllReduceBackendConfig,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Torch symmetric memory allreduce.

        Source: python/sglang/srt/distributed/device_communicators/torch_symm_mem.py
        Kernel: torch.ops.symm_mem.multimem_all_reduce_ or two_shot_all_reduce_
        """
        if not self.torch_symm_mem_available:
            logger.warning("Torch symmetric memory not available, falling back to NCCL")
            return self._nccl_allreduce(hidden_states, residual, layernorm, config)

        try:
            # Check if tensor is eligible for symmetric memory allreduce
            if not self.torch_symm_mem_communicator.should_torch_symm_mem_allreduce(
                hidden_states
            ):
                return self._nccl_allreduce(hidden_states, residual, layernorm, config)

            # Perform allreduce
            output = self.torch_symm_mem_communicator.all_reduce(hidden_states)

            # Apply layernorm (layernorm will handle residual internally)
            if layernorm is not None:
                return layernorm(output, residual)

            return output, residual
        except Exception as e:
            logger.warning(
                f"Torch symmetric memory allreduce failed: {e}, falling back to NCCL"
            )
            return self._nccl_allreduce(hidden_states, residual, layernorm, config)

    def _custom_allreduce_allreduce(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        layernorm: Optional[torch.nn.Module],
        config: AllReduceBackendConfig,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Custom allreduce using sgl_kernel.

        Source: python/sglang/srt/distributed/device_communicators/custom_all_reduce.py
        Kernel: sgl_kernel custom allreduce ops
        """
        if not self.custom_allreduce_available:
            logger.warning("Custom allreduce not available, falling back to NCCL")
            return self._nccl_allreduce(hidden_states, residual, layernorm, config)

        try:
            # Check if tensor is eligible for custom allreduce
            output = self.custom_allreduce.custom_all_reduce(hidden_states)

            if output is None:
                # Custom allreduce declined, fall back to NCCL
                return self._nccl_allreduce(hidden_states, residual, layernorm, config)

            # Apply layernorm (layernorm will handle residual internally)
            if layernorm is not None:
                return layernorm(output, residual)

            return output, residual
        except Exception as e:
            logger.warning(f"Custom allreduce failed: {e}, falling back to NCCL")
            return self._nccl_allreduce(hidden_states, residual, layernorm, config)

    def _nccl_allreduce(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        layernorm: Optional[torch.nn.Module],
        config: AllReduceBackendConfig,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Standard NCCL allreduce.

        Source: python/sglang/srt/distributed/parallel_state.py
        Kernel: torch.distributed.all_reduce with NCCL backend
        """
        # Perform allreduce
        output = tensor_model_parallel_all_reduce(hidden_states)

        # Apply layernorm (layernorm will handle residual internally)
        if layernorm is not None:
            return layernorm(output, residual)

        return output, residual
