"""
AllReduce configuration management for adaptive backend selection.

Supported backends:
1. flashinfer_fusion: FlashInfer fused allreduce + residual + rmsnorm
   - Source: python/sglang/srt/layers/flashinfer_comm_fusion.py
   - Kernel: flashinfer.comm.trtllm_allreduce_fusion

2. torch_symm_mem: PyTorch symmetric memory allreduce
   - Source: python/sglang/srt/distributed/device_communicators/torch_symm_mem.py
   - Kernel: torch.ops.symm_mem.multimem_all_reduce_ / two_shot_all_reduce_

3. custom_allreduce: SGLang custom allreduce kernel
   - Source: python/sglang/srt/distributed/device_communicators/custom_all_reduce.py
   - Kernel: sgl_kernel custom allreduce ops

4. nccl: Standard NCCL allreduce
   - Source: python/sglang/srt/distributed/parallel_state.py
   - Kernel: torch.distributed.all_reduce with NCCL backend
"""

import functools
import json
import logging
import os
from typing import Any, Dict, Optional

from sglang.srt.utils import get_device_name

logger = logging.getLogger(__name__)


class AllReduceBackendConfig:
    """Configuration for allreduce backend selection."""

    # Available backend types
    BACKEND_FLASHINFER_FUSION = "flashinfer_fusion"
    BACKEND_TORCH_SYMM_MEM = "torch_symm_mem"
    BACKEND_CUSTOM_ALLREDUCE = "custom_allreduce"
    BACKEND_NCCL = "nccl"

    def __init__(
        self,
        backend_type: str = BACKEND_NCCL,
        use_residual_rmsnorm_fusion: bool = False,
        backend_name: str = "auto",
    ):
        """
        Initialize allreduce backend configuration.

        Args:
            backend_type: Type of backend to use
            use_residual_rmsnorm_fusion: Whether to fuse with residual+rmsnorm (for flashinfer)
            backend_name: Human-readable name for identification
        """
        self.backend_type = backend_type
        self.use_residual_rmsnorm_fusion = use_residual_rmsnorm_fusion
        self.backend_name = backend_name

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "backend_type": self.backend_type,
            "use_residual_rmsnorm_fusion": self.use_residual_rmsnorm_fusion,
            "backend_name": self.backend_name,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "AllReduceBackendConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)

    def __repr__(self) -> str:
        return (
            f"AllReduceBackendConfig(backend_type={self.backend_type}, "
            f"use_residual_rmsnorm_fusion={self.use_residual_rmsnorm_fusion}, "
            f"backend_name={self.backend_name})"
        )


def get_all_backend_configs():
    """
    Get all possible backend configurations to test during tuning.

    Returns:
        List of AllReduceBackendConfig objects
    """
    configs = []

    # Config 1: FlashInfer fusion (fused allreduce + residual + rmsnorm)
    configs.append(
        AllReduceBackendConfig(
            backend_type=AllReduceBackendConfig.BACKEND_FLASHINFER_FUSION,
            use_residual_rmsnorm_fusion=True,
            backend_name="flashinfer_fusion",
        )
    )

    # Config 2: Torch symmetric memory
    configs.append(
        AllReduceBackendConfig(
            backend_type=AllReduceBackendConfig.BACKEND_TORCH_SYMM_MEM,
            use_residual_rmsnorm_fusion=False,
            backend_name="torch_symm_mem",
        )
    )

    # Config 3: Custom allreduce
    configs.append(
        AllReduceBackendConfig(
            backend_type=AllReduceBackendConfig.BACKEND_CUSTOM_ALLREDUCE,
            use_residual_rmsnorm_fusion=False,
            backend_name="custom_allreduce",
        )
    )

    # Config 4: Standard NCCL
    configs.append(
        AllReduceBackendConfig(
            backend_type=AllReduceBackendConfig.BACKEND_NCCL,
            use_residual_rmsnorm_fusion=False,
            backend_name="nccl",
        )
    )

    return configs


def get_config_file_name(hidden_size: int, tp_size: int) -> str:
    """
    Get configuration file name based on device, hidden size, and TP size.

    Args:
        hidden_size: Hidden dimension of the model
        tp_size: Tensor parallel size

    Returns:
        Configuration file name
    """
    device_name = get_device_name().replace(" ", "_")
    return (
        f"allreduce_config_hidden={hidden_size},tp={tp_size},device={device_name}.json"
    )


@functools.lru_cache(maxsize=None)
def get_allreduce_configs(
    hidden_size: int, tp_size: int
) -> Optional[Dict[int, AllReduceBackendConfig]]:
    """
    Load tuned allreduce configurations for the given hidden size and TP size.

    The return value will be a dictionary that maps batch sizes to configurations.
    To evaluate on a given batch size bs, the closest batch size in the grid should
    be picked and the associated configuration chosen.

    Args:
        hidden_size: Hidden dimension of the model
        tp_size: Tensor parallel size

    Returns:
        Dictionary mapping batch sizes to AllReduceBackendConfig, or None if not found
    """
    config_file_name = get_config_file_name(hidden_size, tp_size)

    config_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "configs")

    config_file_path = os.path.join(config_dir, config_file_name)

    if os.path.exists(config_file_path):
        with open(config_file_path) as f:
            logger.info(f"Using allreduce config from {config_file_path}")
            config_dict = json.load(f)
            # Convert dict to AllReduceBackendConfig objects
            return {
                int(batch_size): AllReduceBackendConfig.from_dict(config)
                for batch_size, config in config_dict.items()
            }

    logger.debug(
        f"Allreduce config file not found at {config_file_path}. "
        "Using default configuration."
    )
    return None


def get_default_allreduce_config(batch_size: int) -> AllReduceBackendConfig:
    """
    Get default allreduce configuration based on batch size heuristics.

    Based on empirical observations:
    - Small batch sizes (<=128): FlashInfer fusion is fastest
    - Large batch sizes (>128): torch_symm_mem or custom_allreduce is fastest

    Args:
        batch_size: Number of tokens

    Returns:
        Default AllReduceBackendConfig
    """
    if batch_size <= 128:
        return AllReduceBackendConfig(
            backend_type=AllReduceBackendConfig.BACKEND_FLASHINFER_FUSION,
            use_residual_rmsnorm_fusion=True,
            backend_name="flashinfer_fusion",
        )
    else:
        return AllReduceBackendConfig(
            backend_type=AllReduceBackendConfig.BACKEND_TORCH_SYMM_MEM,
            use_residual_rmsnorm_fusion=False,
            backend_name="torch_symm_mem",
        )


def select_allreduce_config(
    batch_size: int,
    hidden_size: int,
    tp_size: int,
) -> AllReduceBackendConfig:
    """
    Select optimal allreduce configuration for the given batch size, hidden size, and TP size.

    Args:
        batch_size: Number of tokens
        hidden_size: Hidden dimension
        tp_size: Tensor parallel size

    Returns:
        Selected AllReduceBackendConfig
    """
    # Try to load tuned configurations
    configs = get_allreduce_configs(hidden_size, tp_size)

    if configs:
        # Find the closest batch size in the config
        closest_batch_size = min(configs.keys(), key=lambda x: abs(x - batch_size))
        return configs[closest_batch_size]

    # Fall back to default heuristics
    return get_default_allreduce_config(batch_size)


def save_allreduce_configs(
    configs: Dict[int, AllReduceBackendConfig],
    hidden_size: int,
    tp_size: int,
    output_dir: Optional[str] = None,
) -> None:
    """
    Save tuned allreduce configurations to file.

    Args:
        configs: Dictionary mapping batch sizes to AllReduceBackendConfig
        hidden_size: Hidden dimension
        tp_size: Tensor parallel size
        output_dir: Output directory (optional)
    """
    # Always use the fixed path: python/sglang/srt/layers/allreduce/configs
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "configs"
        )

    os.makedirs(output_dir, exist_ok=True)

    config_file_name = get_config_file_name(hidden_size, tp_size)
    config_file_path = os.path.join(output_dir, config_file_name)

    # Convert AllReduceBackendConfig objects to dicts
    config_dict = {
        batch_size: config.to_dict() for batch_size, config in configs.items()
    }

    print(f"Saving allreduce configs to {config_file_path}...")
    with open(config_file_path, "w") as f:
        json.dump(config_dict, f, indent=4)
        f.write("\n")

    print(f"Configs saved successfully!")
