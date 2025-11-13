"""
Tuning script for allreduce configurations.

This script tests different allreduce backend configurations across different batch sizes
to find the optimal configuration for each batch size.

Supported backends and their source code locations:
1. flashinfer_fusion: FlashInfer fused allreduce + residual + rmsnorm
   - Source: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/flashinfer_comm_fusion.py
   - Kernel: flashinfer.comm.trtllm_allreduce_fusion

2. torch_symm_mem: PyTorch symmetric memory allreduce
   - Source: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/distributed/device_communicators/torch_symm_mem.py
   - Kernel: torch.ops.symm_mem.multimem_all_reduce_ / two_shot_all_reduce_

3. custom_allreduce: SGLang custom allreduce kernel
   - Source: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/distributed/device_communicators/custom_all_reduce.py
   - Kernel: sgl_kernel custom allreduce ops

4. nccl: Standard NCCL allreduce
   - Source: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/distributed/parallel_state.py
   - Kernel: torch.distributed.all_reduce with NCCL backend

Usage:
    # Tune for a specific model
    torchrun --nproc_per_node=2 python/sglang/srt/layers/allreduce/tuning_allreduce_config.py \
        --model nvidia/DeepSeek-V3-0324-FP4 --tp-size 2

    # Tune for custom hidden size
    torchrun --nproc_per_node=2 python/sglang/srt/layers/allreduce/tuning_allreduce_config.py \
        --hidden-size 7168 --tp-size 2
"""

import argparse
import logging
import os
import time
from datetime import datetime
from typing import List, Optional

import torch

from sglang.srt.distributed import (
    get_tp_group,
    initialize_model_parallel,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.distributed.device_communicators.custom_all_reduce import (
    CustomAllreduce,
)
from sglang.srt.distributed.device_communicators.torch_symm_mem import (
    TorchSymmMemCommunicator,
)
from sglang.srt.distributed.parallel_state import cleanup_dist_env_and_memory
from sglang.srt.layers.allreduce.config import (
    AllReduceBackendConfig,
    get_all_backend_configs,
    save_allreduce_configs,
)
from sglang.srt.layers.layernorm import RMSNorm

logger = logging.getLogger(__name__)

# Try to import FlashInfer
try:
    import flashinfer.comm as flashinfer_comm

    _flashinfer_available = hasattr(flashinfer_comm, "trtllm_allreduce_fusion")
except ImportError:
    flashinfer_comm = None
    _flashinfer_available = False


# Global communicators
_FI_WORKSPACE_TENSOR = None
_FI_IPC_HANDLES = None
_TORCH_SYMM_MEM_COMM = None
_CUSTOM_ALLREDUCE = None


def setup_communicators(
    world_size: int,
    rank: int,
    hidden_size: int,
    max_batch_size: int,
    device: torch.device,
):
    """Setup all available communicators."""
    global _FI_WORKSPACE_TENSOR, _FI_IPC_HANDLES, _TORCH_SYMM_MEM_COMM, _CUSTOM_ALLREDUCE

    # Setup FlashInfer workspace
    if _flashinfer_available and flashinfer_comm is not None:
        try:
            _FI_IPC_HANDLES, _FI_WORKSPACE_TENSOR = (
                flashinfer_comm.trtllm_create_ipc_workspace_for_all_reduce_fusion(
                    tp_rank=rank,
                    tp_size=world_size,
                    max_token_num=max_batch_size,
                    hidden_dim=hidden_size,
                    group=get_tp_group().device_group,
                    use_fp32_lamport=False,
                )
            )
            logger.info("FlashInfer workspace initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to setup FlashInfer workspace: {e}")
            _FI_WORKSPACE_TENSOR = None

    # Setup Torch Symmetric Memory Communicator
    try:
        _TORCH_SYMM_MEM_COMM = TorchSymmMemCommunicator(
            group=get_tp_group().device_group,
            device=device,
        )
        if not _TORCH_SYMM_MEM_COMM.disabled:
            logger.info("Torch symmetric memory communicator initialized successfully")
        else:
            logger.warning("Torch symmetric memory communicator is disabled")
    except Exception as e:
        logger.warning(f"Failed to setup torch symmetric memory communicator: {e}")
        _TORCH_SYMM_MEM_COMM = None

    # Setup Custom AllReduce
    try:
        _CUSTOM_ALLREDUCE = CustomAllreduce(
            group=get_tp_group().cpu_group,
            device=device,
        )
        if not _CUSTOM_ALLREDUCE.disabled:
            logger.info("Custom allreduce initialized successfully")
        else:
            logger.warning("Custom allreduce is disabled")
    except Exception as e:
        logger.warning(f"Failed to setup custom allreduce: {e}")
        _CUSTOM_ALLREDUCE = None


def cleanup_communicators():
    """Cleanup all communicators."""
    global _FI_WORKSPACE_TENSOR, _FI_IPC_HANDLES, _TORCH_SYMM_MEM_COMM, _CUSTOM_ALLREDUCE

    if _flashinfer_available and _FI_IPC_HANDLES is not None:
        try:
            group = get_tp_group().device_group
            flashinfer_comm.trtllm_destroy_ipc_workspace_for_all_reduce(
                _FI_IPC_HANDLES, group
            )
        except Exception as e:
            logger.error(f"Failed to cleanup FlashInfer workspace: {e}")

    if _CUSTOM_ALLREDUCE is not None:
        try:
            _CUSTOM_ALLREDUCE.close()
        except Exception as e:
            logger.error(f"Failed to cleanup custom allreduce: {e}")

    _FI_WORKSPACE_TENSOR = None
    _FI_IPC_HANDLES = None
    _TORCH_SYMM_MEM_COMM = None
    _CUSTOM_ALLREDUCE = None


def benchmark_allreduce_config(
    config: AllReduceBackendConfig,
    batch_size: int,
    hidden_size: int,
    world_size: int,
    rank: int,
    num_iters: int = 100,
    warmup_iters: int = 10,
) -> Optional[float]:
    """
    Benchmark a specific allreduce configuration.

    Returns:
        Average latency in microseconds, or None if config is not supported
    """
    try:
        # Create input tensors
        input_tensor = torch.randn(
            batch_size, hidden_size, dtype=torch.bfloat16, device="cuda"
        )
        residual_tensor = torch.randn(
            batch_size, hidden_size, dtype=torch.bfloat16, device="cuda"
        )

        # Create RMSNorm layer
        rms_norm = RMSNorm(hidden_size, eps=1e-6).cuda()

        def run_iteration():
            """Run one iteration of the allreduce operation."""
            nonlocal input_tensor, residual_tensor
            if config.backend_type == AllReduceBackendConfig.BACKEND_FLASHINFER_FUSION:
                # FlashInfer fused allreduce + residual + rmsnorm
                # Source: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/flashinfer_comm_fusion.py
                if (
                    _FI_WORKSPACE_TENSOR is None
                    or not config.use_residual_rmsnorm_fusion
                ):
                    return False

                norm_out = torch.empty_like(input_tensor)
                residual_out = torch.empty_like(residual_tensor)

                flashinfer_comm.trtllm_allreduce_fusion(
                    allreduce_in=input_tensor,
                    world_size=world_size,
                    world_rank=rank,
                    token_num=batch_size,
                    hidden_dim=hidden_size,
                    workspace_ptrs=_FI_WORKSPACE_TENSOR,
                    launch_with_pdl=True,
                    use_oneshot=True,
                    trigger_completion_at_end=False,
                    fp32_acc=True,
                    pattern_code=flashinfer_comm.AllReduceFusionPattern.kARResidualRMSNorm,
                    allreduce_out=None,
                    residual_in=residual_tensor,
                    residual_out=residual_out,
                    norm_out=norm_out,
                    quant_out=None,
                    scale_out=None,
                    rms_gamma=rms_norm.weight,
                    rms_eps=rms_norm.variance_epsilon,
                    scale_factor=None,
                    layout_code=None,
                )

            elif config.backend_type == AllReduceBackendConfig.BACKEND_TORCH_SYMM_MEM:
                # Torch symmetric memory allreduce
                # Source: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/distributed/device_communicators/torch_symm_mem.py
                if _TORCH_SYMM_MEM_COMM is None or _TORCH_SYMM_MEM_COMM.disabled:
                    return False

                if not _TORCH_SYMM_MEM_COMM.should_torch_symm_mem_allreduce(
                    input_tensor
                ):
                    return False

                output = _TORCH_SYMM_MEM_COMM.all_reduce(input_tensor)
                # Use fused residual + rmsnorm kernel
                output, _ = rms_norm(output, residual_tensor)

            elif config.backend_type == AllReduceBackendConfig.BACKEND_CUSTOM_ALLREDUCE:
                # Custom allreduce
                # Source: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/distributed/device_communicators/custom_all_reduce.py
                if _CUSTOM_ALLREDUCE is None or _CUSTOM_ALLREDUCE.disabled:
                    return False

                output = _CUSTOM_ALLREDUCE.custom_all_reduce(input_tensor)
                if output is None:
                    return False

                # Use fused residual + rmsnorm kernel
                output, _ = rms_norm(output, residual_tensor)

            else:  # NCCL
                # Standard NCCL allreduce
                # Source: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/distributed/parallel_state.py
                output = tensor_model_parallel_all_reduce(input_tensor)
                # Use fused residual + rmsnorm kernel
                output, _ = rms_norm(output, residual_tensor)

            return True

        # Warmup
        success_count = 0
        for _ in range(warmup_iters):
            if run_iteration():
                success_count += 1

        if success_count == 0:
            return None

        torch.cuda.synchronize()

        # Benchmark
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]

        success_count = 0
        for i in range(num_iters):
            start_events[i].record()
            if run_iteration():
                success_count += 1
            end_events[i].record()

        if success_count == 0:
            return None

        torch.cuda.synchronize()

        # Calculate average latency
        latencies = []
        for i in range(num_iters):
            if i < success_count:
                latencies.append(
                    start_events[i].elapsed_time(end_events[i]) * 1000
                )  # us

        if not latencies:
            return None

        avg_latency = sum(latencies) / len(latencies)

        # Cleanup
        del input_tensor, residual_tensor
        torch.cuda.empty_cache()

        return avg_latency

    except Exception as e:
        logger.warning(f"Failed to benchmark config {config.backend_name}: {e}")
        return None


def tune_batch_size(
    batch_size: int,
    hidden_size: int,
    world_size: int,
    rank: int,
    configs: List[AllReduceBackendConfig],
) -> AllReduceBackendConfig:
    """
    Tune allreduce configuration for a specific batch size.

    Returns:
        Best configuration for this batch size
    """
    best_config = None
    best_latency = float("inf")

    if rank == 0:
        print(f"\n{'='*80}")
        print(f"Tuning batch_size={batch_size}, hidden_size={hidden_size}")
        print(f"{'='*80}")

    for config in configs:
        if rank == 0:
            print(f"\nTesting {config.backend_name}...", flush=True)

        latency = benchmark_allreduce_config(
            config=config,
            batch_size=batch_size,
            hidden_size=hidden_size,
            world_size=world_size,
            rank=rank,
            num_iters=100,
            warmup_iters=10,
        )

        if latency is not None:
            if rank == 0:
                print(f"  Latency: {latency:.2f} us", flush=True)

            if latency < best_latency:
                best_latency = latency
                best_config = config
        else:
            if rank == 0:
                print(f"  Skipped (not supported)", flush=True)

    if best_config is None:
        # Fallback to NCCL
        best_config = AllReduceBackendConfig(
            backend_type=AllReduceBackendConfig.BACKEND_NCCL,
            backend_name="nccl",
        )
        if rank == 0:
            print(f"\nWarning: No valid config found, using NCCL fallback")

    if rank == 0:
        print(f"\nBest config for batch_size={batch_size}: {best_config.backend_name}")
        print(f"Best latency: {best_latency:.2f} us")
        print(f"{'='*80}\n")

    return best_config


def get_default_batch_sizes() -> List[int]:
    """Get default batch sizes to tune: range(1,513)"""
    return list(range(1, 513))


def main(args):
    # Initialize distributed environment
    from sglang.srt.distributed.parallel_state import init_distributed_environment
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size < 2:
        raise ValueError("AllReduce tuning requires at least 2 GPUs")

    mock_server_args = ServerArgs(
        model_path=args.model if args.model else "mock_model",
        tp_size=world_size,
        enable_symm_mem=True,  # Enable symmetric memory for tuning
        trust_remote_code=True,  # Allow loading models with custom code
    )
    set_global_server_args_for_scheduler(mock_server_args)

    # Initialize distributed environment (this will create _WORLD group)
    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        distributed_init_method="env://",
        backend="nccl",
    )

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # Initialize model parallel
    initialize_model_parallel(tensor_model_parallel_size=world_size)

    # Get batch sizes to tune
    if args.batch_sizes is not None:
        batch_sizes = args.batch_sizes
    else:
        batch_sizes = get_default_batch_sizes()

    # Get hidden size
    if args.hidden_size is not None:
        hidden_size = args.hidden_size
    elif args.model is not None:
        # Load model config to get hidden size
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
        if hasattr(config, "text_config"):
            config = config.get_text_config()
        hidden_size = config.hidden_size
        if rank == 0:
            print(f"Loaded hidden_size={hidden_size} from model {args.model}")
    else:
        raise ValueError("Must specify either --model or --hidden-size")

    # Setup all communicators
    max_batch_size = max(batch_sizes)
    setup_communicators(
        world_size=world_size,
        rank=rank,
        hidden_size=hidden_size,
        max_batch_size=max_batch_size,
        device=device,
    )

    if rank == 0:
        print(f"\nWorld size: {world_size}")
        print(f"Hidden size: {hidden_size}")
        print(f"Batch sizes to tune: {batch_sizes}")
        print(f"\nStarting tuning at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Get all configurations to test
    all_configs = get_all_backend_configs()

    if rank == 0:
        print(f"\nConfigurations to test:")
        for config in all_configs:
            print(f"  - {config.backend_name}: {config}")

    # Tune each batch size
    best_configs = {}
    start_time = time.time()

    for batch_size in batch_sizes:
        best_config = tune_batch_size(
            batch_size=batch_size,
            hidden_size=hidden_size,
            world_size=world_size,
            rank=rank,
            configs=all_configs,
        )
        best_configs[batch_size] = best_config

    end_time = time.time()

    # Save results
    if rank == 0:
        print(f"\n{'='*80}")
        print("Tuning completed!")
        print(f"Total time: {end_time - start_time:.2f} seconds")
        print(f"{'='*80}\n")

        print("Best configurations:")
        for batch_size, config in best_configs.items():
            print(f"  batch_size={batch_size:5d}: {config.backend_name}")

        # Save to file
        save_allreduce_configs(
            configs=best_configs,
            hidden_size=hidden_size,
            tp_size=world_size,
            output_dir=args.output_dir,
        )

    # Cleanup
    cleanup_communicators()
    cleanup_dist_env_and_memory()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tune allreduce configurations for different batch sizes"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name to load hidden_size from",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=None,
        help="Hidden dimension (required if --model is not specified)",
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=2,
        help="Tensor parallel size (should match number of GPUs)",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=None,
        help="Batch sizes to tune (default: range(1,513))",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for config files (default: ./configs/)",
    )

    args = parser.parse_args()

    main(args)
