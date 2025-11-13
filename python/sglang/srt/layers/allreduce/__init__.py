"""
AllReduce adaptive configuration system for SGLang.

This module provides adaptive allreduce backend selection based on batch size and hidden size.
"""

from .adaptive_allreduce import AdaptiveAllReduceLayer
from .config import (
    AllReduceBackendConfig,
    get_allreduce_configs,
    get_default_allreduce_config,
    save_allreduce_configs,
    select_allreduce_config,
)
from .manager import cleanup_adaptive_allreduce, get_adaptive_allreduce_layer

__all__ = [
    "AllReduceBackendConfig",
    "get_allreduce_configs",
    "get_default_allreduce_config",
    "save_allreduce_configs",
    "select_allreduce_config",
    "AdaptiveAllReduceLayer",
    "get_adaptive_allreduce_layer",
    "cleanup_adaptive_allreduce",
]
