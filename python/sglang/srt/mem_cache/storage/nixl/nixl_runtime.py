from dataclasses import dataclass
from typing import Any, List

import torch

from .nixl_utils import NixlBackendSelection, NixlRegistration


@dataclass
class NixlAgentContext:
    role: str
    agent_name: str
    agent: Any
    registration: NixlRegistration
    backend_selector: NixlBackendSelection
    lock: Any


@dataclass
class NixlTransferChunk:
    keys: List[str]
    buffers: List[torch.Tensor | tuple]
