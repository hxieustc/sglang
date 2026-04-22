import concurrent.futures
import logging
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Union

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorage,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
)
from sglang.srt.mem_cache.memory_pool_host import HostKVCache

from .nixl_file_store import NixlFileStoreCoordinator
from .nixl_runtime import NixlAgentContext, NixlTransferChunk
from .nixl_utils import (
    NixlBackendConfig,
    NixlBackendSelection,
    NixlFileManager,
    NixlRegistration,
)

try:
    from nixl._api import nixl_agent, nixl_agent_config
except ImportError as e:
    raise ImportError(
        "Please install NIXL by following the instructions at "
        "https://github.com/ai-dynamo/nixl/blob/main/README.md "
        "to use HiCacheNixl storage backend."
    ) from e

logger = logging.getLogger(__name__)


class HiCacheNixl(HiCacheStorage):
    """HiCacheNixl provides high-performance storage using NIXL plugins."""

    def __init__(
        self,
        storage_config: HiCacheStorageConfig,
        file_path: str = "/tmp/hicache_storage",
    ):
        """Initialize NIXL storage connector."""

        # create nixlconfig from the --hicache-storage-backend-extra-config
        nixlconfig = NixlBackendConfig(storage_config.extra_config)

        # select the NIXL backend plugin from extra_config or environment variable
        plugin = nixlconfig.get_specified_plugin()
        file_layout = str(
            nixlconfig.get_runtime_param("file_layout", "hashed2")
        ).lower()

        # Might be better to be unified across HiCache backends and moved to HiCacheController
        file_path = envs.SGLANG_HICACHE_NIXL_BACKEND_STORAGE_DIR.get() or file_path
        self.file_manager = (
            NixlFileManager(file_path, layout=file_layout)
            if plugin not in NixlBackendSelection.OBJ_PLUGINS
            else None
        )
        self.file_store = (
            NixlFileStoreCoordinator(
                self.file_manager,
                tp_rank=storage_config.tp_rank,
                tp_size=storage_config.tp_size,
                file_gc_grace_seconds=float(
                    nixlconfig.get_runtime_param("file_gc_grace_seconds", 3000.0)
                ),
            )
            if self.file_manager
            else None
        )
        self.commit_index = self.file_store.commit_index if self.file_store else None
        self._pending_gc = self.file_store._pending_gc if self.file_store else []
        self._file_stats = self.file_store._file_stats if self.file_store else {}

        # Initialize suffix based on storage config
        tp_rank, tp_size, model_name = (
            storage_config.tp_rank,
            storage_config.tp_size,
            storage_config.model_name,
        )
        self.tp_rank = tp_rank
        self.tp_size = max(1, tp_size)

        self.is_mla_model = storage_config.is_mla_model
        self.is_zero_copy = False

        model_name = "-".join(model_name.split("/")) if model_name else ""

        if self.is_mla_model:
            self.config_suffix = f"_{model_name}"
        else:
            self.config_suffix = f"_{model_name}_{tp_rank}_{tp_size}"

        agent_config = nixl_agent_config(backends=[])
        self._agent_config = agent_config
        self._nixlconfig = nixlconfig
        self._plugin = plugin
        self.num_jobs = max(1, int(nixlconfig.get_runtime_param("numjobs", 4)))

        self.query_ctx = self._create_agent_context("query")
        self.backend_selector = self.query_ctx.backend_selector
        if self.backend_selector.mem_type == "FILE":
            self.file_contexts = [self._create_agent_context("file")]
            self.file_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="HiCacheNixlFile"
            )
            self.read_contexts = self.file_contexts
            self.write_contexts = self.file_contexts
            self.read_ctx = self.file_contexts[0]
            self.write_ctx = self.file_contexts[0]
            self.read_executor = self.file_executor
            self.write_executor = self.file_executor
        else:
            self.read_contexts = [
                self._create_agent_context(f"read_{i}") for i in range(self.num_jobs)
            ]
            self.write_contexts = [
                self._create_agent_context(f"write_{i}") for i in range(self.num_jobs)
            ]
            self.read_ctx = self.read_contexts[0]
            self.write_ctx = self.write_contexts[0]
            self.read_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.num_jobs, thread_name_prefix="HiCacheNixlRead"
            )
            self.write_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.num_jobs, thread_name_prefix="HiCacheNixlWrite"
            )

        self.agent_name = self.read_ctx.agent_name
        self.agent = self.read_ctx.agent
        self.registration = self.read_ctx.registration
        self._startup_gc_scan_pending = self.backend_selector.mem_type == "FILE"

    def _get_suffixed_key(self, key: str) -> str:
        return key + self.config_suffix

    def _create_agent_context(self, role: str) -> NixlAgentContext:
        agent_name = f"hicache_nixl_{role}_{uuid.uuid4()}"
        agent = nixl_agent(agent_name, self._agent_config)
        backend_selector = NixlBackendSelection(self._plugin, self._nixlconfig)
        if not backend_selector.create_backend(agent):
            raise RuntimeError(f"Failed to create NIXL backend for {role} agent")

        if hasattr(self, "backend_selector"):
            if backend_selector.backend_name != self.backend_selector.backend_name:
                raise RuntimeError(
                    "NIXL backend mismatch across agents: "
                    f"{backend_selector.backend_name} != {self.backend_selector.backend_name}"
                )
            if backend_selector.mem_type != self.backend_selector.mem_type:
                raise RuntimeError(
                    "NIXL memory type mismatch across agents: "
                    f"{backend_selector.mem_type} != {self.backend_selector.mem_type}"
                )

        return NixlAgentContext(
            role=role,
            agent_name=agent_name,
            agent=agent,
            registration=NixlRegistration(agent),
            backend_selector=backend_selector,
            lock=threading.RLock(),
        )

    def _chunk_transfer_items(
        self,
        keys: List[str],
        buffers: List[torch.Tensor | tuple],
        max_chunks: int,
    ) -> List[NixlTransferChunk]:
        if len(keys) != len(buffers):
            raise ValueError("Mismatch between keys and buffers lengths")
        if not keys:
            return []

        chunk_count = min(max_chunks, len(keys))
        base, remainder = divmod(len(keys), chunk_count)
        chunks = []
        start = 0
        for i in range(chunk_count):
            size = base + (1 if i < remainder else 0)
            end = start + size
            chunks.append(
                NixlTransferChunk(keys=keys[start:end], buffers=buffers[start:end])
            )
            start = end
        return chunks

    def _get_page_key(self, key: str) -> str:
        return self._get_suffixed_key(key)

    def _expand_page_keys(self, keys: List[str]) -> List[str]:
        page_keys = []
        for key in keys:
            page_key = self._get_page_key(key)
            if self.is_zero_copy and not self.is_mla_model:
                page_keys.extend([page_key, page_key])
            else:
                page_keys.append(page_key)
        return page_keys

    def _get_agent_index(self, page_key: str, num_contexts: int) -> int:
        return self.file_store.get_page_owner_rank(page_key) % num_contexts

    def _get_page_owner_rank(self, page_key: str) -> int:
        return self.file_store.get_page_owner_rank(page_key)

    def _publish_page_versions(
        self,
        page_versions: Dict[str, str],
        owner_by_page: Optional[Dict[str, int]] = None,
        storage_keys_by_page: Optional[Dict[str, List[str]]] = None,
    ) -> bool:
        return self.file_store.publish_page_versions(
            page_versions,
            owner_by_page=owner_by_page,
            storage_keys_by_page=storage_keys_by_page,
        )

    def _execute_transfer_parallel(
        self,
        buffers: List[torch.Tensor | tuple],
        keys: List[str],
        direction: str,
        contexts: List[NixlAgentContext],
        executor: concurrent.futures.ThreadPoolExecutor,
        lane_keys: Optional[List[str]] = None,
    ) -> bool:
        if len(buffers) != len(keys):
            logger.error("Mismatch between number of tensors/buffers and files/objects")
            return False
        if not keys:
            return True
        if lane_keys is not None and len(lane_keys) != len(keys):
            logger.error("Mismatch between transfer keys and lane keys")
            return False
        if len(contexts) == 1 or len(keys) == 1:
            ctx = contexts[0]
            if lane_keys:
                ctx = contexts[self._get_agent_index(lane_keys[0], len(contexts))]
            return self._execute_transfer(buffers, keys, direction, ctx)

        if lane_keys is not None:
            grouped: Dict[int, Dict[str, List[Any]]] = {}
            for key, buffer, lane_key in zip(keys, buffers, lane_keys):
                idx = self._get_agent_index(lane_key, len(contexts))
                group = grouped.setdefault(idx, {"keys": [], "buffers": []})
                group["keys"].append(key)
                group["buffers"].append(buffer)

            futures = [
                executor.submit(
                    self._execute_transfer,
                    group["buffers"],
                    group["keys"],
                    direction,
                    contexts[idx],
                )
                for idx, group in grouped.items()
            ]
            return all(future.result() for future in futures)

        chunks = self._chunk_transfer_items(keys, buffers, len(contexts))
        futures = [
            executor.submit(
                self._execute_transfer,
                chunk.buffers,
                chunk.keys,
                direction,
                contexts[i],
            )
            for i, chunk in enumerate(chunks)
        ]
        return all(future.result() for future in futures)

    def register_buffers(
        self,
        buffers: Union[torch.Tensor, List[torch.Tensor], List[tuple]],
        ctx: Optional[NixlAgentContext] = None,
    ) -> Optional[Any]:
        """Register tensor(s) or target locations in host memory (list of addr,len tuples) with NIXL."""
        registration = (ctx or self.read_ctx).registration
        if isinstance(buffers[0], tuple):
            tuples = [(x[0], x[1], 0, "") for x in buffers]
            return registration._register_memory(tuples, "DRAM")
        else:
            return registration._register_memory(buffers)

    def register_files(
        self, file_paths: List[str], open_file: Optional[bool] = True
    ) -> Optional[Any]:
        """Register files with NIXL."""
        tuples = self.file_manager.files_to_nixl_tuples(file_paths)
        try:
            return self.read_ctx.registration._register_memory(tuples, "FILE")
        finally:
            self.file_manager.close_nixl_tuples(tuples)

    def register_objects(
        self, keys: List[str], sizes: Optional[List[int]] = None
    ) -> Optional[Any]:
        """Register objects with NIXL."""
        if not keys:
            return None
        tuples = [(0, 0, key, "") for key in keys]
        return self.read_ctx.registration._register_memory(tuples, "OBJ")

    def _execute_transfer(
        self,
        buffers: Optional[List[torch.Tensor | tuple]],
        keys: List[str],
        direction: str,
        ctx: NixlAgentContext,
        lock_held: bool = False,
    ) -> bool:
        if len(buffers) != len(keys):
            logger.error("Mismatch between number of tensors/buffers and files/objects")
            return False

        if lock_held:
            return self._execute_transfer_impl(buffers, keys, direction, ctx)
        with ctx.lock:
            return self._execute_transfer_impl(buffers, keys, direction, ctx)

    def _execute_transfer_impl(
        self,
        buffers: Optional[List[torch.Tensor | tuple]],
        keys: List[str],
        direction: str,
        ctx: NixlAgentContext,
    ) -> bool:
        # Registering file and object keys per transfer, to be updated when
        # pre-registration for file and object is added to HiCache.
        tuples = []

        try:
            if ctx.backend_selector.mem_type == "FILE":
                tuples = self.file_manager.files_to_nixl_tuples(keys)
                if not tuples or not ctx.registration._register_memory(tuples, "FILE"):
                    logger.error("Failed to prepare files for transfer")
                    return False
            else:  # mem_type == "OBJ"
                tuples = [(0, 0, key, "") for key in keys]
                if not tuples or not ctx.registration._register_memory(tuples, "OBJ"):
                    logger.error("Failed to register objects")
                    return False

            # Prepare transfer descriptors
            if isinstance(buffers[0], torch.Tensor):
                tensor_sizes = [
                    tensor.element_size() * tensor.numel() for tensor in buffers
                ]
                storage_tuples = [(x[0], s, x[2]) for x, s in zip(tuples, tensor_sizes)]
                host_descs = ctx.agent.get_xfer_descs(buffers)

                if direction in ("READ", "WRITE"):
                    self.register_buffers(buffers, ctx=ctx)

            elif isinstance(buffers[0], tuple):
                storage_tuples = [(x[0], y[1], x[2]) for x, y in zip(tuples, buffers)]
                host_descs = ctx.agent.get_xfer_descs(
                    [(x[0], x[1], 0) for x in buffers], "DRAM"
                )

                if direction in ("READ", "WRITE"):
                    self.register_buffers(buffers, ctx=ctx)

            else:
                return False

            storage_descs = ctx.agent.get_xfer_descs(
                storage_tuples, ctx.backend_selector.mem_type
            )

            if (host_descs is None) or (storage_descs is None):
                logger.error("Failed to get transfer descriptors")
                return False

            try:
                xfer_req = ctx.agent.initialize_xfer(
                    direction, host_descs, storage_descs, ctx.agent_name
                )
            except Exception:
                if not self.register_buffers(buffers, ctx=ctx):
                    logger.error("Failed to register tensors/buffers")
                    return False

                try:
                    xfer_req = ctx.agent.initialize_xfer(
                        direction, host_descs, storage_descs, ctx.agent_name
                    )
                except Exception as e:
                    logger.error(f"Failed to create transfer request: {e}")
                    return False

            try:
                state = ctx.agent.transfer(xfer_req)
                while state != "DONE":
                    state = ctx.agent.check_xfer_state(xfer_req)
                    if state == "ERR":
                        ctx.agent.release_xfer_handle(xfer_req)
                        logger.error("Transfer failed")
                        return False
                    time.sleep(0.0001)

                ctx.agent.release_xfer_handle(xfer_req)
                return True

            except Exception as e:
                logger.error(f"Failed to execute transfer: {e}")
                import traceback

                logger.error(f"Traceback: {traceback.format_exc()}")
                return False
        finally:
            if ctx.backend_selector.mem_type == "FILE":
                self.file_manager.close_nixl_tuples(tuples)

    def get(
        self,
        key: str,
        target_location: Optional[torch.Tensor | int] = None,
        target_sizes: Optional[int] = None,
    ) -> torch.Tensor | None:
        # To be removed, being compatible with the current API
        if target_location is None:
            return None
        if target_sizes:
            result = self.batch_get([key], [target_location], [target_sizes])
        else:
            result = self.batch_get([key], [target_location])
        return result[0] if result else None

    def batch_get(
        self,
        keys: List[str],
        target_locations: Optional[List[torch.Tensor | int]] = None,
        target_sizes: Optional[List[int]] = None,
    ) -> List[torch.Tensor | None]:
        if not keys:
            return []

        # To be removed, being compatible with the current API
        if not target_locations:
            return [None] * len(keys)

        if target_sizes and (len(target_sizes) != len(target_locations)):
            logger.error("Mismatch between number of target_locations and target_sizes")
            return [None] * len(keys)

        if target_sizes:
            dest = list(zip(target_locations, target_sizes))
        else:
            dest = target_locations

        # Add suffix to keys
        suffixed_keys = [self._get_suffixed_key(key) for key in keys]

        if self.backend_selector.mem_type == "FILE":
            page_keys = [self._get_page_key(key) for key in keys]
            file_paths = self.file_store.resolve_committed_file_paths(
                page_keys, suffixed_keys
            )
            if file_paths is None:
                return [None] * len(keys)
            success = self._execute_transfer_parallel(
                dest,
                file_paths,
                "READ",
                self.read_contexts,
                self.read_executor,
                lane_keys=page_keys,
            )
        else:
            success = self._execute_transfer_parallel(
                dest,
                suffixed_keys,
                "READ",
                self.read_contexts,
                self.read_executor,
            )
        return target_locations if success and not target_sizes else [None] * len(keys)

    def set(
        self,
        key: str,
        value: Optional[torch.Tensor] = None,
        target_location: Optional[int] = None,
        target_sizes: Optional[int] = None,
    ) -> bool:
        if target_location and target_sizes:
            return self.batch_set([key], None, [target_location], [target_sizes])
        else:
            return self.batch_set([key], [value])

    def batch_set(
        self,
        keys: List[str],
        values: Optional[List[torch.Tensor]] = None,
        target_locations: Optional[List[int]] = None,
        target_sizes: Optional[List[int]] = None,
    ) -> bool:
        if not keys or (not values and (not target_locations or not target_sizes)):
            logger.error("Keys or values were not passed")
            return False

        if not values:
            values = list(zip(target_locations, target_sizes))

        # Add suffix to keys
        suffixed_keys = [self._get_suffixed_key(key) for key in keys]

        if self.backend_selector.mem_type == "FILE":
            page_keys = [self._get_page_key(key) for key in keys]
            grouped = self.file_store.group_local_owned_write_items(
                page_keys, suffixed_keys, values
            )
            if grouped is None:
                logger.error("Failed to prepare files for write")
                return False
            return self.file_store.submit_write_groups(
                grouped,
                self.write_executor,
                self.write_contexts,
                self._execute_transfer,
            )
        else:  # mem_type == "OBJ"
            return self._execute_transfer_parallel(
                values,
                suffixed_keys,
                "WRITE",
                self.write_contexts,
                self.write_executor,
            )

    ############################################################################
    # batch_*_v1 functions
    # zero copy + non-zero-copy version for get, set, exists, batch_exists
    ############################################################################

    def clear(self) -> None:
        self.file_manager.clear()

    def register_mem_pool_host(self, mem_pool_host: HostKVCache):
        super().register_mem_pool_host(mem_pool_host)

        # enable zero-copy automatically if mem layout is page_first or page_first_direct
        self.is_zero_copy = self.mem_pool_host.layout in [
            "page_first",
            "page_first_direct",
        ]
        if self.file_store is not None:
            self.file_store.set_layout_mode(self.is_zero_copy, self.is_mla_model)
        if self._startup_gc_scan_pending and self.backend_selector.mem_type == "FILE":
            self.file_store.enqueue_stale_versions_from_scan()
            self._startup_gc_scan_pending = False

        logger.info(
            f"HiCacheNixl: Registered mem_pool_host with layout {self.mem_pool_host.layout}, zero_copy set to {self.is_zero_copy}"
        )

    def exists(self, key: str) -> bool:
        results = self.batch_exists([key])
        return results > 0

    def batch_exists(
        self,
        keys: List[str],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> int:
        # Add suffix to key

        if self.backend_selector.mem_type == "FILE":
            committed_count = 0
            for key in keys:
                if (
                    self.file_store.commit_index.load_committed_version(
                        self._get_page_key(key)
                    )
                    is None
                ):
                    return committed_count
                committed_count += 1
            return committed_count

        if self.is_zero_copy:
            key_list = self._get_key_list_from_meta(keys)
            key_denominator = 1 if self.is_mla_model else 2
        else:
            key_list = [self._get_suffixed_key(key) for key in keys]
            key_denominator = 1

        # obtain list of tuples by calling self.registration.create_query_tuples()
        tuples = []
        for key in key_list:
            tuples += self.query_ctx.registration.create_query_tuples(
                key,
                self.query_ctx.backend_selector.mem_type,
                self.file_manager if self.backend_selector.mem_type == "FILE" else None,
            )

        query_res = self.query_ctx.agent.query_memory(
            tuples,
            self.query_ctx.backend_selector.backend_name,
            mem_type=self.query_ctx.backend_selector.mem_type,
        )

        for i in range(len(query_res)):
            if query_res[i] is None:
                return i // key_denominator
        return len(query_res) // key_denominator

    def _get_key_list_from_meta(self, keys: List[str]) -> List[str]:
        # construct the key list for NIXL transfer based on the keys and the suffix, for each key, we will have one suffixed key for k buffer and one suffixed key for v buffer if it's not an MLA model, and only one suffixed key for k buffer if it's an MLA model, since MLA model only has k/v interleaved buffer
        key_list = []

        for key_ in keys:
            suffixed_key = self._get_suffixed_key(key_)
            if self.is_mla_model:
                key_list.append(f"{suffixed_key}_k")
            else:
                key_list.append(f"{suffixed_key}_k")
                key_list.append(f"{suffixed_key}_v")

        return key_list

    def _get_location_and_size_list_from_meta(
        self, keys: List[str], host_indices: torch.Tensor
    ):
        # zero copy: mem_pool_host.get_data_page() does not work due to non-contiguous tensors, causing issues for NIXL transfer
        ptr_list, element_size_list = self.mem_pool_host.get_page_buffer_meta(
            host_indices
        )
        key_list = self._get_key_list_from_meta(keys)

        if len(key_list) != len(ptr_list):
            logger.error(
                f"HiCacheNixl: mismatch between number of keys and number of buffer meta entries, keys: {len(keys)}, key_list: {len(key_list)}, buffer meta entries: {len(ptr_list)}"
            )
            return [], [], [], []

        return key_list, [], ptr_list, element_size_list

    def _batch_get_preprocess(self, keys: List[str], host_indices: torch.Tensor):
        page_num = len(host_indices) // self.mem_pool_host.page_size

        if len(keys) == 0 or len(keys) != page_num:
            logger.warning(
                f"HiCacheNixl: empty keys or mismatch in keys and host_indices lengths. keys: {len(keys)}, host_indices: {len(host_indices)}, page_size: {self.mem_pool_host.page_size}"
            )
            return [], [], [], []

        if self.is_zero_copy:
            key_list, _, ptr_list, element_size_list = (
                self._get_location_and_size_list_from_meta(keys, host_indices)
            )
            return key_list, [], ptr_list, element_size_list
        else:
            # non zero copy: create contiguous, temporary tensors
            target_tensors = [
                self.mem_pool_host.get_dummy_flat_data_page() for i in range(page_num)
            ]

            key_list = [self._get_suffixed_key(key) for key in keys]
            ptr_list = [tensor.data_ptr() for tensor in target_tensors]
            element_size_list = [
                tensor.numel() * tensor.element_size() for tensor in target_tensors
            ]

            return key_list, target_tensors, ptr_list, element_size_list

    def _batch_get_zero_copy_impl(
        self,
        keys: List[str],
        key_strs: List[str],
        target_tensors: List[torch.Tensor],
        target_locations: List[int],
        target_sizes: List[int],
    ) -> List[int]:

        if not key_strs or not target_locations or not target_sizes:
            return [False] * len(keys)

        if (len(key_strs) != len(target_locations)) or (
            len(target_sizes) != len(target_locations)
        ):
            logger.error(
                "Mismatch between number of key_strs, target_locations and target_sizes"
            )
            return [False] * len(keys)

        if self.is_zero_copy:
            dest = list(zip(target_locations, target_sizes))
        else:
            dest = target_tensors

        if self.backend_selector.mem_type == "FILE":
            page_keys = self._expand_page_keys(keys)
            file_paths = self.file_store.resolve_committed_file_paths(
                page_keys, key_strs
            )
            if file_paths is None:
                logger.warning(
                    "HiCacheNixl batch_get_v1 skipped pages without a committed version"
                )
                return [False] * len(keys)
            success = self._execute_transfer_parallel(
                dest,
                file_paths,
                "READ",
                self.read_contexts,
                self.read_executor,
                lane_keys=page_keys,
            )
        else:
            success = self._execute_transfer_parallel(
                dest, key_strs, "READ", self.read_contexts, self.read_executor
            )

        return [True] * len(key_strs) if success else [False] * len(key_strs)

    def _batch_get_postprocess(
        self,
        host_indices: torch.Tensor,
        target_tensors: List[torch.Tensor],
        results: List[bool],
    ) -> List[bool]:

        page_num = len(host_indices) // self.mem_pool_host.page_size

        if self.is_zero_copy:
            # zero copy: update final results based on the boolean results from NIXL transfer
            if self.is_mla_model:
                return results
            else:
                results = [
                    (results[2 * i] and results[2 * i + 1]) for i in range(page_num)
                ]
                return results
        else:
            # non zero copy: copy data from temporary tensors to mem_pool_host page by page
            for i in range(page_num):
                if not results[i]:
                    break
                self.mem_pool_host.set_from_flat_data_page(
                    host_indices[i * self.mem_pool_host.page_size], target_tensors[i]
                )

            return results

    def batch_get_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:

        key_strs, target_tensors, buffer_ptrs, buffer_sizes = (
            self._batch_get_preprocess(keys, host_indices)
        )

        if not key_strs or not buffer_ptrs or not buffer_sizes:
            logger.error(
                "HiCacheNixl batch_get_v1: preprocessing failed, empty key_strs, buffer_ptrs or buffer_sizes"
            )
            return [False] * len(keys)

        start_time = time.perf_counter()

        results_get = self._batch_get_zero_copy_impl(
            keys, key_strs, target_tensors, buffer_ptrs, buffer_sizes
        )

        end_time = time.perf_counter()
        elapsed_time_ms = (end_time - start_time) * 1000
        total_bytes = sum(s for s in buffer_sizes if s is not None)

        logger.debug(
            f"HiCacheNixl batch_get_v1 transferred: {len(keys)} keys (pages), {host_indices.numel()} host_indices, {total_bytes} bytes, total time: {elapsed_time_ms:.3f} ms, effective bandwidth: {total_bytes / (elapsed_time_ms / 1000) / (1024 * 1024):.2f} MB/s"
        )

        return self._batch_get_postprocess(host_indices, target_tensors, results_get)

    def _batch_set_preprocess(self, keys: List[str], host_indices: torch.Tensor):

        page_num = len(host_indices) // self.mem_pool_host.page_size

        if len(keys) == 0 or len(keys) != page_num:
            logger.warning(
                f"HiCacheNixl: empty keys or mismatch in keys and host_indices lengths. keys: {len(keys)}, host_indices: {len(host_indices)}, page_size: {self.mem_pool_host.page_size}"
            )
            return [], [], [], []

        if self.is_zero_copy:
            key_list, _, ptr_list, element_size_list = (
                self._get_location_and_size_list_from_meta(keys, host_indices)
            )
            return key_list, [], ptr_list, element_size_list
        else:
            # non zero copy: NIXL still requires contiguous tensors for transfer
            target_tensors = [
                self.mem_pool_host.get_data_page(
                    host_indices[i * self.mem_pool_host.page_size], flat=False
                ).contiguous()
                for i in range(page_num)
            ]

            key_list = [self._get_suffixed_key(key) for key in keys]
            ptr_list = [tensor.data_ptr() for tensor in target_tensors]
            element_size_list = [
                tensor.numel() * tensor.element_size() for tensor in target_tensors
            ]

            return key_list, target_tensors, ptr_list, element_size_list

    def _batch_set_zero_copy_impl(
        self,
        keys: List[str],
        key_strs: List[str],
        target_tensors: List[torch.Tensor],
        target_locations: List[int],
        target_sizes: List[int],
    ) -> List[bool]:

        if not key_strs or not target_locations or not target_sizes:
            return [False] * len(keys)

        if (len(key_strs) != len(target_locations)) or (
            len(target_sizes) != len(target_locations)
        ):
            logger.error(
                "Mismatch between number of key_strs, target_locations and target_sizes"
            )
            return [False] * len(keys)

        if self.is_zero_copy:
            src = list(zip(target_locations, target_sizes))
        else:
            src = target_tensors

        if self.backend_selector.mem_type == "FILE":
            page_keys = self._expand_page_keys(keys)
            grouped = self.file_store.group_local_owned_write_items(
                page_keys, key_strs, src
            )
            if grouped is None:
                logger.error("Failed to prepare files for transfer")
                return [False] * len(keys)
            success = self.file_store.submit_write_groups(
                grouped,
                self.write_executor,
                self.write_contexts,
                self._execute_transfer,
            )
        else:  # mem_type == "OBJ"
            success = self._execute_transfer_parallel(
                src, key_strs, "WRITE", self.write_contexts, self.write_executor
            )

        return [True] * len(keys) if success else [False] * len(keys)

    def batch_set_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:

        if len(keys) == 0:
            return []

        key_strs, target_tensors, buffer_ptrs, buffer_sizes = (
            self._batch_set_preprocess(keys, host_indices)
        )

        if not key_strs or not buffer_ptrs or not buffer_sizes:
            logger.error(
                "HiCacheNixl batch_set_v1: preprocessing failed, empty key_strs, buffer_ptrs or buffer_sizes"
            )
            return [False] * len(keys)

        start_time = time.perf_counter()

        results_set = self._batch_set_zero_copy_impl(
            keys, key_strs, target_tensors, buffer_ptrs, buffer_sizes
        )

        end_time = time.perf_counter()
        elapsed_time_ms = (end_time - start_time) * 1000
        total_bytes = sum(s for s in buffer_sizes if s is not None)
        logger.debug(
            f"HiCacheNixl batch_set_v1 transferred: {len(keys)} keys (pages), {host_indices.numel()} host_indices, {total_bytes} bytes, total time: {elapsed_time_ms:.3f} ms, effective bandwidth: {total_bytes / (elapsed_time_ms / 1000) / (1024 * 1024):.2f} MB/s"
        )

        return results_set

    def close(self) -> None:
        self.read_executor.shutdown(wait=True)
        if self.write_executor is not self.read_executor:
            self.write_executor.shutdown(wait=True)

    def get_stats(self):
        if self.backend_selector.mem_type != "FILE":
            return None
        return self.file_store.get_stats()
