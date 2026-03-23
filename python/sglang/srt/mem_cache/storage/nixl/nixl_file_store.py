import concurrent.futures
import hashlib
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import torch

from .nixl_runtime import NixlAgentContext
from .page_commit_index import NixlPageCommitIndex

logger = logging.getLogger(__name__)


@dataclass
class NixlFileWriteGroup:
    buffers: List[torch.Tensor | tuple]
    file_paths: List[str]
    page_versions: Dict[str, str]
    owner_by_page: Dict[str, int]
    storage_keys_by_page: Dict[str, List[str]]


class NixlFileStoreCoordinator:
    """Owns FILE-mode committed-version, ownership, and GC helpers."""

    def __init__(
        self,
        file_manager,
        tp_rank: int,
        tp_size: int,
        file_gc_grace_seconds: float,
    ) -> None:
        self.file_manager = file_manager
        self.commit_index = NixlPageCommitIndex(file_manager)
        self.tp_rank = tp_rank
        self.tp_size = max(1, tp_size)
        self.file_gc_grace_seconds = file_gc_grace_seconds
        self.is_zero_copy = False
        self.is_mla_model = False
        self._pending_gc: List[tuple[float, str]] = []
        self._file_stats = {
            "skipped_non_owner_writes": 0,
            "committed_version_publishes": 0,
            "pending_gc_queue_max": 0,
            "stale_version_deletions": 0,
        }

    def set_layout_mode(self, is_zero_copy: bool, is_mla_model: bool) -> None:
        self.is_zero_copy = is_zero_copy
        self.is_mla_model = is_mla_model

    def get_page_owner_rank(self, page_key: str) -> int:
        digest = hashlib.blake2b(page_key.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % self.tp_size

    def resolve_committed_file_paths(
        self, page_keys: List[str], storage_keys: List[str]
    ) -> Optional[List[str]]:
        file_paths = []
        for page_key, storage_key in zip(page_keys, storage_keys):
            version = self.commit_index.load_committed_version(page_key)
            if version is None:
                return None
            file_paths.append(
                self.file_manager.get_versioned_file_path(storage_key, version)
            )
        return file_paths

    def group_local_owned_write_items(
        self,
        page_keys: List[str],
        storage_keys: List[str],
        buffers: List[torch.Tensor | tuple],
    ) -> Optional[Dict[int, NixlFileWriteGroup]]:
        owned_items = self._collect_local_owned_write_items(
            page_keys, storage_keys, buffers
        )
        if not owned_items:
            return {}

        owned_page_keys = [item[0] for item in owned_items]
        owned_storage_keys = [item[1] for item in owned_items]
        owned_buffers = [item[2] for item in owned_items]
        prepared = self._prepare_versioned_write_file_paths(
            owned_page_keys, owned_storage_keys
        )
        if prepared is None:
            return None

        grouped: Dict[int, NixlFileWriteGroup] = {}
        for page_key, storage_key, buffer, file_path in zip(
            owned_page_keys,
            owned_storage_keys,
            owned_buffers,
            prepared["file_paths"],
        ):
            idx = 0
            group = grouped.setdefault(
                idx,
                NixlFileWriteGroup(
                    buffers=[],
                    file_paths=[],
                    page_versions={},
                    owner_by_page={},
                    storage_keys_by_page={},
                ),
            )
            group.buffers.append(buffer)
            group.file_paths.append(file_path)
            group.page_versions[page_key] = prepared["page_versions"][page_key]
            group.owner_by_page[page_key] = self.tp_rank
            group.storage_keys_by_page.setdefault(page_key, []).append(storage_key)

        return grouped

    def submit_write_groups(
        self,
        grouped: Dict[int, NixlFileWriteGroup],
        executor: concurrent.futures.ThreadPoolExecutor,
        contexts: List[NixlAgentContext],
        execute_transfer: Callable[
            [List[torch.Tensor | tuple], List[str], str, NixlAgentContext, bool], bool
        ],
    ) -> bool:
        if not grouped:
            return True

        futures = [
            executor.submit(
                self._execute_write_group,
                group,
                contexts[idx],
                execute_transfer,
            )
            for idx, group in grouped.items()
        ]
        return all(future.result() for future in futures)

    def enqueue_stale_versions_from_scan(self) -> None:
        if self.file_gc_grace_seconds < 0 or not self.file_manager.base_dir:
            return

        committed_paths = set()
        for meta_path, entry in self.commit_index.iter_entries():
            version = entry.get("version")
            page_key = os.path.basename(meta_path)[: -len(".meta.json")]
            for storage_key in self._get_storage_keys_for_scan_entry(page_key, entry):
                committed_paths.add(
                    self.file_manager.get_versioned_file_path(storage_key, version)
                )

        now = time.monotonic()
        for root, _, files in os.walk(self.file_manager.base_dir):
            for file_name in files:
                if ".v." not in file_name or file_name.endswith(".meta.json"):
                    continue
                file_path = os.path.join(root, file_name)
                if file_path in committed_paths:
                    continue
                try:
                    age = time.time() - os.path.getmtime(file_path)
                except OSError:
                    continue
                if age >= self.file_gc_grace_seconds:
                    self._pending_gc.append(
                        (now - self.file_gc_grace_seconds, file_path)
                    )

        self._update_gc_queue_max()
        self._drain_pending_gc()

    def get_stats(self) -> Dict[str, int]:
        return {
            **self._file_stats,
            "pending_gc_queue_len": len(self._pending_gc),
        }

    def publish_page_versions(
        self,
        page_versions: Dict[str, str],
        owner_by_page: Optional[Dict[str, int]] = None,
        storage_keys_by_page: Optional[Dict[str, List[str]]] = None,
    ) -> bool:
        return self._publish_page_versions(
            page_versions,
            owner_by_page=owner_by_page,
            storage_keys_by_page=storage_keys_by_page,
        )

    def _execute_write_group(
        self,
        group: NixlFileWriteGroup,
        ctx: NixlAgentContext,
        execute_transfer: Callable[
            [List[torch.Tensor | tuple], List[str], str, NixlAgentContext, bool], bool
        ],
    ) -> bool:
        with ctx.lock:
            if not execute_transfer(
                group.buffers, group.file_paths, "WRITE", ctx, True
            ):
                return False
            return self._publish_page_versions(
                group.page_versions,
                group.owner_by_page,
                group.storage_keys_by_page,
            )

    def _prepare_versioned_write_file_paths(
        self, page_keys: List[str], storage_keys: List[str]
    ) -> Optional[Dict[str, Any]]:
        page_versions: Dict[str, str] = {}
        file_paths: List[str] = []

        for page_key, storage_key in zip(page_keys, storage_keys):
            version = page_versions.setdefault(
                page_key, self.commit_index.make_version_id()
            )
            file_path = self.file_manager.get_versioned_file_path(storage_key, version)
            if not self.file_manager.create_file(file_path):
                return None
            file_paths.append(file_path)

        return {"page_versions": page_versions, "file_paths": file_paths}

    def _collect_local_owned_write_items(
        self,
        page_keys: List[str],
        storage_keys: List[str],
        buffers: List[torch.Tensor | tuple],
    ) -> List[tuple[str, str, torch.Tensor | tuple]]:
        owned_items: List[tuple[str, str, torch.Tensor | tuple]] = []
        skipped_count = 0
        for page_key, storage_key, buffer in zip(page_keys, storage_keys, buffers):
            owner_rank = self.get_page_owner_rank(page_key)
            if owner_rank != self.tp_rank:
                skipped_count += 1
                continue
            owned_items.append((page_key, storage_key, buffer))
        self._record_skipped_non_owner_writes(skipped_count)
        return owned_items

    def _get_storage_keys_for_scan_entry(
        self, page_key: str, entry: Dict[str, Any]
    ) -> List[str]:
        storage_keys = entry.get("storage_keys")
        if storage_keys:
            return list(storage_keys)
        if self.is_zero_copy:
            if self.is_mla_model:
                return [f"{page_key}_k"]
            return [f"{page_key}_k", f"{page_key}_v"]
        return [page_key]

    def _publish_page_versions(
        self,
        page_versions: Dict[str, str],
        owner_by_page: Optional[Dict[str, int]] = None,
        storage_keys_by_page: Optional[Dict[str, List[str]]] = None,
    ) -> bool:
        for page_key, version in page_versions.items():
            previous_version = self.commit_index.load_committed_version(page_key)
            owner = owner_by_page.get(page_key) if owner_by_page else None
            if not self.commit_index.publish_committed_version(
                page_key,
                version,
                owner,
                storage_keys=(
                    storage_keys_by_page.get(page_key) if storage_keys_by_page else None
                ),
            ):
                return False
            self._record_file_stat("committed_version_publishes")
            logger.debug(
                "NIXL FILE committed version publish "
                f"page_key={page_key} version={version} owner_rank={owner}"
            )
            if (
                previous_version
                and previous_version != version
                and storage_keys_by_page is not None
            ):
                for storage_key in storage_keys_by_page.get(page_key, []):
                    self._pending_gc.append(
                        (
                            time.monotonic(),
                            self.file_manager.get_versioned_file_path(
                                storage_key, previous_version
                            ),
                        )
                    )
        self._update_gc_queue_max()
        logger.debug(f"NIXL FILE pending GC queue length={len(self._pending_gc)}")
        self._drain_pending_gc()
        return True

    def _drain_pending_gc(self) -> None:
        if not self._pending_gc or self.file_gc_grace_seconds < 0:
            return

        now = time.monotonic()
        keep: List[tuple[float, str]] = []
        for created_at, file_path in self._pending_gc:
            if now - created_at < self.file_gc_grace_seconds:
                keep.append((created_at, file_path))
                continue
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    self._record_file_stat("stale_version_deletions")
                    logger.debug(f"NIXL FILE deleted stale version {file_path}")
            except Exception as e:
                logger.warning(f"Failed to remove stale NIXL file {file_path}: {e}")
                keep.append((created_at, file_path))
        self._pending_gc[:] = keep

    def _record_skipped_non_owner_writes(self, skipped_count: int) -> None:
        if skipped_count <= 0:
            return
        self._record_file_stat("skipped_non_owner_writes", skipped_count)
        logger.debug(
            "NIXL FILE skipped non-owner writes "
            f"count={skipped_count} local_rank={self.tp_rank}"
        )

    def _record_file_stat(self, key: str, delta: int = 1) -> None:
        self._file_stats[key] += delta

    def _update_gc_queue_max(self) -> None:
        self._file_stats["pending_gc_queue_max"] = max(
            self._file_stats["pending_gc_queue_max"], len(self._pending_gc)
        )
