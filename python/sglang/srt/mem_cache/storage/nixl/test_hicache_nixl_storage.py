#!/usr/bin/env python3

import concurrent.futures
import multiprocessing
import os
import threading
import unittest
from typing import List
from unittest.mock import MagicMock

os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", "/tmp")

import torch

from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig
from sglang.srt.mem_cache.storage.nixl.hicache_nixl import HiCacheNixl
from sglang.srt.mem_cache.storage.nixl.nixl_utils import (
    NixlBackendConfig,
    NixlFileManager,
    NixlRegistration,
)
from sglang.srt.mem_cache.storage.nixl.page_commit_index import NixlPageCommitIndex


class _FakeHostKVCache:
    def __init__(self, page_size: int = 1, dtype: torch.dtype = torch.float32):
        self.page_size = page_size
        self.layout = "layer_first"
        self.dtype = dtype
        self._shape = (2, 1, self.page_size, 1, 1)
        self._lock = threading.RLock()
        self._pages = {}

    def get_data_page(self, index: int, flat: bool = True) -> torch.Tensor:
        with self._lock:
            if index not in self._pages:
                self._pages[index] = torch.full(
                    self._shape, float(index), dtype=self.dtype
                )
            return self._pages[index].clone()

    def get_dummy_flat_data_page(self) -> torch.Tensor:
        return torch.zeros(self._shape, dtype=self.dtype)

    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
        with self._lock:
            self._pages[index] = data_page.clone()

    def get_page_buffer_meta(self, indices):
        ptr_list = []
        size_list = []
        for idx in indices[:: self.page_size]:
            tensor = self.get_data_page(int(idx), flat=False)
            ptr_list.append(tensor.data_ptr())
            size_list.append(tensor.numel() * tensor.element_size())
        return ptr_list, size_list


def _owner_publish_worker(test_dir: str, rank: int, tp_size: int, key: str, queue):
    os.environ["SGLANG_HICACHE_NIXL_BACKEND_STORAGE_DIR"] = test_dir
    storage_config = HiCacheStorageConfig(
        tp_rank=rank,
        tp_size=tp_size,
        pp_rank=0,
        pp_size=1,
        is_mla_model=False,
        enable_storage_metrics=False,
        is_page_first_layout=False,
        model_name="test_model",
        extra_config={
            "runtime": {"file_gc_grace_seconds": 0},
            "plugin": {"posix": {"active": True}},
        },
    )
    hicache = HiCacheNixl(storage_config=storage_config)
    try:
        hicache._execute_transfer = (
            lambda buffers, keys, direction, ctx, lock_held=False: True
        )
        tensor = torch.ones(2, 2, dtype=torch.float32) * (rank + 1)
        page_key = hicache._get_page_key(key)
        queue.put(
            {
                "rank": rank,
                "ok": hicache.set(key, tensor),
                "owner_rank": hicache._get_page_owner_rank(page_key),
                "stats": hicache.get_stats(),
                "exists": hicache.batch_exists([key]),
            }
        )
    finally:
        hicache.close()


class TestNixlUnified(unittest.TestCase):
    """Unified test suite for all NIXL components."""

    def setUp(self):
        """Set up test environment."""
        # Create test directories
        self.test_dir = "/tmp/test_nixl_unified"
        os.makedirs(self.test_dir, exist_ok=True)
        os.environ["SGLANG_HICACHE_NIXL_BACKEND_STORAGE_DIR"] = self.test_dir

        # Mock NIXL agent for registration tests
        self.mock_agent = MagicMock()
        self.mock_agent.get_reg_descs.return_value = "mock_reg_descs"
        self.mock_agent.register_memory.return_value = "mock_registered_memory"

        # Create instances
        self.file_manager = NixlFileManager(self.test_dir)
        self.commit_index = NixlPageCommitIndex(self.file_manager)
        self.registration = NixlRegistration(self.mock_agent)

        # Create storage config for testing
        self.storage_config = HiCacheStorageConfig(
            tp_rank=0,
            tp_size=1,
            pp_rank=0,
            pp_size=1,
            is_mla_model=False,
            enable_storage_metrics=False,
            is_page_first_layout=False,
            model_name="test_model",
            extra_config={"plugin": {"posix": {"active": True}}},
        )

        try:
            self.hicache = HiCacheNixl(storage_config=self.storage_config)
        except ImportError:
            self.skipTest("NIXL not available, skipping NIXL storage tests")

    def tearDown(self):
        """Clean up test directories."""
        if hasattr(self, "hicache") and hasattr(self.hicache, "close"):
            self.hicache.close()
        if os.path.exists(self.test_dir):
            import shutil

            shutil.rmtree(self.test_dir, ignore_errors=True)

    def delete_test_file(self, file_path: str) -> bool:
        """Helper method to delete a test file.

        Args:
            file_path: Path to the file to delete

        Returns:
            bool: True if file was deleted or didn't exist, False on error
        """
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
            return True
        except Exception as e:
            return False

    def verify_tensors_equal(self, expected: torch.Tensor, actual: torch.Tensor):
        """Helper to verify tensor equality."""
        self.assertIsNotNone(actual, "Retrieved tensor is None")
        self.assertTrue(
            torch.allclose(expected, actual, atol=1e-6),
            f"Tensors not equal:\nExpected: {expected}\nActual: {actual}",
        )

    def verify_tensor_lists_equal(
        self, expected: List[torch.Tensor], actual: List[torch.Tensor]
    ):
        """Helper to verify lists of tensors are equal."""
        self.assertEqual(len(expected), len(actual), "Lists have different lengths")
        for exp, act in zip(expected, actual):
            self.verify_tensors_equal(exp, act)

    # ============================================================================
    # HiCache Integration Tests
    # ============================================================================

    def test_single_set_get(self):
        """Test single tensor set/get operations."""
        key = "test_key"
        value = torch.randn(10, 10, device="cpu")
        dst_tensor = torch.zeros_like(value, device="cpu")

        # Test set
        self.assertTrue(self.hicache.set(key, value))
        self.assertTrue(self.hicache.exists(key))

        # Test get
        retrieved = self.hicache.get(key, dst_tensor)
        self.verify_tensors_equal(value, dst_tensor)
        self.verify_tensors_equal(value, retrieved)

        # Same test in addr,len mode with another key and dst_tensor
        key2 = "test_key2"
        dst_tensor2 = torch.zeros_like(value, device="cpu")
        src_addr, src_len = value.data_ptr(), value.numel() * value.element_size()
        dst_addr, dst_len = (
            dst_tensor2.data_ptr(),
            dst_tensor2.numel() * dst_tensor2.element_size(),
        )

        # Test set
        self.assertTrue(self.hicache.set(key, None, src_addr, src_len))
        self.assertTrue(self.hicache.exists(key))

        # Test get
        retrieved2 = self.hicache.get(key, dst_addr, dst_len)
        self.assertTrue(retrieved2 is None)
        self.verify_tensors_equal(value, dst_tensor2)

    def test_batch_set_get(self):
        """Test batch tensor set/get operations."""
        keys = ["key1", "key2", "key3"]
        values = [
            torch.randn(5, 5, device="cpu"),
            torch.randn(3, 3, device="cpu"),
            torch.randn(7, 7, device="cpu"),
        ]
        dst_tensors = [torch.zeros_like(v, device="cpu") for v in values]

        # Test batch set
        self.assertTrue(self.hicache.batch_set(keys, values))
        self.assertTrue(all(self.hicache.exists(key) for key in keys))

        # Test batch get
        retrieved = self.hicache.batch_get(keys, dst_tensors)
        self.verify_tensor_lists_equal(values, retrieved)

        # Same test in addr,len mode with another key and dst_tensor
        keys2 = ["key4", "key5", "key6"]
        dst_tensors2 = [torch.zeros_like(v, device="cpu") for v in values]
        src_addrs = [v.data_ptr() for v in values]
        src_lens = [v.numel() * v.element_size() for v in values]
        dst_addrs = [dt.data_ptr() for dt in dst_tensors2]
        dst_lens = [dt.numel() * dt.element_size() for dt in dst_tensors2]

        # Test batch set
        self.assertTrue(self.hicache.batch_set(keys2, None, src_addrs, src_lens))
        self.assertTrue(all(self.hicache.exists(key) for key in keys2))

        # Test batch get
        retrieved2 = self.hicache.batch_get(keys, dst_addrs, dst_lens)
        self.assertTrue(all(ret is None for ret in retrieved2))
        self.verify_tensor_lists_equal(values, dst_tensors2)

    def test_mixed_operations(self):
        """Test mixing single and batch operations."""
        # Test interleaved set/get operations
        key1, key2 = "key1", "key2"
        value1 = torch.randn(4, 4, device="cpu")
        value2 = torch.randn(6, 6, device="cpu")
        dst1 = torch.zeros_like(value1)
        dst2 = torch.zeros_like(value2)

        # Single set/get
        self.assertTrue(self.hicache.set(key1, value1))
        retrieved1 = self.hicache.get(key1, dst1)
        self.verify_tensors_equal(value1, retrieved1)

        # Batch set/get
        self.assertTrue(self.hicache.batch_set([key2], [value2]))
        retrieved2 = self.hicache.batch_get([key2], [dst2])
        self.verify_tensors_equal(value2, retrieved2[0])

    def test_data_integrity(self):
        """Test data integrity across operations."""
        # Test with various tensor types and sizes
        test_cases = [
            ("float32", torch.randn(10, 10, dtype=torch.float32)),
            ("float64", torch.randn(5, 5, dtype=torch.float64)),
            ("int32", torch.randint(-100, 100, (8, 8), dtype=torch.int32)),
            ("int64", torch.randint(-100, 100, (6, 6), dtype=torch.int64)),
            ("bool", torch.randint(0, 2, (4, 4)).bool()),
        ]

        for name, tensor in test_cases:
            with self.subTest(tensor_type=name):
                key = f"test_{name}"
                dst_tensor = torch.zeros_like(tensor)

                # Set and immediately get
                self.assertTrue(self.hicache.set(key, tensor))
                retrieved1 = self.hicache.get(key, dst_tensor)
                self.verify_tensors_equal(tensor, retrieved1)

                # Get again to verify persistence
                dst_tensor.zero_()
                retrieved2 = self.hicache.get(key, dst_tensor)
                self.verify_tensors_equal(tensor, retrieved2)

    def test_basic_file_operations(self):
        """Test basic file operations."""
        test_file = os.path.join(self.test_dir, "test_file.bin")
        self.file_manager.create_file(test_file)
        self.assertTrue(os.path.exists(test_file))
        self.assertEqual(os.path.getsize(test_file), 0)  # Empty file

        # Test file deletion
        self.assertTrue(self.delete_test_file(test_file))
        self.assertFalse(os.path.exists(test_file))

    def test_ready_marker_lifecycle(self):
        """Ready markers should be created, observed, and cleared correctly."""
        file_path = self.file_manager.get_file_path("abcdef0123456789")
        self.file_manager.create_file(file_path)

        self.assertFalse(self.file_manager.is_ready(file_path))
        self.assertTrue(self.file_manager.mark_ready(file_path))
        self.assertTrue(self.file_manager.is_ready(file_path))

        marker_path = self.file_manager.get_marker_path(file_path)
        self.assertTrue(os.path.exists(marker_path))

        self.file_manager.clear_ready(file_path)
        self.assertFalse(self.file_manager.is_ready(file_path))
        self.assertFalse(os.path.exists(marker_path))

    def test_commit_index_publish_and_load(self):
        """Committed versions should be atomically published and reloadable."""
        page_key = self.hicache._get_page_key("abcdef0123456789")
        version = self.commit_index.make_version_id()

        self.assertIsNone(self.commit_index.load_committed_version(page_key))
        self.assertTrue(
            self.commit_index.publish_committed_version(page_key, version, owner=1)
        )
        self.assertEqual(self.commit_index.load_committed_version(page_key), version)
        self.assertEqual(self.commit_index.load_entry(page_key)["owner"], 1)

    def test_default_file_layout_is_hashed2(self):
        """Default file layout should use a 2-level hashed directory structure."""
        key = "abcdef0123456789"
        expected = os.path.join(self.test_dir, "ab", "cd", key)

        self.assertEqual(self.file_manager.layout, "hashed2")
        self.assertEqual(self.file_manager.get_file_path(key), expected)

    def test_versioned_and_metadata_paths_follow_layout(self):
        """Versioned data and metadata files should follow the selected layout."""
        key = "abcdef0123456789"

        self.assertEqual(
            self.file_manager.get_versioned_file_path(key, "v1"),
            os.path.join(self.test_dir, "ab", "cd", f"{key}.v.v1"),
        )
        self.assertEqual(
            self.file_manager.get_metadata_path(key),
            os.path.join(self.test_dir, "ab", "cd", f"{key}.meta.json"),
        )

    def test_flat_file_layout(self):
        """Flat layout should place files directly under the base directory."""
        flat_manager = NixlFileManager(self.test_dir, layout="flat")
        key = "abcdef0123456789"

        self.assertEqual(
            flat_manager.get_file_path(key), os.path.join(self.test_dir, key)
        )

    def test_invalid_file_layout_raises(self):
        """Unknown file layouts should fail fast."""
        with self.assertRaises(ValueError):
            NixlFileManager(self.test_dir, layout="unknown")

    def test_create_nixl_tuples(self):
        """Test creation of NIXL tuples."""
        test_file = os.path.join(self.test_dir, "test_file.bin")
        self.file_manager.create_file(test_file)

        # Test tuple creation
        tuples = self.file_manager.files_to_nixl_tuples([test_file])
        self.assertIsNotNone(tuples)
        self.assertTrue(len(tuples) > 0)
        self.file_manager.close_nixl_tuples(tuples)

    def test_error_handling(self):
        """Test error handling in file operations."""
        # Test non-existent file
        self.assertTrue(
            self.delete_test_file("nonexistent_file.bin")
        )  # Returns True if file doesn't exist

        # Test invalid file path
        self.assertFalse(self.file_manager.create_file(""))  # Empty path should fail

    def test_register_buffers(self):
        """Test registration of memory buffers."""
        # Create test tensor
        tensor = torch.randn(10, 10)

        # Test buffer registration
        self.assertIsNotNone(self.hicache.register_buffers(tensor))

        # Test batch registration
        tensors = [torch.randn(5, 5) for _ in range(3)]
        self.assertIsNotNone(self.hicache.register_buffers(tensors))

    def test_register_files_closes_file_descriptors(self):
        """Test that register_files closes all opened file descriptors."""
        files = [os.path.join(self.test_dir, f"fd_test_file_{i}.bin") for i in range(3)]
        for file in files:
            self.file_manager.create_file(file)

        captured_fds = []

        def register_and_capture(items, mem_type):
            self.assertEqual(mem_type, "FILE")
            captured_fds.extend(item[2] for item in items)
            for fd in captured_fds:
                os.fstat(fd)
            return "mock_registered_memory"

        self.hicache.registration._register_memory = register_and_capture

        self.assertEqual(self.hicache.register_files(files), "mock_registered_memory")
        self.assertEqual(len(captured_fds), len(files))

        for fd in captured_fds:
            with self.assertRaises(OSError):
                os.fstat(fd)

    def test_register_files_with_tuples(self):
        """Test registration of files using file paths."""
        files = [os.path.join(self.test_dir, f"test_file_{i}.bin") for i in range(3)]
        for file in files:
            self.file_manager.create_file(file)

        self.assertIsNotNone(self.hicache.register_files(files))

    def test_runtime_only_params_are_not_passed_to_backend_init(self):
        """Runtime params should stay in HiCache and not leak into NIXL backend init."""
        config = NixlBackendConfig(
            {
                "numjobs": 8,
                "runtime": {"numjobs": 6},
                "plugin": {"posix": {"active": True, "use_uring": True}},
            }
        )

        initparams = config.get_backend_initparams("POSIX")

        self.assertEqual(initparams["active"], "True")
        self.assertEqual(initparams["use_uring"], "True")
        self.assertNotIn("numjobs", initparams)
        self.assertNotIn("runtime", initparams)
        self.assertEqual(config.get_runtime_param("numjobs", 4), 6)

    def test_hicache_default_numjobs_creates_agent_pools(self):
        """FILE mode should keep the configured numjobs but serialize through one lane."""
        self.assertEqual(self.hicache.num_jobs, 4)
        self.assertEqual(len(self.hicache.read_contexts), 1)
        self.assertEqual(len(self.hicache.write_contexts), 1)

    def test_hicache_runtime_numjobs_override(self):
        """Runtime numjobs override should control the transfer pool size."""
        storage_config = HiCacheStorageConfig(
            tp_rank=0,
            tp_size=2,
            pp_rank=0,
            pp_size=1,
            is_mla_model=False,
            enable_storage_metrics=False,
            is_page_first_layout=False,
            model_name="test_model",
            extra_config={
                "runtime": {"numjobs": 2},
                "plugin": {"posix": {"active": True}},
            },
        )

        hicache = HiCacheNixl(storage_config=storage_config)
        try:
            self.assertEqual(hicache.num_jobs, 2)
            self.assertEqual(len(hicache.read_contexts), 1)
            self.assertEqual(len(hicache.write_contexts), 1)
        finally:
            hicache.close()

    def test_same_page_has_stable_owner_rank(self):
        """A logical page should deterministically map to one owner TP rank."""
        page_key = self.hicache._get_page_key("logical_page")
        owner_rank = self.hicache._get_page_owner_rank(page_key)
        self.assertEqual(owner_rank, self.hicache._get_page_owner_rank(page_key))

        expanded = self.hicache._expand_page_keys(["logical_page"])
        for expanded_page_key in expanded:
            self.assertEqual(
                self.hicache._get_page_owner_rank(expanded_page_key),
                owner_rank,
            )

    def test_non_owner_write_is_skipped(self):
        """Non-owner ranks should not publish committed versions for a page."""
        storage_config = HiCacheStorageConfig(
            tp_rank=1,
            tp_size=2,
            pp_rank=0,
            pp_size=1,
            is_mla_model=False,
            enable_storage_metrics=False,
            is_page_first_layout=False,
            model_name="test_model",
            extra_config={"plugin": {"posix": {"active": True}}},
        )

        hicache = HiCacheNixl(storage_config=storage_config)
        try:
            key = "owner_controlled_page"
            page_key = hicache._get_page_key(key)
            owner_rank = hicache._get_page_owner_rank(page_key)
            if owner_rank == hicache.tp_rank:
                self.skipTest("Chosen key hashes to local owner rank")
            self.assertTrue(hicache.set(key, torch.randn(2, 2)))
            self.assertEqual(hicache.batch_exists([key]), 0)
        finally:
            hicache.close()

    def test_publish_enqueues_previous_version_for_gc(self):
        """Publishing a new committed version should queue the previous one for lazy GC."""
        page_key = self.hicache._get_page_key("gc_page")
        storage_key = page_key
        old_version = "oldversion"
        new_version = "newversion"
        old_file = self.file_manager.get_versioned_file_path(storage_key, old_version)
        self.file_manager.create_file(old_file)
        self.assertTrue(
            self.commit_index.publish_committed_version(
                page_key, old_version, owner=self.hicache.tp_rank
            )
        )

        self.assertTrue(
            self.hicache._publish_page_versions(
                {page_key: new_version},
                {page_key: self.hicache.tp_rank},
                {page_key: [storage_key]},
            )
        )
        self.assertTrue(
            any(path == old_file for _, path in self.hicache._pending_gc),
            "Expected stale version to be queued for lazy GC",
        )

    def test_non_owner_write_does_not_create_orphaned_version_files(self):
        """Non-owner writes should not create versioned files that can never be committed."""
        storage_config = HiCacheStorageConfig(
            tp_rank=1,
            tp_size=2,
            pp_rank=0,
            pp_size=1,
            is_mla_model=False,
            enable_storage_metrics=False,
            is_page_first_layout=False,
            model_name="test_model",
            extra_config={"plugin": {"posix": {"active": True}}},
        )
        hicache = HiCacheNixl(storage_config=storage_config)
        try:
            hicache._execute_transfer = (
                lambda buffers, keys, direction, ctx, lock_held=False: True
            )
            key = "non_owner_orphan_page"
            page_key = hicache._get_page_key(key)
            if hicache._get_page_owner_rank(page_key) == hicache.tp_rank:
                self.skipTest("Chosen key hashes to local owner rank")
            self.assertTrue(hicache.set(key, torch.randn(2, 2)))
            versioned_paths = []
            for root, _, files in os.walk(self.test_dir):
                for file_name in files:
                    if ".v." in file_name:
                        versioned_paths.append(os.path.join(root, file_name))
            self.assertEqual(versioned_paths, [])
        finally:
            hicache.close()

    def test_startup_gc_scan_removes_stale_stable_versions(self):
        """Startup GC scan should delete old uncommitted versioned files."""
        page_key = self.hicache._get_page_key("startup_gc_page")
        storage_key = page_key
        committed_version = "committed"
        stale_version = "stale"
        committed_path = self.file_manager.get_versioned_file_path(
            storage_key, committed_version
        )
        stale_path = self.file_manager.get_versioned_file_path(
            storage_key, stale_version
        )
        self.file_manager.create_file(committed_path)
        self.file_manager.create_file(stale_path)
        self.assertTrue(
            self.commit_index.publish_committed_version(
                page_key,
                committed_version,
                owner=self.hicache.tp_rank,
                storage_keys=[storage_key],
            )
        )

        storage_config = HiCacheStorageConfig(
            tp_rank=0,
            tp_size=1,
            pp_rank=0,
            pp_size=1,
            is_mla_model=False,
            enable_storage_metrics=False,
            is_page_first_layout=False,
            model_name="test_model",
            extra_config={
                "runtime": {"file_gc_grace_seconds": 0},
                "plugin": {"posix": {"active": True}},
            },
        )

        hicache = HiCacheNixl(storage_config=storage_config)
        try:
            hicache.register_mem_pool_host(_FakeHostKVCache(page_size=1))
            self.assertTrue(os.path.exists(committed_path))
            self.assertFalse(os.path.exists(stale_path))
            stats = hicache.get_stats()
            self.assertGreaterEqual(stats["stale_version_deletions"], 1)
        finally:
            hicache.close()

    def test_startup_gc_scan_preserves_committed_files_without_storage_keys(self):
        """Startup scan must preserve older metadata files that do not record storage_keys."""
        page_key = self.hicache._get_page_key("legacy_gc_page")
        committed_version = "legacy"
        committed_path = self.file_manager.get_versioned_file_path(
            page_key, committed_version
        )
        self.file_manager.create_file(committed_path)
        meta_path = self.file_manager.get_metadata_path(page_key)
        os.makedirs(os.path.dirname(meta_path), exist_ok=True)
        with open(meta_path, "w", encoding="utf-8") as f:
            f.write(
                f'{{"owner": {self.hicache.tp_rank}, "version": "{committed_version}"}}'
            )

        storage_config = HiCacheStorageConfig(
            tp_rank=0,
            tp_size=1,
            pp_rank=0,
            pp_size=1,
            is_mla_model=False,
            enable_storage_metrics=False,
            is_page_first_layout=False,
            model_name="test_model",
            extra_config={
                "runtime": {"file_gc_grace_seconds": 0},
                "plugin": {"posix": {"active": True}},
            },
        )

        hicache = HiCacheNixl(storage_config=storage_config)
        try:
            self.assertTrue(os.path.exists(committed_path))
        finally:
            hicache.close()

    def test_startup_gc_scan_is_deferred_until_mem_pool_registration(self):
        """FILE startup GC scan should wait until zero-copy layout is known."""
        storage_config = HiCacheStorageConfig(
            tp_rank=0,
            tp_size=1,
            pp_rank=0,
            pp_size=1,
            is_mla_model=False,
            enable_storage_metrics=False,
            is_page_first_layout=True,
            model_name="test_model",
            extra_config={
                "runtime": {"file_gc_grace_seconds": 0},
                "plugin": {"posix": {"active": True}},
            },
        )
        hicache = HiCacheNixl(storage_config=storage_config)
        try:
            self.assertTrue(hicache._startup_gc_scan_pending)
            hicache.register_mem_pool_host(_FakeHostKVCache(page_size=1))
            self.assertFalse(hicache._startup_gc_scan_pending)
        finally:
            hicache.close()

    def test_hicache_default_file_layout_is_hashed2(self):
        """HiCache should default to the hashed2 NIXL file layout."""
        self.assertEqual(self.hicache.file_manager.layout, "hashed2")

    def test_hicache_runtime_file_layout_override(self):
        """Runtime file_layout override should be propagated to NixlFileManager."""
        storage_config = HiCacheStorageConfig(
            tp_rank=0,
            tp_size=2,
            pp_rank=0,
            pp_size=1,
            is_mla_model=False,
            enable_storage_metrics=False,
            is_page_first_layout=False,
            model_name="test_model",
            extra_config={
                "runtime": {"file_layout": "flat"},
                "plugin": {"posix": {"active": True}},
            },
        )

        hicache = HiCacheNixl(storage_config=storage_config)
        try:
            self.assertEqual(hicache.file_manager.layout, "flat")
        finally:
            hicache.close()

    def test_batch_exists_uses_committed_versions_for_file_backend(self):
        """FILE-mode existence should depend on committed metadata, not bare files."""
        key = "abcdef0123456789"
        page_key = self.hicache._get_page_key(key)

        self.assertEqual(self.hicache.batch_exists([key]), 0)

        version = self.commit_index.make_version_id()
        self.assertTrue(self.commit_index.publish_committed_version(page_key, version))
        self.assertEqual(self.hicache.batch_exists([key]), 1)

    def test_file_mode_reports_stats_for_skipped_publishes(self):
        """Skipped non-owner writes and committed publishes should surface in stats."""
        storage_config = HiCacheStorageConfig(
            tp_rank=1,
            tp_size=2,
            pp_rank=0,
            pp_size=1,
            is_mla_model=False,
            enable_storage_metrics=False,
            is_page_first_layout=False,
            model_name="test_model",
            extra_config={"plugin": {"posix": {"active": True}}},
        )
        hicache = HiCacheNixl(storage_config=storage_config)
        try:
            hicache._execute_transfer = (
                lambda buffers, keys, direction, ctx, lock_held=False: True
            )
            key = "non_owner_stats_page"
            page_key = hicache._get_page_key(key)
            if hicache._get_page_owner_rank(page_key) == hicache.tp_rank:
                self.skipTest("Chosen key hashes to local owner rank")
            self.assertTrue(hicache.set(key, torch.randn(2, 2)))
            stats = hicache.get_stats()
            self.assertEqual(stats["committed_version_publishes"], 0)
            self.assertGreaterEqual(stats["skipped_non_owner_writes"], 1)
        finally:
            hicache.close()

    def test_concurrent_batch_get_set_v1_file_mode_stress(self):
        """Concurrent FILE-mode v1 get/set should preserve committed metadata semantics."""
        storage_config = HiCacheStorageConfig(
            tp_rank=0,
            tp_size=1,
            pp_rank=0,
            pp_size=1,
            is_mla_model=False,
            enable_storage_metrics=False,
            is_page_first_layout=False,
            model_name="test_model",
            extra_config={
                "runtime": {"file_gc_grace_seconds": 0},
                "plugin": {"posix": {"active": True}},
            },
        )
        hicache = HiCacheNixl(storage_config=storage_config)
        try:
            hicache.register_mem_pool_host(_FakeHostKVCache(page_size=1))
            hicache._execute_transfer = (
                lambda buffers, keys, direction, ctx, lock_held=False: True
            )

            keys = [f"stress_page_{i}" for i in range(4)]
            host_indices = torch.arange(len(keys), dtype=torch.int64)
            self.assertTrue(all(hicache.batch_set_v1(keys, host_indices)))

            def _writer(iteration: int):
                return hicache.batch_set_v1(keys, host_indices + iteration)

            def _reader():
                return hicache.batch_get_v1(keys, host_indices)

            with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
                futures = []
                for iteration in range(6):
                    futures.append(executor.submit(_writer, iteration))
                    futures.append(executor.submit(_reader))
                results = [future.result() for future in futures]

            for result in results:
                self.assertEqual(len(result), len(keys))
                self.assertTrue(all(isinstance(item, bool) for item in result))

            self.assertEqual(hicache.batch_exists(keys), len(keys))
            stats = hicache.get_stats()
            self.assertGreaterEqual(stats["committed_version_publishes"], len(keys))
            self.assertGreaterEqual(stats["stale_version_deletions"], 1)
            self.assertGreaterEqual(
                stats["pending_gc_queue_max"], stats["pending_gc_queue_len"]
            )
        finally:
            hicache.close()

    def test_multi_process_owner_rank_publish_behavior(self):
        """Across TP processes, only the owner rank should publish a committed version."""
        ctx = multiprocessing.get_context("spawn")
        key = "multi_process_owner_page"
        queue = ctx.Queue()
        processes = [
            ctx.Process(
                target=_owner_publish_worker,
                args=(self.test_dir, rank, 2, key, queue),
            )
            for rank in range(2)
        ]

        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=30)
            self.assertEqual(process.exitcode, 0)

        results = [queue.get(timeout=5) for _ in processes]
        owner_rank = results[0]["owner_rank"]
        self.assertTrue(all(result["owner_rank"] == owner_rank for result in results))

        publish_counts = {
            result["rank"]: result["stats"]["committed_version_publishes"]
            for result in results
        }
        skip_counts = {
            result["rank"]: result["stats"]["skipped_non_owner_writes"]
            for result in results
        }
        self.assertEqual(publish_counts[owner_rank], 1)
        self.assertEqual(skip_counts[owner_rank], 0)
        self.assertEqual(publish_counts[1 - owner_rank], 0)
        self.assertGreaterEqual(skip_counts[1 - owner_rank], 1)


if __name__ == "__main__":
    unittest.main()
