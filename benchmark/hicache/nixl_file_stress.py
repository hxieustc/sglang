from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import multiprocessing
import os
import random
import shutil
import statistics
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig
from sglang.srt.mem_cache.storage.nixl.hicache_nixl import HiCacheNixl


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stress and benchmark HiCache NIXL FILE/POSIX with randomized "
            "multi-threaded and multi-process page reads/writes."
        )
    )
    parser.add_argument("--storage-dir", default="/tmp/sglang_nixl_bench")
    parser.add_argument("--clean-storage-dir", action="store_true")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--threads-per-rank", type=int, default=2)
    parser.add_argument("--ops-per-thread", type=int, default=100)
    parser.add_argument("--prefill-pages", type=int, default=64)
    parser.add_argument("--num-pages", type=int, default=1024)
    parser.add_argument("--batch-size-min", type=int, default=1)
    parser.add_argument("--batch-size-max", type=int, default=8)
    parser.add_argument("--write-ratio", type=float, default=0.4)
    parser.add_argument("--read-ratio", type=float, default=0.5)
    parser.add_argument("--exists-ratio", type=float, default=0.1)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--verify-sample-rate", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--model-type", choices=["mha", "mla"], default="mha")
    parser.add_argument(
        "--layout",
        choices=["layer_first", "page_first"],
        default="page_first",
        help="page_first exercises zero-copy, layer_first uses the non-zero-copy path.",
    )
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=16)
    parser.add_argument("--mha-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--mla-kv-dim", type=int, default=576)
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
    )
    parser.add_argument("--numjobs", type=int, default=1)
    parser.add_argument("--file-layout", choices=["hashed2", "flat"], default="hashed2")
    parser.add_argument("--file-gc-grace-seconds", type=float, default=300.0)
    parser.add_argument("--model-name", default="bench_model")
    parser.add_argument(
        "--log-every",
        type=int,
        default=0,
        help="Print per-thread progress every N operations; 0 disables progress logs.",
    )
    args = parser.parse_args()
    ratio_sum = args.write_ratio + args.read_ratio + args.exists_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        parser.error("write-ratio + read-ratio + exists-ratio must equal 1.0")
    if args.batch_size_min <= 0 or args.batch_size_max < args.batch_size_min:
        parser.error("invalid batch size range")
    if args.tp_size <= 0 or args.threads_per_rank <= 0 or args.ops_per_thread <= 0:
        parser.error("tp-size, threads-per-rank, and ops-per-thread must be positive")
    if not (0.0 <= args.verify_sample_rate <= 1.0):
        parser.error("verify-sample-rate must be in [0, 1]")
    return args


def _dtype_from_name(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def _build_host_indices(slot_indices: List[int], page_size: int) -> torch.Tensor:
    values: List[int] = []
    for slot in slot_indices:
        start = slot * page_size
        values.extend(range(start, start + page_size))
    return torch.tensor(values, dtype=torch.int64)


def _logical_key(page_id: int) -> str:
    return f"page_{page_id:016x}"


def _blake2_tensor(tensor: torch.Tensor) -> str:
    data = tensor.contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.blake2b(data, digest_size=16).hexdigest()


class _BenchmarkHostKVCacheBase:
    def __init__(self, page_size: int, layout: str, dtype: torch.dtype):
        self.page_size = page_size
        self.layout = layout
        self.dtype = dtype
        self._lock = threading.RLock()

    def get_page_buffer_meta(self, indices):
        raise NotImplementedError

    def get_data_page(self, index: int, flat: bool = True) -> torch.Tensor:
        raise NotImplementedError

    def get_dummy_flat_data_page(self) -> torch.Tensor:
        raise NotImplementedError

    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
        raise NotImplementedError

    def fill_slots(self, slot_indices: List[int], write_tokens: List[int]) -> None:
        raise NotImplementedError

    def hash_slot(self, slot_index: int) -> str:
        raise NotImplementedError


class _BenchmarkMHAHostKVCache(_BenchmarkHostKVCacheBase):
    def __init__(
        self,
        slot_count: int,
        page_size: int,
        num_layers: int,
        head_num: int,
        head_dim: int,
        layout: str,
        dtype: torch.dtype,
    ):
        super().__init__(page_size=page_size, layout=layout, dtype=dtype)
        self.slot_count = slot_count
        self.num_layers = num_layers
        self.head_num = head_num
        self.head_dim = head_dim
        self.k_pages = torch.zeros(
            (slot_count, num_layers, page_size, head_num, head_dim), dtype=dtype
        )
        self.v_pages = torch.zeros_like(self.k_pages)

    def get_page_buffer_meta(self, indices):
        ptr_list = []
        size_list = []
        for idx in indices[:: self.page_size]:
            page_idx = int(idx) // self.page_size
            ptr_list.append(self.k_pages[page_idx].data_ptr())
            size_list.append(
                self.k_pages[page_idx].numel() * self.k_pages[page_idx].element_size()
            )
            ptr_list.append(self.v_pages[page_idx].data_ptr())
            size_list.append(
                self.v_pages[page_idx].numel() * self.v_pages[page_idx].element_size()
            )
        return ptr_list, size_list

    def get_data_page(self, index: int, flat: bool = True) -> torch.Tensor:
        page_idx = int(index) // self.page_size
        with self._lock:
            return torch.stack(
                [self.k_pages[page_idx], self.v_pages[page_idx]], dim=0
            ).clone()

    def get_dummy_flat_data_page(self) -> torch.Tensor:
        return torch.zeros(
            (2, self.num_layers, self.page_size, self.head_num, self.head_dim),
            dtype=self.dtype,
        )

    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
        page_idx = int(index) // self.page_size
        with self._lock:
            self.k_pages[page_idx].copy_(data_page[0])
            self.v_pages[page_idx].copy_(data_page[1])

    def fill_slots(self, slot_indices: List[int], write_tokens: List[int]) -> None:
        with self._lock:
            for slot, token in zip(slot_indices, write_tokens):
                k_value = float((token % 1024) + 1)
                v_value = float(-((token % 1024) + 1))
                self.k_pages[slot].fill_(k_value)
                self.v_pages[slot].fill_(v_value)

    def hash_slot(self, slot_index: int) -> str:
        with self._lock:
            h = hashlib.blake2b(digest_size=16)
            h.update(
                self.k_pages[slot_index]
                .contiguous()
                .view(torch.uint8)
                .numpy()
                .tobytes()
            )
            h.update(
                self.v_pages[slot_index]
                .contiguous()
                .view(torch.uint8)
                .numpy()
                .tobytes()
            )
            return h.hexdigest()


class _BenchmarkMLAHostKVCache(_BenchmarkHostKVCacheBase):
    def __init__(
        self,
        slot_count: int,
        page_size: int,
        num_layers: int,
        kv_dim: int,
        layout: str,
        dtype: torch.dtype,
    ):
        super().__init__(page_size=page_size, layout=layout, dtype=dtype)
        self.slot_count = slot_count
        self.num_layers = num_layers
        self.kv_dim = kv_dim
        self.pages = torch.zeros(
            (slot_count, num_layers, page_size, 1, kv_dim), dtype=dtype
        )

    def get_page_buffer_meta(self, indices):
        ptr_list = []
        size_list = []
        for idx in indices[:: self.page_size]:
            page_idx = int(idx) // self.page_size
            ptr_list.append(self.pages[page_idx].data_ptr())
            size_list.append(
                self.pages[page_idx].numel() * self.pages[page_idx].element_size()
            )
        return ptr_list, size_list

    def get_data_page(self, index: int, flat: bool = True) -> torch.Tensor:
        page_idx = int(index) // self.page_size
        with self._lock:
            return self.pages[page_idx].clone()

    def get_dummy_flat_data_page(self) -> torch.Tensor:
        return torch.zeros(
            (self.num_layers, self.page_size, 1, self.kv_dim), dtype=self.dtype
        )

    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
        page_idx = int(index) // self.page_size
        with self._lock:
            self.pages[page_idx].copy_(data_page)

    def fill_slots(self, slot_indices: List[int], write_tokens: List[int]) -> None:
        with self._lock:
            for slot, token in zip(slot_indices, write_tokens):
                value = float((token % 2048) + 1)
                self.pages[slot].fill_(value)

    def hash_slot(self, slot_index: int) -> str:
        with self._lock:
            return _blake2_tensor(self.pages[slot_index])


@dataclass
class OperationMetrics:
    count: int = 0
    failures: int = 0
    latency_ms: List[float] = field(default_factory=list)
    bytes_transferred: int = 0


@dataclass
class RankMetrics:
    writes: OperationMetrics = field(default_factory=OperationMetrics)
    reads: OperationMetrics = field(default_factory=OperationMetrics)
    exists: OperationMetrics = field(default_factory=OperationMetrics)
    read_verify_ok: int = 0
    read_verify_fail: int = 0
    read_verify_skipped: int = 0
    exists_verify_ok: int = 0
    exists_verify_fail: int = 0
    exists_verify_skipped: int = 0
    startup_time_s: float = 0.0
    total_time_s: float = 0.0
    backend_stats: Optional[dict] = None


def _page_bytes(args: argparse.Namespace, dtype: torch.dtype) -> int:
    itemsize = torch.empty((), dtype=dtype).element_size()
    if args.model_type == "mha":
        return (
            2
            * args.num_layers
            * args.page_size
            * args.mha_heads
            * args.head_dim
            * itemsize
        )
    return args.num_layers * args.page_size * args.mla_kv_dim * itemsize


def _make_host_pool(args: argparse.Namespace, slot_count: int, dtype: torch.dtype):
    if args.model_type == "mha":
        return _BenchmarkMHAHostKVCache(
            slot_count=slot_count,
            page_size=args.page_size,
            num_layers=args.num_layers,
            head_num=args.mha_heads,
            head_dim=args.head_dim,
            layout=args.layout,
            dtype=dtype,
        )
    return _BenchmarkMLAHostKVCache(
        slot_count=slot_count,
        page_size=args.page_size,
        num_layers=args.num_layers,
        kv_dim=args.mla_kv_dim,
        layout=args.layout,
        dtype=dtype,
    )


def _oracle_snapshot(
    oracle_lock, oracle, page_keys: List[str]
) -> List[Optional[tuple[int, str]]]:
    with oracle_lock:
        return [oracle.get(page_key) for page_key in page_keys]


def _oracle_expected_exists(snapshot: List[Optional[tuple[int, str]]]) -> int:
    count = 0
    for entry in snapshot:
        if entry is None:
            return count
        count += 1
    return count


def _record_latency(metrics: OperationMetrics, latency_s: float) -> None:
    metrics.count += 1
    metrics.latency_ms.append(latency_s * 1000.0)


def _sample_slots(thread_idx: int, batch_size: int, slots_per_thread: int) -> List[int]:
    base = thread_idx * slots_per_thread
    return [base + i for i in range(batch_size)]


def _run_prefill(
    hicache: HiCacheNixl,
    host_pool,
    rank: int,
    args: argparse.Namespace,
    oracle,
    oracle_lock,
    dtype: torch.dtype,
) -> None:
    if args.prefill_pages <= 0:
        return
    slots_per_round = max(args.batch_size_max, 1)
    next_token = 1 + rank * 1_000_000
    page_bytes = _page_bytes(args, dtype)
    remaining = min(args.prefill_pages, args.num_pages)
    page_id = 0
    while remaining > 0:
        batch_size = min(slots_per_round, remaining)
        slot_indices = list(range(batch_size))
        keys = [_logical_key(page_id + i) for i in range(batch_size)]
        page_keys = [hicache._get_page_key(key) for key in keys]
        write_tokens = [next_token + i for i in range(batch_size)]
        host_pool.fill_slots(slot_indices, write_tokens)
        results = hicache.batch_set_v1(
            keys, _build_host_indices(slot_indices, args.page_size)
        )
        for key, page_key, slot, result in zip(keys, page_keys, slot_indices, results):
            if not result:
                continue
            if hicache._get_page_owner_rank(page_key) != rank:
                continue
            payload_hash = host_pool.hash_slot(slot)
            with oracle_lock:
                version = oracle.get(page_key, (0, ""))[0] + 1
                oracle[page_key] = (version, payload_hash)
        remaining -= batch_size
        page_id += batch_size
        next_token += batch_size
        _ = page_bytes


def _thread_worker(
    rank: int,
    thread_idx: int,
    hicache: HiCacheNixl,
    host_pool,
    args: argparse.Namespace,
    dtype: torch.dtype,
    oracle,
    oracle_lock,
) -> RankMetrics:
    rng = random.Random(args.seed + rank * 10_000 + thread_idx)
    metrics = RankMetrics()
    slots_per_thread = max(args.batch_size_max * 2, 1)
    next_write_token = (rank + 1) * 1_000_000_000 + thread_idx * 1_000_000
    page_bytes = _page_bytes(args, dtype)

    for op_idx in range(args.ops_per_thread):
        if args.log_every and (op_idx + 1) % args.log_every == 0:
            print(
                f"[rank={rank} thread={thread_idx}] completed {op_idx + 1}/{args.ops_per_thread} ops",
                flush=True,
            )

        batch_size = rng.randint(args.batch_size_min, args.batch_size_max)
        page_ids = [rng.randrange(args.num_pages) for _ in range(batch_size)]
        keys = [_logical_key(page_id) for page_id in page_ids]
        page_keys = [hicache._get_page_key(key) for key in keys]
        slot_indices = _sample_slots(thread_idx, batch_size, slots_per_thread)
        host_indices = _build_host_indices(slot_indices, args.page_size)
        op_roll = rng.random()

        if op_roll < args.write_ratio:
            write_tokens = [next_write_token + i for i in range(batch_size)]
            next_write_token += batch_size
            host_pool.fill_slots(slot_indices, write_tokens)
            start = time.perf_counter()
            results = hicache.batch_set_v1(keys, host_indices)
            elapsed = time.perf_counter() - start
            _record_latency(metrics.writes, elapsed)
            metrics.writes.bytes_transferred += batch_size * page_bytes
            if not all(results):
                metrics.writes.failures += 1

            if args.verify:
                for page_key, slot, result in zip(page_keys, slot_indices, results):
                    if not result:
                        continue
                    if hicache._get_page_owner_rank(page_key) != rank:
                        continue
                    payload_hash = host_pool.hash_slot(slot)
                    with oracle_lock:
                        version = oracle.get(page_key, (0, ""))[0] + 1
                        oracle[page_key] = (version, payload_hash)
            continue

        if op_roll < args.write_ratio + args.read_ratio:
            pre_snapshot = None
            if args.verify and rng.random() <= args.verify_sample_rate:
                pre_snapshot = _oracle_snapshot(oracle_lock, oracle, page_keys)

            start = time.perf_counter()
            results = hicache.batch_get_v1(keys, host_indices)
            elapsed = time.perf_counter() - start
            _record_latency(metrics.reads, elapsed)
            metrics.reads.bytes_transferred += batch_size * page_bytes
            if not all(results):
                metrics.reads.failures += 1

            if pre_snapshot is not None:
                post_snapshot = _oracle_snapshot(oracle_lock, oracle, page_keys)
                if pre_snapshot != post_snapshot:
                    metrics.read_verify_skipped += 1
                else:
                    verified = True
                    for entry, result, slot in zip(pre_snapshot, results, slot_indices):
                        if entry is None:
                            if result:
                                verified = False
                                break
                            continue
                        if not result:
                            verified = False
                            break
                        if host_pool.hash_slot(slot) != entry[1]:
                            verified = False
                            break
                    if verified:
                        metrics.read_verify_ok += 1
                    else:
                        metrics.read_verify_fail += 1
            continue

        pre_snapshot = None
        if args.verify and rng.random() <= args.verify_sample_rate:
            pre_snapshot = _oracle_snapshot(oracle_lock, oracle, page_keys)

        start = time.perf_counter()
        exists_count = hicache.batch_exists(keys)
        elapsed = time.perf_counter() - start
        _record_latency(metrics.exists, elapsed)
        metrics.exists.bytes_transferred += batch_size * page_bytes

        if pre_snapshot is not None:
            post_snapshot = _oracle_snapshot(oracle_lock, oracle, page_keys)
            if pre_snapshot != post_snapshot:
                metrics.exists_verify_skipped += 1
            else:
                expected = _oracle_expected_exists(pre_snapshot)
                if exists_count == expected:
                    metrics.exists_verify_ok += 1
                else:
                    metrics.exists_verify_fail += 1

    return metrics


def _merge_metrics(metrics_list: List[RankMetrics]) -> RankMetrics:
    merged = RankMetrics()
    for metrics in metrics_list:
        for field_name in ("writes", "reads", "exists"):
            src = getattr(metrics, field_name)
            dst = getattr(merged, field_name)
            dst.count += src.count
            dst.failures += src.failures
            dst.latency_ms.extend(src.latency_ms)
            dst.bytes_transferred += src.bytes_transferred
        merged.read_verify_ok += metrics.read_verify_ok
        merged.read_verify_fail += metrics.read_verify_fail
        merged.read_verify_skipped += metrics.read_verify_skipped
        merged.exists_verify_ok += metrics.exists_verify_ok
        merged.exists_verify_fail += metrics.exists_verify_fail
        merged.exists_verify_skipped += metrics.exists_verify_skipped
    return merged


def _rank_worker(
    rank: int, args: argparse.Namespace, oracle, oracle_lock, result_queue
) -> None:
    os.environ["SGLANG_HICACHE_NIXL_BACKEND_STORAGE_DIR"] = args.storage_dir
    dtype = _dtype_from_name(args.dtype)
    startup_start = time.perf_counter()
    storage_config = HiCacheStorageConfig(
        tp_rank=rank,
        tp_size=args.tp_size,
        pp_rank=0,
        pp_size=1,
        is_mla_model=args.model_type == "mla",
        enable_storage_metrics=False,
        is_page_first_layout=args.layout == "page_first",
        model_name=args.model_name,
        extra_config={
            "runtime": {
                "numjobs": args.numjobs,
                "file_layout": args.file_layout,
                "file_gc_grace_seconds": args.file_gc_grace_seconds,
            },
            "plugin": {"posix": {"active": True}},
        },
    )

    hicache = HiCacheNixl(storage_config=storage_config)
    try:
        slot_count = args.threads_per_rank * max(args.batch_size_max * 2, 1)
        host_pool = _make_host_pool(args, slot_count=slot_count, dtype=dtype)
        hicache.register_mem_pool_host(host_pool)
        startup_time_s = time.perf_counter() - startup_start

        _run_prefill(hicache, host_pool, rank, args, oracle, oracle_lock, dtype)

        wall_start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.threads_per_rank,
            thread_name_prefix=f"rank{rank}_bench",
        ) as executor:
            futures = [
                executor.submit(
                    _thread_worker,
                    rank,
                    thread_idx,
                    hicache,
                    host_pool,
                    args,
                    dtype,
                    oracle,
                    oracle_lock,
                )
                for thread_idx in range(args.threads_per_rank)
            ]
            merged = _merge_metrics([future.result() for future in futures])
        merged.startup_time_s = startup_time_s
        merged.total_time_s = time.perf_counter() - wall_start
        merged.backend_stats = hicache.get_stats()
        result_queue.put((rank, merged))
    finally:
        hicache.close()


def _latency_summary(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"avg_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0}
    ordered = sorted(values)
    return {
        "avg_ms": statistics.fmean(ordered),
        "p50_ms": ordered[int(0.50 * (len(ordered) - 1))],
        "p95_ms": ordered[int(0.95 * (len(ordered) - 1))],
        "p99_ms": ordered[int(0.99 * (len(ordered) - 1))],
    }


def _print_metrics(results: Dict[int, RankMetrics]) -> None:
    total = _merge_metrics(list(results.values()))
    total_wall = max(
        (metrics.total_time_s for metrics in results.values()), default=0.0
    )
    print("\n=== HiCache NIXL FILE/POSIX Benchmark Summary ===")
    print(f"Ranks: {len(results)}")
    print(f"Total wall time: {total_wall:.3f}s")
    for rank, metrics in sorted(results.items()):
        print(
            f"\n[rank {rank}] startup={metrics.startup_time_s:.3f}s run={metrics.total_time_s:.3f}s"
        )
        for name in ("writes", "reads", "exists"):
            op = getattr(metrics, name)
            lat = _latency_summary(op.latency_ms)
            throughput = (
                op.count / metrics.total_time_s if metrics.total_time_s > 0 else 0.0
            )
            mibps = (
                op.bytes_transferred / (1024 * 1024) / metrics.total_time_s
                if metrics.total_time_s > 0
                else 0.0
            )
            print(
                f"  {name:>6}: count={op.count:5d} failures={op.failures:4d} "
                f"ops/s={throughput:8.2f} MiB/s={mibps:10.2f} "
                f"lat(avg/p50/p95/p99)={lat['avg_ms']:.3f}/{lat['p50_ms']:.3f}/{lat['p95_ms']:.3f}/{lat['p99_ms']:.3f} ms"
            )
        if metrics.backend_stats is not None:
            print(f"  backend_stats={metrics.backend_stats}")

    print("\n[overall]")
    for name in ("writes", "reads", "exists"):
        op = getattr(total, name)
        lat = _latency_summary(op.latency_ms)
        throughput = op.count / total_wall if total_wall > 0 else 0.0
        mibps = (
            op.bytes_transferred / (1024 * 1024) / total_wall if total_wall > 0 else 0.0
        )
        print(
            f"  {name:>6}: count={op.count:5d} failures={op.failures:4d} "
            f"ops/s={throughput:8.2f} MiB/s={mibps:10.2f} "
            f"lat(avg/p50/p95/p99)={lat['avg_ms']:.3f}/{lat['p50_ms']:.3f}/{lat['p95_ms']:.3f}/{lat['p99_ms']:.3f} ms"
        )
    print(
        "  verify_reads="
        f"ok:{total.read_verify_ok} fail:{total.read_verify_fail} skipped:{total.read_verify_skipped}"
    )
    print(
        "  verify_exists="
        f"ok:{total.exists_verify_ok} fail:{total.exists_verify_fail} skipped:{total.exists_verify_skipped}"
    )


def main() -> None:
    args = _parse_args()
    if args.clean_storage_dir and os.path.exists(args.storage_dir):
        shutil.rmtree(args.storage_dir, ignore_errors=True)
    os.makedirs(args.storage_dir, exist_ok=True)

    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    manager = None
    if args.verify and args.tp_size > 1:
        manager = ctx.Manager()
        oracle = manager.dict()
        oracle_lock = manager.RLock()
    else:
        oracle = {}
        oracle_lock = ctx.RLock()

    print("Starting benchmark with configuration:")
    for key, value in sorted(vars(args).items()):
        print(f"  {key}: {value}")

    processes = [
        ctx.Process(
            target=_rank_worker,
            args=(rank, args, oracle, oracle_lock, result_queue),
            name=f"nixl-bench-rank{rank}",
        )
        for rank in range(args.tp_size)
    ]

    global_start = time.perf_counter()
    for process in processes:
        process.start()

    results: Dict[int, RankMetrics] = {}
    for _ in processes:
        rank, metrics = result_queue.get()
        results[rank] = metrics

    for process in processes:
        process.join(timeout=120)
        if process.exitcode != 0:
            raise SystemExit(
                f"Process {process.name} exited with code {process.exitcode}"
            )

    if manager is not None:
        manager.shutdown()

    total_elapsed = time.perf_counter() - global_start
    _print_metrics(results)
    print(f"\nCompleted in {total_elapsed:.3f}s")


if __name__ == "__main__":
    main()
