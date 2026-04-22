import argparse
import os
import shutil
import statistics
import tempfile
import time
from dataclasses import dataclass
from typing import Iterable, List


@dataclass
class BatchStats:
    batch_size: int
    iterations: int
    open_ms_mean: float
    open_ms_p50: float
    open_ms_p95: float
    close_ms_mean: float
    close_ms_p50: float
    close_ms_p95: float
    close_us_per_fd_mean: float
    total_ms_mean: float
    close_share_pct: float


def parse_batch_sizes(batch_sizes: str) -> List[int]:
    sizes = []
    for item in batch_sizes.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError(f"Batch size must be positive, got {value}")
        sizes.append(value)
    if not sizes:
        raise ValueError("At least one batch size is required")
    return sizes


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    values = sorted(values)
    idx = (len(values) - 1) * pct
    lo = int(idx)
    hi = min(lo + 1, len(values) - 1)
    frac = idx - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def create_files(base_dir: str, count: int) -> List[str]:
    paths = []
    for idx in range(count):
        path = os.path.join(base_dir, f"fd_bench_{idx:08d}.bin")
        with open(path, "wb"):
            pass
        paths.append(path)
    return paths


def batched_open(paths: Iterable[str]) -> List[int]:
    return [os.open(path, os.O_RDWR) for path in paths]


def batched_close(fds: Iterable[int]) -> None:
    for fd in fds:
        os.close(fd)


def run_benchmark(paths: List[str], iterations: int) -> BatchStats:
    open_times = []
    close_times = []
    total_times = []

    for _ in range(iterations):
        start_open = time.perf_counter_ns()
        fds = batched_open(paths)
        end_open = time.perf_counter_ns()

        start_close = time.perf_counter_ns()
        batched_close(fds)
        end_close = time.perf_counter_ns()

        open_ns = end_open - start_open
        close_ns = end_close - start_close
        total_ns = end_close - start_open

        open_times.append(open_ns / 1_000_000)
        close_times.append(close_ns / 1_000_000)
        total_times.append(total_ns / 1_000_000)

    open_mean = statistics.mean(open_times)
    close_mean = statistics.mean(close_times)
    total_mean = statistics.mean(total_times)

    return BatchStats(
        batch_size=len(paths),
        iterations=iterations,
        open_ms_mean=open_mean,
        open_ms_p50=percentile(open_times, 0.50),
        open_ms_p95=percentile(open_times, 0.95),
        close_ms_mean=close_mean,
        close_ms_p50=percentile(close_times, 0.50),
        close_ms_p95=percentile(close_times, 0.95),
        close_us_per_fd_mean=(close_mean * 1000.0) / len(paths),
        total_ms_mean=total_mean,
        close_share_pct=(close_mean / total_mean * 100.0) if total_mean > 0 else 0.0,
    )


def print_stats(stats: BatchStats) -> None:
    print(
        "batch_size={batch_size:6d} iterations={iterations:6d} "
        "open_mean_ms={open_ms_mean:8.3f} open_p50_ms={open_ms_p50:8.3f} open_p95_ms={open_ms_p95:8.3f} "
        "close_mean_ms={close_ms_mean:8.3f} close_p50_ms={close_ms_p50:8.3f} close_p95_ms={close_ms_p95:8.3f} "
        "close_mean_us_per_fd={close_us_per_fd_mean:8.3f} total_mean_ms={total_ms_mean:8.3f} "
        "close_share_pct={close_share_pct:6.2f}".format(**stats.__dict__)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Quantify the overhead of batched os.close() on a long list of already-opened files. "
            "This isolates the candidate-1 delta versus main for the NIXL FILE path."
        )
    )
    parser.add_argument(
        "--batch-sizes",
        default="64,128,256,512,1024,2048,4096",
        help="Comma-separated batch sizes to test. Default: %(default)s",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=200,
        help="Iterations per batch size. Default: %(default)s",
    )
    parser.add_argument(
        "--base-dir",
        default=None,
        help="Optional directory to place benchmark files. Defaults to a temporary directory.",
    )
    parser.add_argument(
        "--keep-files",
        action="store_true",
        help="Keep generated files/directories after the benchmark exits.",
    )
    args = parser.parse_args()

    batch_sizes = parse_batch_sizes(args.batch_sizes)
    max_batch_size = max(batch_sizes)

    temp_dir = args.base_dir or tempfile.mkdtemp(prefix="nixl-fd-close-bench-")
    created_temp_dir = args.base_dir is None

    try:
        print(f"benchmark_dir={temp_dir}")
        print("Creating files...")
        paths = create_files(temp_dir, max_batch_size)
        print("Running benchmark...")
        print()
        for batch_size in batch_sizes:
            stats = run_benchmark(paths[:batch_size], args.iterations)
            print_stats(stats)
    finally:
        if created_temp_dir and not args.keep_files:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
