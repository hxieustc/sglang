# NIXL Integration for HiCache

This directory contains the **NIXL (NVIDIA Inference Xfer Library)** integration for **HiCache**, enabling high-performance storage across multiple backends.

NIXL provides a unified API for accessing various storage plugins, including but not limited to:

- **Deepseek's 3FS APIs** for high-throughput file operations
- **GPU Direct Storage (GDS)** for direct data movement between storage and GPU memory, bypassing CPU memory copies
- **Amazon S3-compatible object storage** for key-value access patterns

Additional backend integrations are planned for future releases.

## NIXL Resources

- **Project Repository**: [NIXL on GitHub](https://github.com/ai-dynamo/nixl)
- **Documentation**: [NIXL Documentation](https://github.com/ai-dynamo/nixl/tree/main/docs)

## Overview

The NIXL integration consists of two main files:

- **`hicache_nixl.py`** - Main HiCache storage connector using NIXL
- **`nixl_utils.py`** - Utility classes for backend selection, registration, and file management

## Components

### HiCacheNixl
The main storage connector that provides:
- Single and batch tensor set/get operations
- Automatic backend selection (3FS > POSIX > GDS_MT > GDS > OBJ)
- High-performance file-based (or) object based storage access using NIXL
- A dedicated query agent for storage hit checks
- Immutable versioned FILE writes with committed metadata
- Owner-only publishing across TP ranks for FILE mode
- Lazy GC of superseded FILE versions

### NixlUtils
Consolidated utility classes:
- **NixlBackendSelection** - Handles backend selection and creation
- **NixlBackendConfig** - Handles backend configuration
- **NixlRegistration** - Manages memory registration for tensors, files and objects
- **NixlFileManager** - Handles file system operations and NIXL tuple creation
- **NixlPageCommitIndex** - Tracks committed FILE versions

## Design Notes

### Why FILE Mode Uses Versioned Commits

HiCache drives storage work from multiple controller threads:

- prefetch hit queries call `batch_exists()`
- prefetch I/O calls `batch_get_v1()`
- backup/writeback calls `batch_set_v1()`

For FILE backends, correctness depends on a committed-version protocol instead of raw file visibility:

- writes create immutable versioned files
- a metadata record publishes the committed version that readers may consume
- `batch_exists()` and FILE reads use committed metadata, not bare file presence
- only the deterministic owner TP rank for a logical page may publish a new committed version
- superseded versions are queued for lazy deletion after a grace period

This prevents readers from observing partially written files and avoids multi-process last-writer-wins races on the committed page view.

### FILE-Mode Execution Model

FILE mode is intentionally conservative with the current NIXL POSIX backend:

- one dedicated **query agent** handles `batch_exists()`
- FILE transfers run through a single serialized execution lane inside the process
- runtime knobs like `numjobs` are still accepted, but FILE-mode transfer execution remains serialized for stability

Runtime-only knobs are consumed by HiCache and are not forwarded to NIXL backend initialization.

### File Layout for FILE Backends

For FILE-based plugins such as POSIX / GDS / GDS_MT / 3FS, NIXL stores KV cache data as files under the configured storage directory.

NIXL now supports two layout strategies:

- `hashed2` (default): `base_dir/<first2>/<next2>/<full_key>`
- `flat`: `base_dir/<full_key>`

The default `hashed2` layout spreads files across subdirectories using the first few characters of the hash-like cache key. This reduces the chance of creating a single very hot directory with a very large number of entries, which can otherwise stress directory metadata lookup and inode access on some filesystems.

Use `flat` only if you explicitly want the original single-directory layout for compatibility or controlled experiments.

## Using NIXL as the HiCache Storage Backend

### 1. How Backend Plugin Selection Works

The NIXL backend can support **multiple storage plugins** (e.g., POSIX, GDS, GDS_MT, 3FS, object store, etc).

* Each plugin has its own configuration section in the TOML file.
* A plugin is considered **usable** if:

  * Its required library is available on the system (POSIX, GDS, GDS_MT are natively supported by NIXL).
  * Its configuration is valid.
  * It is marked as `active = true` in the configuration file (if applicable).
* Some plugins (e.g., 3FS, GDS) require additional system libraries or hardware support.
* NIXL selects the backend based on **internal priority and availability**, if neither a config file nor an in-command-line config string is provided.

If a plugin is configured but its dependencies are missing, it will be skipped.

Runtime-only knobs such as transfer pool sizing are configured alongside backend settings, but are consumed by HiCache rather than passed through to NIXL backend creation.


### 2. Setting the Storage Directory (Optional)

For POSIX / GDS / GDS_MT file-based backends, the default storage location is `/tmp/hicache_storage`. However, you can customize where cached data is stored:

```bash
export SGLANG_HICACHE_NIXL_BACKEND_STORAGE_DIR=/path/to/storage/dir
```

### 3. How to Provide Configuration for Backends

There are three ways to specify configurations for the backends: default config, file based config, and command-line (JSON string based) config.

#### 1. Using Default Configuration

To enable HiCache with the NIXL backend, start the SGLang server with:

```bash
python3 -m sglang.launch_server \
  --model-path <model> \
  --host <ip> \
  --port <port> \
  --page-size 64 \
  --enable-hierarchical-cache \
  --hicache-ratio 2 \
  --hicache-size 64 \
  --hicache-write-policy write_through \
  --hicache-storage-backend nixl
```

By default, NIXL will use its internal backend selection logic to choose an available storage plugin (and use default configs for the selected storage plugin).


#### 2. Using a Configuration File (Recommended)

For non-trivial setups with complex configurations, it is recommended to use a **TOML configuration file** to define which backend plugin to use and its configurations, via `--hicache-storage-backend-extra-config`:

Below is an example command (note: detailed configs are defined in the config file):

```bash
python3 -m sglang.launch_server \
  --model-path <model> \
  --host <ip> \
  --port <port> \
  --page-size 64 \
  --enable-hierarchical-cache \
  --hicache-ratio 2 \
  --hicache-size 64 \
  --hicache-write-policy write_through \
  --hicache-storage-backend nixl \
  --hicache-storage-backend-extra-config "@config.nixl.toml"
```

> **Important**
>
> * The `@` prefix tells SGLang to load the configuration from a file.
> * The file can be in **TOML format** (other formats, JSON / YAML, are also supported).
> * This is the preferred way to configure NIXL storage backends.

The structure of the config file is described in further details in [Configuration File Spec](#Configuration-File-Specification).

Example TOML snippet showing both backend config and runtime transfer scaling:

```toml
[runtime]
numjobs = 4
file_layout = "hashed2"
file_gc_grace_seconds = 300

[plugin.posix]
active = true
use_uring = true
```


#### 3. Using Command-line JSON String

For debugging or quick testing, you may pass a **JSON-style string** directly via `--hicache-storage-backend-extra-config`.

This requires explicitly specifying the plugin type via an environment variable, and this method can be applicable to **only a few** plugins (e.g., POSIX, GDS, GDS_MT)

The below example shows how to use command-line string to use the POSIX plugin where URING is enabled for async POSIX storage.

```bash
export SGLANG_HICACHE_NIXL_BACKEND_PLUGIN=POSIX

python3 -m sglang.launch_server \
  --model-path <model> \
  --host <ip> \
  --port <port> \
  --page-size 64 \
  --enable-hierarchical-cache \
  --hicache-ratio 2 \
  --hicache-size 64 \
  --hicache-write-policy write_through \
  --hicache-storage-backend nixl \
  --hicache-storage-backend-extra-config "{'use_uring': 'true', 'numjobs': 4, 'file_layout': 'hashed2', 'file_gc_grace_seconds': 300}"
```

⚠️ **Note**:
This method is convenient for testing / experimenting. For production or multi-plugin setups, it is always recommended to use the config file based approach.


## Running Unit Tests

### Prerequisites
- NIXL library installed and available (latest main required for supporting object query)
- PyTorch installed
- Python 3.8+

### Unit tests from current directory
From the current directory run:

#### Run all NIXL tests:
```bash
FLASHINFER_WORKSPACE_BASE=/tmp PYTHONPATH=. python -m unittest test_hicache_nixl_storage.py
```

#### Run with verbose output:
```bash
FLASHINFER_WORKSPACE_BASE=/tmp PYTHONPATH=. python -m unittest -v test_hicache_nixl_storage.py
```

#### Run a specific test:
```bash
FLASHINFER_WORKSPACE_BASE=/tmp PYTHONPATH=. python -m unittest -v test_hicache_nixl_storage.TestNixlUnified.test_single_set_get
```

## Test Coverage

Tests for this integration, a test suite can be found at `test_hicache_nixl_storage.py` which covers:

### HiCache Integration Tests (4 tests)
- Single tensor set/get operations
- Batch tensor set/get operations
- Mixed single and batch operations
- Data integrity for various tensor types

### File Management Tests (5 tests)
- Basic file operations
- NIXL tuple creation
- Error handling in file operations

### Registration Tests (2 tests)
- Tensor registration with memory type detection
- File registration using NIXL tuples

### Runtime / Concurrency Tests
- Runtime-only config filtering for backend init params
- Default `numjobs=4` configuration with serialized FILE execution
- Runtime override of `numjobs`
- Concurrent FILE-mode `batch_get_v1()` / `batch_set_v1()` stress coverage across multiple Python threads
- Multi-process owner-rank publishing behavior

### Versioned FILE Commit / GC Tests
- Atomic committed metadata publish/load
- Previous committed version enqueued for lazy GC
- Startup GC scan removing stale stable files
- Stats coverage for skipped non-owner writes and committed publishes

### File Layout Tests (5 tests)
- Default `hashed2` path generation
- Explicit `flat` layout path generation
- Invalid layout rejection
- HiCache default file layout propagation
- HiCache runtime file layout override propagation

## Expected Output

When tests run successfully, you should see:
- NIXL agent initialization messages
- Backend selection messages (e.g., "Backend POSIX was instantiated")
- Test results with "ok" for passed tests
- Summary showing "Ran X tests in Y seconds" and "OK"

## Troubleshooting

### Import Errors
If you encounter `ModuleNotFoundError`, ensure:
- You're running from the correct directory
- `PYTHONPATH` is set correctly
- NIXL library is properly installed

### NIXL Errors
If NIXL operations fail:
- Check that NIXL is properly installed
- Verify that required plugins are available
- Ensure file permissions are correct for test directories

### Throughput Tuning
If transfer throughput is lower than expected:
- Check the effective runtime config in `extra_config`
- For FILE backends, prioritize correctness and metadata layout tuning first; transfer execution is serialized today
- For non-FILE backends, increase `numjobs` only when the backend and filesystem can sustain parallel transfer work

### File Layout Tuning
If FILE backend metadata access becomes a bottleneck:
- keep the default `hashed2` layout for better directory fanout
- consider `flat` only for compatibility testing or if your filesystem performs better with a single directory
- when comparing layouts, measure metadata-heavy workloads rather than only large sequential transfer bandwidth

### FILE-Mode Operational Signals
The backend now logs and tracks:
- skipped non-owner writes
- committed version publishes
- pending GC queue length
- stale-version deletions

## File Structure

```
python/sglang/srt/mem_cache/nixl/
├── hicache_nixl.py          # Main HiCache storage connector
├── nixl_utils.py            # All NIXL utility classes
├── README.md                # This file
└── test_hicache_nixl_storage.py
```

## Dependencies

- **NIXL**: NVIDIA Inference Xfer Library (version 0.4 or later)
  - Required plugins: POSIX (minimum), 3FS/GDS (optional for better performance)
  - See [NIXL Installation Guide](https://github.com/ai-dynamo/nixl/blob/main/README.md#installation)
- **PyTorch**: For tensor operations (version 1.8 or later)
- **Python 3.8+**: For type hints and modern features

## Supported Features

### Memory Types
- **Tensor side**: multi-dimensional tensors of all numeric types (int32, int64, float32, float64) are supported.
  - Tensors can be on CPU or GPU (as long as a GPU capable backend such as GDS_MT is available).
  - Currently each tensor is mapped to a file or key, but it can be extended to support multiple keys per file or key.

- **Storage side**: file and object are supported through their relevant backends (e.g., 3FS or OBJ).

### Backend Priority

The NIXL backend selection follows this priority order:
1. **3FS** - Highest performance (if available)
    - Best for high-throughput file operations using Deepseek 3FS APIs
2. **POSIX** - Standard file I/O (fallback)
    - Universal compatibility
    - Good for development and testing - Leverages both libaio/liburing
3. **GDS_MT** - Multi-threaded GDS (if available)
    - Optimized for concurrent operations
    - Supports GPU Direct storage with multiple light weight threads
4. **GDS** - GPU Direct Storage (if available)
    - Direct GPU-storage data path
    - Best for filesystems benefiting from batch operations and smaller IOs.
5. **OBJ** - Amazon S3 based Object Storage
    - Key-value based storage
The system automatically selects the best available backend, with POSIX as the default fallback.



## Configuration File Specification

This section defines the structure, supported sections, configuration keys, data types, defaults, and semantics for the NIXL HiCache backend configuration file (`config.nixl.toml`).

The configuration file is written in **TOML** and consists of multiple **plugin-specific sections** under the `plugin.*` namespace. Each section configures one storage backend plugin. Only one plugin should be enabled via setting `active = true` in the corresponding plugin-specific section.

An example of the configuration is provided in [`nixl.config.toml.sample`](./nixl.config.toml.sample).

### 1. General Structure

```toml
[plugin.<backend_name>]
<key> = <value>
```

* `<backend_name>` identifies the storage backend plugin.
* Each plugin is configured independently.
* Plugins are selected at runtime based on:

  * Availability of required libraries/hardware
  * Plugin configuration validity
  * Internal backend priority rules
* Unless otherwise stated, all configuration keys are **optional** and have sensible defaults.


### 2. POSIX File System Backend (`plugin.posix`)

#### Section

```toml
[plugin.posix]
```

#### Description

Configures the POSIX file-system-based backend.
This backend supports multiple asynchronous I/O mechanisms and automatically selects the most performant option supported by the system.

**Backend priority (highest to lowest):**

1. Linux AIO
2. `io_uring`
3. POSIX AIO


#### Configuration Keys

| Key             | Type    | Default   | Description                                                                                              |
| --------------- | ------- | --------- | -------------------------------------------------------------------------------------------------------- |
| `use_uring`     | string  | `"false"` | Enables Linux `io_uring` for asynchronous I/O when set to `"true"`. Recommended on modern Linux kernels. |
| `use_posix_aio` | string  | `"false"` | Enables POSIX AIO as an alternative async I/O mechanism.                                                 |
| `use_aio`       | string  | `"false"` | Enables generic Linux AIO.                                                                               |
| `active`        | boolean |    N/A    | Controls whether this plugin is eligible for backend selection.                                          |

**Notes**

* Boolean-like options use **string values** (`"true"` / `"false"`) for compatibility.
* **Only one backend** (i.e., only one of `use_uring`, `use_aio`, `use_posix_aio`) should be included in the config.


### 3. NVIDIA GPUDirect Storage Backend (`plugin.gds`)

#### Section

```toml
[plugin.gds]
```

#### Description

Configures NVIDIA GPUDirect Storage (GDS) backend.
This backend enables direct data transfers between storage and GPU memory.

**Requirements**

* NVIDIA GPU with GDS support
* Compatible NVIDIA driver and CUDA runtime
* Supported filesystem


#### Configuration Keys

| Key                | Type    | Default            | Description                                            |
| ------------------ | ------- | ------------------ | ------------------------------------------------------ |
| `batch_pool_size`  | integer | `128`              | Number of I/O requests maintained in the request pool. |
| `batch_limit`      | integer | `128`              | Maximum number of requests issued in a single batch.   |
| `max_request_size` | integer | `16777216` (16 MB) | Maximum size (in bytes) of a single I/O request.       |
| `active`           | boolean |         N/A        | Controls whether this plugin is eligible for backend selection.|


### 4. Multi-Threaded GDS Backend (`plugin.gds_mt`)

#### Section

```toml
[plugin.gds_mt]
```

#### Description

Configures the multi-threaded variant of the NVIDIA GDS backend, allowing parallel request processing using multiple CPU threads.



#### Configuration Keys

| Key            | Type    | Default | Description                                           |
| -------------- | ------- | ------- | ----------------------------------------------------- |
| `thread_count` | integer | `4`     | Number of worker threads used to submit GDS requests. |
| `active`       | boolean |    N/A  | Controls whether this plugin is eligible for backend selection. |


### 5. 3FS Backend (`plugin.3fs`)

#### Section

```toml
[plugin.3fs]
```

#### Description

Configures the 3FS (third-party filesystem) backend.

**Requirements**

* 3FS client library installed
* Filesystem mounted and accessible on the host


#### Configuration Keys

| Key           | Type    | Default  | Description                        |
| ------------- | ------- | -------- | ---------------------------------- |
| `mount_point` | string  | *none*   | Mount point of the 3FS filesystem. |
| `mem_config`  | string  | `"dram"` | Memory configuration mode.         |
| `iopool_size` | integer | `64`     | Size of the I/O pool.              |
| `active`      | boolean |   N/A    | Controls whether this plugin is eligible for backend selection.                                          |

##### `mem_config` Valid Values

| Value     | Description                                         |
| --------- | --------------------------------------------------- |
| `dram`    | Use DRAM for buffering                              |
| `dram_zc` | Use DRAM with zero-copy support                     |
| `auto`    | Automatically select based on platform capabilities |

##### `iopool_size` Constraints

* Valid range: **[2⁶, 2²⁰]**
* Values outside this range may cause initialization failure.


### 6. Object Storage Backend (`plugin.obj`)

#### Section

```toml
[plugin.obj]
```

#### Description

Configures an object storage backend compatible with S3 APIs (e.g., AWS S3, MinIO, Ceph).


#### Configuration Keys

| Key                      | Type    | Default      | Description                                    |
| ------------------------ | ------- | ------------ | ---------------------------------------------- |
| `num_threads`            | integer | `4`          | Number of client worker threads.               |
| `endpoint_override`      | string  | `""`         | Custom endpoint URL (for non-AWS S3 services). |
| `scheme`                 | string  | `"http"`     | Connection scheme (`http` or `https`).         |
| `region`                 | string  | `""`         | Cloud region (if applicable).                  |
| `req_checksum`           | string  | `"required"` | Request checksum behavior.                     |
| `ca_bundle`              | string  | `""`         | Path to a custom CA bundle.                    |
| `access_key`             | string  | `""`         | Access key credential.                         |
| `secrete_key`            | string  | `""`         | Secret key credential.                         |
| `session_token`          | string  | `""`         | Session token (optional).                      |
| `use_virtual_addressing` | string  | `"true"`     | Enables virtual-hosted-style addressing.       |
| `bucket`                 | string  | `""`         | Default bucket name.                           |
| `active`                 | boolean |      N/A     | Controls whether this plugin is eligible for backend selection.                                          |

##### `req_checksum` Valid Values

| Value       | Description                                    |
| ----------- | ---------------------------------------------- |
| `required`  | Always include a checksum                      |
| `supported` | Include checksum when supported by the backend |


### 7. Notes and Best Practices

* All plugin sections are optional.
* Multiple plugins may be configured in a single file. However, it is recommended that **only one plugin** is configured `active = true`.
* Plugins whose dependencies are unavailable will be skipped.
* Use a TOML configuration file instead of inline JSON for:

  * Multi-plugin setups
  * Production deployments
  * Clear validation and maintainability


## Note

This is v0 of the NIXL connector. Future versions will focus on further performance optimizations such as memory pre-registration (pre-allocating and registering memory buffers to reduce registration overhead during transfers) and block merging (combining related blocks as offsets within the same file to reduce file operations and improve throughput). These optimizations require changes at a higher layer, as the current HiCache API doesn't expose information like block relationships or hash patterns that would enable these optimizations.
