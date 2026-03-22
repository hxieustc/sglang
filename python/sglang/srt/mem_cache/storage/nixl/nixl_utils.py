import logging
import os
import resource
import threading
from collections import OrderedDict
from typing import Any, List, Optional, Tuple, Union

import torch

logger = logging.getLogger(__name__)


class NixlBackendConfig:
    """Handles NIXL backend configurations"""

    def __init__(self, config: Optional[dict[str, str]] = None):
        """Initialize backend configuration.
        Args:
            config: configurations in a dictionary. This config comes from --hicache-storage-backend-extra-config

            config can be in two forms:
            1. fully qualified form (for all plugins, some of them are enabled, others not):
                {'plugin': { 'posix': {...}, 'gds': {...}, ...}}
            2. flat form (for a specific selected plugin), assuming all params apply to a selected plugin
                {'param1': 'value1', 'param2': 'value2', ...}
        """
        self.config = config or {}

    def get_specified_plugin(self) -> str:
        """decide which plugin to use: either config or SGLANG_HICACHE_NIXL_BACKEND_PLUGIN specifies the plugin, if not, use "auto" """

        if "plugin" in self.config:
            # fully qualified form: {'plugin': { 'posix': {...}, 'gds': {...}, ...}}
            # choose the FIRST active plugin
            for key, item in self.config["plugin"].items():
                if item.get("active", False) in [True, "true", "True"]:
                    plugin = key.upper()
                    break
        else:
            # config is empty, or in flat form {'param1': 'value1', 'param2': 'value2', ...}
            plugin = os.getenv("SGLANG_HICACHE_NIXL_BACKEND_PLUGIN", "auto")

        return plugin

    def get_backend_initparams(self, backend_name) -> dict:
        """Get initialization parameters from config of NIXL backend for backend creation.
        Args:
            backend_name: a specific backend's name (already converted "auto" into a specific backend name)

        """

        initparams = {}

        # config can be in two forms:
        if "plugin" in self.config:
            # fully qualified form: {'plugin': { 'posix': {...}, 'gds': {...}, ...}}
            if backend_name.lower() in self.config["plugin"]:
                config_data = self.config["plugin"][backend_name.lower()]
            else:
                logger.debug(
                    f"No specific config found for plugin {backend_name} in extra_config. Use default init params."
                )
                config_data = {}
        else:
            # flat form {'param1': 'value1', 'param2': 'value2', ...}
            config_data = self.config

        for key, value in config_data.items():
            initparams[key] = str(value)

        return initparams


class NixlBackendSelection:
    """Handles NIXL backend selection and creation."""

    # Priority order for File-based plugins in case of auto selection
    FILE_PLUGINS = ["3FS", "POSIX", "GDS_MT", "GDS"]
    # Priority order for File-based plugins in case of auto selection (add more as needed)
    OBJ_PLUGINS = ["OBJ"]  # Based on Amazon S3 SDK

    def __init__(
        self, plugin: str = "auto", nixlconfig: Optional[NixlBackendConfig] = None
    ):
        """Initialize backend selection.
        Args:
            plugin: Plugin to use (default "auto" selects best available).
                   Can be a file plugin (3FS, POSIX, GDS, GDS_MT) or
                   an object plugin (OBJ).
        """
        self.plugin = plugin
        self.backend_name = None
        self.mem_type = None
        self.nixlconfig = nixlconfig

    def set_bucket(self, bucket_name: str) -> None:
        """Set AWS bucket name in environment variable."""
        os.environ["AWS_DEFAULT_BUCKET"] = bucket_name
        logger.debug(f"Set AWS bucket name to: {bucket_name}")

    def create_backend(self, agent) -> bool:
        """Create the appropriate NIXL backend based on configuration."""
        try:
            plugin_list = agent.get_plugin_list()
            logger.debug(f"Available NIXL plugins: {plugin_list}")

            # Handle explicit plugin selection or auto priority
            if self.plugin == "auto":
                # Try all file plugins first
                for plugin in self.FILE_PLUGINS:
                    if plugin in plugin_list:
                        self.backend_name = plugin
                        break
                # If no file plugin found, try object plugins
                if not self.backend_name:
                    for plugin in self.OBJ_PLUGINS:
                        if plugin in plugin_list:
                            self.backend_name = plugin
                            break
            else:
                # Use explicitly requested plugin
                self.backend_name = self.plugin

            if self.backend_name not in plugin_list:
                logger.error(
                    f"Backend {self.backend_name} not available in plugins: {plugin_list}"
                )
                return False

            # obtain initparams for the backend from the NIXL config
            initparams = (
                self.nixlconfig.get_backend_initparams(self.backend_name)
                if self.nixlconfig
                else {}
            )

            # Create backend and set memory type
            if self.backend_name in self.OBJ_PLUGINS and "bucket" not in initparams:
                bucket = os.environ.get("AWS_DEFAULT_BUCKET")
                if not bucket:
                    logger.error(
                        "AWS_DEFAULT_BUCKET environment variable must be set for object storage"
                    )
                    return False

                initparams["bucket"] = bucket

            # create backend using initialization parameters
            agent.create_backend(self.backend_name, initparams)

            logger.info(
                f"NixlBackendSelection.create_backend: backend_name {self.backend_name} initparams {initparams} customParams {agent.get_backend_params(self.backend_name)} supported plugins {plugin_list}"
            )

            self.mem_type = "OBJ" if self.backend_name in self.OBJ_PLUGINS else "FILE"
            logger.debug(
                f"Created NIXL backend: {self.backend_name} with memory type: {self.mem_type}"
            )
            return True

        except Exception as e:
            logger.error(
                f"Failed to create NIXL backend: {e}, backend_name {self.backend_name}, supported plugins {plugin_list} initparams {initparams}"
            )
            return False


class NixlRegistration:
    """Handles NIXL memory registration."""

    def __init__(self, agent):
        self.agent = agent

    def create_query_tuples(
        self, key: str, mem_type: str, file_manager=None
    ) -> List[Tuple]:
        """Create NIXL tuples for querying memory.
        Args:
            key: Key to query (file path for FILE or object key for OBJ)
            mem_type: Memory type ("FILE" or "OBJ")
            file_manager: Optional NixlFileManager for FILE memory type
        Returns:
            List of NIXL tuples for querying
        """
        if mem_type == "FILE":
            if file_manager is None:
                logger.error("file_manager required for FILE memory type")
                return []
            return [(0, 0, 0, file_manager.get_file_path(key))]
        else:  # OBJ
            return [(0, 0, 0, key)]

    def _register_memory(
        self,
        items: Union[List[tuple], torch.Tensor, List[torch.Tensor]],
        mem_type: Optional[str] = None,
    ) -> Optional[Any]:
        """Common registration logic for files, objects, and buffers.
        Args:
            items: List of tuples or tensors to register
            mem_type: Memory type ("FILE", "OBJ") or None for tensor or list of tensors
        """
        if isinstance(items, list) and not items:
            return None

        reg_descs = self.agent.get_reg_descs(items, mem_type)
        if reg_descs is None:
            logger.error("Failed to create registration descriptors")
            return None

        try:
            registered_memory = self.agent.register_memory(reg_descs)
            return registered_memory  # Could be None in case of error
        except Exception as e:
            if not mem_type:
                logger.error(f"Failed to register Tensors with NIXL: {e}")
            else:
                logger.error(
                    f"Failed to register memory of type {mem_type} with NIXL: {e}"
                )
            return None


class NixlFileManager:
    """Handles file system operations for NIXL."""

    def __init__(self, base_dir: str, max_open_files: Optional[int] = None):
        """
        Initialize file manager.
        Args:
            base_dir: Base directory for storing tensor files
            max_open_files: Maximum number of cached file descriptors to retain
        """
        self.base_dir = base_dir
        self.max_open_files = (
            max_open_files
            if max_open_files is not None
            else self._get_default_max_open_files()
        )
        self._fd_cache: OrderedDict[str, int] = OrderedDict()
        self._lock = threading.Lock()
        if base_dir == "":
            logger.debug(f"Initialized file manager without a base directory")
        else:
            os.makedirs(base_dir, exist_ok=True)
            logger.debug(f"Initialized file manager with base directory: {base_dir}")

    @staticmethod
    def _get_default_max_open_files() -> int:
        try:
            soft_limit, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
        except (OSError, ValueError):
            return 1_000_000

        if soft_limit in (-1, resource.RLIM_INFINITY):
            return 1_000_000

        return max(1, min(1_000_000, int(soft_limit)))

    def clear(self) -> None:
        """Clear all files in the base directory."""
        if self.base_dir == "":
            logger.warning("Base directory is empty, skipping clear operation")
            return

        try:
            self.close_all_files()
            for root, dirs, files in os.walk(self.base_dir):
                for file in files:
                    os.remove(os.path.join(root, file))
            logger.debug(f"Cleared all files in base directory: {self.base_dir}")
        except Exception as e:
            logger.error(
                f"Failed to clear files in base directory {self.base_dir}: {e}"
            )

    def get_file_path(self, key: str) -> str:
        """Get full file path for a given key."""
        return os.path.join(self.base_dir, key)

    def create_file(self, file_path: str) -> bool:
        """Create a file if it doesn't exist."""
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            if not os.path.exists(file_path):
                with open(file_path, "wb") as f:
                    pass  # Create empty file
            return True
        except Exception as e:
            logger.error(f"Failed to create file {file_path}: {e}")
            return False

    def open_file(self, file_path: str) -> Optional[int]:
        """Open a file (or reuse a cached base descriptor) and return its file descriptor."""
        with self._lock:
            cached_fd = self._fd_cache.get(file_path)
            if cached_fd is not None:
                return cached_fd

        try:
            fd = os.open(file_path, os.O_RDWR)
        except Exception as e:
            logger.error(f"Failed to open file {file_path}: {e}")
            return None

        with self._lock:
            cached_fd = self._fd_cache.get(file_path)
            if cached_fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
                return cached_fd

            self._fd_cache[file_path] = fd
            self._evict_idle_fds_locked()
            return fd

    def close_file(self, fd: int) -> bool:
        """Close a file descriptor."""
        try:
            os.close(fd)
            return True
        except Exception as e:
            logger.error(f"Failed to close file descriptor {fd}: {e}")
            return False

    def _acquire_file(self, file_path: str) -> Optional[int]:
        with self._lock:
            base_fd = self._fd_cache.get(file_path)
            if base_fd is not None:
                try:
                    return os.dup(base_fd)
                except OSError as e:
                    logger.error(f"Failed to dup file descriptor for {file_path}: {e}")
                    return None

        try:
            fd = os.open(file_path, os.O_RDWR)
        except Exception as e:
            logger.error(f"Failed to open file {file_path}: {e}")
            return None

        with self._lock:
            cached_fd = self._fd_cache.get(file_path)
            if cached_fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
                try:
                    return os.dup(cached_fd)
                except OSError as e:
                    logger.error(f"Failed to dup file descriptor for {file_path}: {e}")
                    return None

            self._fd_cache[file_path] = fd
            self._evict_idle_fds_locked()
            try:
                return os.dup(fd)
            except OSError as e:
                logger.error(f"Failed to dup file descriptor for {file_path}: {e}")
                return None

    def _evict_idle_fds_locked(self) -> None:
        while len(self._fd_cache) > self.max_open_files:
            evict_path, evict_fd = self._fd_cache.popitem(last=False)
            self.close_file(evict_fd)

    def release_nixl_tuples(self, tuples: List[Tuple[int, int, int, str]]) -> None:
        """Close per-transfer file descriptors stored in NIXL FILE tuples."""
        for _, _, fd, _path in tuples:
            if fd is not None and fd >= 0:
                self.close_file(fd)

    def close_all_files(self) -> None:
        """Close all cached file descriptors."""
        with self._lock:
            cached_items = list(self._fd_cache.items())
            self._fd_cache.clear()

        for _path, fd in cached_items:
            self.close_file(fd)

    def files_to_nixl_tuples(
        self, file_paths: List[str]
    ) -> List[Tuple[int, int, int, str]]:
        """Create NIXL tuples (offset, length, fd, file_path) for given files."""
        tuples = []
        for path in file_paths:
            if (fd := self._acquire_file(path)) is None:
                # Clean up on failure
                self.release_nixl_tuples(tuples)
                return []
            tuples.append((0, 0, fd, path))
        return tuples
