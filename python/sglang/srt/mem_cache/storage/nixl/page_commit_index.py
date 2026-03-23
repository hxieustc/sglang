import json
import logging
import os
import uuid
from typing import Optional

logger = logging.getLogger(__name__)


class NixlPageCommitIndex:
    """Tracks committed immutable FILE versions for logical NIXL pages."""

    def __init__(self, file_manager):
        self.file_manager = file_manager

    def make_version_id(self) -> str:
        return uuid.uuid4().hex

    def get_metadata_path(self, page_key: str) -> str:
        return self.file_manager.get_metadata_path(page_key)

    def load_entry(self, page_key: str) -> Optional[dict]:
        meta_path = self.get_metadata_path(page_key)
        if not os.path.exists(meta_path):
            return None

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(
                f"Failed to load NIXL committed-page metadata for {page_key}: {e}"
            )
            return None

    def load_committed_version(self, page_key: str) -> Optional[str]:
        entry = self.load_entry(page_key)
        if not entry:
            return None
        return entry.get("version")

    def publish_committed_version(
        self, page_key: str, version: str, owner: Optional[int] = None
    ) -> bool:
        meta_path = self.get_metadata_path(page_key)
        temp_path = f"{meta_path}.{uuid.uuid4().hex}.tmp"
        payload = {"version": version}
        if owner is not None:
            payload["owner"] = owner

        try:
            os.makedirs(os.path.dirname(meta_path), exist_ok=True)
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, sort_keys=True)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_path, meta_path)
            return True
        except Exception as e:
            logger.error(
                f"Failed to publish NIXL committed-page metadata for {page_key}: {e}"
            )
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception:
                pass
            return False
