"""Shared user cache location for desktop services and clients."""

import logging
import os
import tempfile
from pathlib import Path

import platformdirs

logger = logging.getLogger(__name__)


def user_cache_root() -> Path:
    """Return the current user cache root without creating directories.

    ZCU_TOOLS_CACHE_DIR overrides the platform default and must be a non-empty
    absolute path; invalid overrides raise ValueError without falling back.
    Without an override, use platformdirs.user_cache_dir("zcu_tools", "zcu_tools").
    If platform lookup fails, use <tempdir>/.zcu_tools. Read the environment on
    each call so GUI persistence and session discovery share the current root.
    """
    override = os.environ.get("ZCU_TOOLS_CACHE_DIR")
    if override is not None:
        root = Path(override)
        if not override or not root.is_absolute():
            raise ValueError("ZCU_TOOLS_CACHE_DIR must be a non-empty absolute path")
        return root

    try:
        return Path(platformdirs.user_cache_dir("zcu_tools", "zcu_tools"))
    except Exception:
        logger.exception("User cache lookup failed; using temporary storage")
        return Path(tempfile.gettempdir()) / ".zcu_tools"
