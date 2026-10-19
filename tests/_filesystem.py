"""Filesystem capabilities shared by tests that exercise real local paths."""

import errno
import sys
from pathlib import Path

import pytest

# CPython reports opening a directory as EACCES on Windows and EISDIR on POSIX.
DIRECTORY_OPEN_ERROR = PermissionError if sys.platform == "win32" else IsADirectoryError


def symlink_or_skip(
    link: Path, target: str | Path, *, target_is_directory: bool = False
) -> None:
    """Create a symlink, skipping only when the host denies that capability."""
    try:
        link.symlink_to(target, target_is_directory=target_is_directory)
    except NotImplementedError:
        pytest.skip("Symlink creation is not implemented on this host")
    except OSError as error:
        if error.errno in (errno.EACCES, errno.EPERM, errno.ENOSYS, errno.ENOTSUP) or (
            getattr(error, "winerror", None) == 1314
        ):
            pytest.skip(f"Symlink creation is unavailable: {error}")
        raise
