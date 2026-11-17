"""Best-effort publication of caller-owned Engine captures (ADR-0078)."""

import os
import warnings
from contextlib import suppress
from pathlib import Path
from uuid import uuid4

from .._numba_cache import user_cache_root
from ._capture import CaptureBuffer
from .frozen import write_frozen_manifest


def capture_directory() -> Path:
    """Return the absolute shared cache directory for v1 debug captures.

    Resolve the same cache root as Numba, then append simulator-debug/v1.
    No directory is created. Invalid ZCU_TOOLS_CACHE_DIR (empty or relative)
    raises ValueError; see user_cache_root for platform fallback behavior.
    """
    return user_cache_root() / "simulator-debug" / "v1"


def write_capture(buffer: CaptureBuffer) -> Path | None:
    """Publish one frozen manifest from buffer and return its absolute path.

    buffer is a caller-owned CaptureBuffer. Call only after all evolves using
    it have returned or raised. Read records once, including an empty snapshot.
    Each call chooses a new capture-<uuid4 hex>.json.gz name; rewriting a buffer
    retains its earlier records in a different file. Write a same-directory
    temporary file with write_frozen_manifest and publish with os.replace.

    Invalid cache configuration and errors obtaining records propagate before
    publication. Publication OSError and codec ValueError/TypeError emit at most
    one RuntimeWarning and return None, even under warnings-as-errors. Nonfinite
    or unsupported JSON values cannot be recorded. Other exceptions propagate.
    On every failed publication, attempt to remove the temporary file.
    No size limit, truncation or eviction is applied; users delete files.
    """
    directory = capture_directory()
    records = buffer.records
    name = f"capture-{uuid4().hex}.json.gz"
    destination = directory / name
    temporary = directory / f".{name}.tmp"
    try:
        directory.mkdir(parents=True, exist_ok=True)
        write_frozen_manifest(temporary, records)
        os.replace(temporary, destination)
        return destination
    except (OSError, ValueError, TypeError) as error:
        with suppress(Warning):
            warnings.warn(
                f"capture could not be written: {error}", RuntimeWarning, stacklevel=2
            )
        return None
    finally:
        with suppress(OSError):
            temporary.unlink(missing_ok=True)
