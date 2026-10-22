"""Source- and runtime-version isolation for simulator machine-code caches."""

import os
import platform
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from typing import NoReturn

from numba import config

from zcu_tools.utils.cache import user_cache_root


def _raise_source_scan_error(error: OSError) -> NoReturn:
    raise error


def configure_numba_cache() -> None:
    """Configure process-wide Numba storage before simulator dispatchers exist.

    Hash every simulator Python source and the Python, Numba, NumPy and SciPy
    versions. Source read failures propagate rather than using unprotected code.
    An existing NUMBA_CACHE_DIR is a parent, not the final cache directory.
    Otherwise use the shared user cache root. Numba retains its ABI/CPU keys.
    Force Numba's built-in user-directory locator; unavailable storage fails
    rather than falling back to unprotected caches. This replaces caller locator
    settings process-wide. The first Numba tool import loads this module once;
    changing sources or cache roots requires a new process. Later-created
    dispatchers in other packages share this root and locator policy.
    """
    source_root = Path(__file__).parent
    digest = sha256()
    for name in (
        platform.python_version(),
        *(version(p) for p in ("numba", "numpy", "scipy")),
    ):
        digest.update(name.encode() + b"\0")
    sources: list[Path] = []
    for directory, _, filenames in os.walk(
        source_root, onerror=_raise_source_scan_error
    ):
        sources.extend(
            Path(directory) / name for name in filenames if name.endswith(".py")
        )
    for path in sorted(sources):
        digest.update(path.relative_to(source_root).as_posix().encode() + b"\0")
        digest.update(sha256(path.read_bytes()).digest())

    namespace = digest.hexdigest()
    override = os.environ.get("NUMBA_CACHE_DIR")
    directory = (
        Path(override) / f"zcu-simulate-{namespace}"
        if override is not None
        else user_cache_root() / "numba" / namespace
    )
    os.environ["NUMBA_CACHE_DIR"] = str(directory)
    os.environ["NUMBA_CACHE_LOCATOR_CLASSES"] = "UserProvidedCacheLocator"
    # Numba may already be imported by an application or another package.
    config.reload_config()


# Python serializes this module's import across the ADC and evolution entries.
# Only a successful import is retained; a failed initialization can be retried.
configure_numba_cache()
