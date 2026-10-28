"""Import-safe cache bootstrap shared by root conftest invocations."""

import atexit
from pathlib import Path
from tempfile import TemporaryDirectory


def persistent_numba_cache_root() -> str:
    """Return the worktree-local parent for reusable Numba machine code.

    All workers and later pytest invocations use this ignored directory. It is
    independent of process-owned user caches and pytest basetemp cleanup.
    The simulator adds its source/version namespace before compiling. Remove
    this directory manually to request a cold compilation run.
    """
    return str(Path(__file__).resolve().parents[1] / ".pytest-tmp-numba-cache")


def temporary_cache_root() -> str:
    """Create a worktree-local cache directory retained until process exit.

    Return its absolute path for ZCU_TOOLS_CACHE_DIR. Register cleanup with
    atexit, which keeps the TemporaryDirectory alive through nested pytest.main
    teardown. Each call creates an independent directory, including in workers.
    """
    cache = TemporaryDirectory(
        prefix=".pytest-tmp-cache-", dir=Path(__file__).resolve().parents[1]
    )
    atexit.register(cache.cleanup)
    return cache.name
