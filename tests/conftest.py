import os
import sys

import pytest
from pytest_cache_guard import GuiCacheGuard

# BLAS thread pinning for xdist workers — must happen BEFORE numpy is imported
# so that OpenBLAS / MKL / OpenMP size their thread pools from the environment
# at library load time (setting them post-load has no effect on OpenBLAS).
#
# Without pinning, 8 xdist workers each spawn 8 BLAS threads → 64 threads
# competing for 8 cores.  Measured penalty: -n auto without pinning = 70.5 s;
# with pinning = 15.1 s.  Serial runs (no PYTEST_XDIST_WORKER) keep the
# default multithreaded BLAS so CPU-bound simulate/* tests run at full speed.
if os.environ.get("PYTEST_XDIST_WORKER"):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

# Configure platformdirs before descendant conftests or test modules can create
# a caretaker. Do not use pytest's basetemp: it is cleared during the session.
_GUI_CACHE_GUARD = GuiCacheGuard()


def pytest_configure(config: pytest.Config) -> None:
    """Register the cache guard with this session's active pytest configuration."""
    config.pluginmanager.register(_GUI_CACHE_GUARD, "gui-cache-guard")


def pytest_unconfigure() -> None:
    """Restore cache environment variables, including after startup errors."""
    # Also restore on startup errors before pytest_configure could register it.
    _GUI_CACHE_GUARD.pytest_unconfigure()


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LIB_PATH = os.path.join(_REPO_ROOT, "lib")
if _LIB_PATH not in sys.path:
    sys.path.insert(0, _LIB_PATH)


@pytest.fixture(autouse=True)
def _np_seed():
    np.random.seed(0)


@pytest.fixture(autouse=True)
def _close_matplotlib_figures():
    yield
    import matplotlib.pyplot as plt

    plt.close("all")
