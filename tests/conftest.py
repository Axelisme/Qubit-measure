import os
import sys
import tempfile
from pathlib import Path
from tempfile import TemporaryDirectory

from tests._cache_environment import temporary_cache_root
from tests._temp_directory_guard import ContextDirectoryGuard

# Configure the cache before third-party imports or descendant conftests can
# create persistence/discovery services. Pytest basetemp is cleared mid-session.
# The bootstrap retains it through process exit. Never restore the user's cache
# environment during nested pytest.main() teardown.
os.environ["ZCU_TOOLS_CACHE_DIR"] = temporary_cache_root()

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
import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LIB_PATH = os.path.join(_REPO_ROOT, "lib")
if _LIB_PATH not in sys.path:
    sys.path.insert(0, _LIB_PATH)


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config: pytest.Config) -> None:
    """Register the context-directory guard and allocate an invocation-owned basetemp.

    The caller's basetemp wins when supplied. Xdist supplies each worker's
    basetemp beneath the controller's directory. Only the allocating
    controller (or serial session) owns cleanup.
    """
    config.pluginmanager.register(ContextDirectoryGuard(), "context-directory-guard")
    if config.option.basetemp is not None:
        return
    parent = Path(_REPO_ROOT) / ".pytest-tmp"
    parent.mkdir(exist_ok=True)
    directory = TemporaryDirectory(prefix=f"{os.getpid()}-", dir=parent)
    config.option.basetemp = directory.name
    config.add_cleanup(directory.cleanup)


@pytest.fixture(scope="module", autouse=True)
def synthetic_parameter_schema_policy():
    """Bootstrap the framework suite with a synthetic version, restoring state.

    Entry users span GUI, notebook, MCP and resources. Individual tests may
    install another policy but must restore it before leaving their own scope.
    """
    from zcu_tools.resources.entry import component_registry
    from zcu_tools.resources.parameter_schema import ParameterSchemaPolicy

    before = component_registry.schema_policy
    policy = ParameterSchemaPolicy(1, ("synthetic-upgrade",))
    component_registry.configure_schema(policy)
    try:
        yield
        assert component_registry.schema_policy == policy, "schema policy polluter"
    finally:
        component_registry.configure_schema(before)


@pytest.fixture(scope="session", autouse=True, name="_owned_tempfile_root")
def fixture_owned_tempfile_root(tmp_path_factory, request):
    """Keep bare tempfile calls inside this process's pytest-owned lifetime."""
    root = tmp_path_factory.mktemp("tempfile")
    guard = request.config.pluginmanager.get_plugin("context-directory-guard")
    guard.start(root)
    previous = tempfile.tempdir
    tempfile.tempdir = str(root)
    try:
        yield
    finally:
        tempfile.tempdir = previous


@pytest.fixture(autouse=True, name="_isolated_fake_device_instances")
def fixture_isolated_fake_device_instances():
    """Restore the process-wide FakeDevice address scope after every test."""
    from zcu_tools.device import FakeDevice

    before = FakeDevice.instance_addresses()
    try:
        with FakeDevice.isolated_instances():
            assert FakeDevice.instance_addresses() == ()
            yield
    finally:
        assert FakeDevice.instance_addresses() == before, (
            "FakeDevice address scope leaked"
        )


@pytest.fixture(autouse=True, name="_np_seed")
def fixture_np_seed():
    """Seed NumPy randomness before each test."""
    np.random.seed(0)


@pytest.fixture(autouse=True, name="_close_matplotlib_figures")
def fixture_close_matplotlib_figures():
    """Close all Matplotlib figures after each test."""
    yield
    import matplotlib.pyplot as plt

    plt.close("all")
