"""Public loading contracts for basic tools and optional Numba tool packages."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.requires_subprocess


def launch_import(
    code: str, tmp_path: Path, *, package_root: Path | None = None
) -> None:
    environment = dict(os.environ)
    environment.update(
        NUMBA_CACHE_DIR=str(tmp_path / "numba-parent"),
        NUMBA_CACHE_LOCATOR_CLASSES="InTreeCacheLocator",
        ZCU_TOOLS_CACHE_DIR=str(tmp_path / "user-cache"),
        PYTHONDONTWRITEBYTECODE="1",
    )
    if package_root is not None:
        environment["PYTHONPATH"] = str(package_root)
    # Each child finishes only a public import contract; timeout guards a hang.
    process = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=25,
        check=False,
    )
    assert process.returncode == 0, process.stdout + process.stderr


_BASIC_IMPORT = """
import os
import sys
from pathlib import Path

scans = []
def observe(event, args):
    if event == "os.walk":
        scans.append(str(args[0]))
sys.addaudithook(observe)
parent = os.environ["NUMBA_CACHE_DIR"]
locator = os.environ["NUMBA_CACHE_LOCATOR_CLASSES"]
from zcu_tools.simulate import flux2value, value2flux
assert value2flux(3.0, 1.0, 4.0) == 1.0
assert flux2value(1.0, 1.0, 4.0) == 3.0
assert scans == [], scans
assert os.environ["NUMBA_CACHE_DIR"] == parent
assert os.environ["NUMBA_CACHE_LOCATOR_CLASSES"] == locator
assert not Path(parent).exists()
assert not Path(os.environ["ZCU_TOOLS_CACHE_DIR"]).exists()
"""


@pytest.mark.parametrize("unavailable", ["numba", "scipy", "none"])
def test_basic_conversion_import_needs_no_optional_client_or_cache_policy(
    tmp_path: Path, unavailable: str
) -> None:
    # Model a missing external optional dependency, including its distribution
    # metadata. The simulator and all its private collaborators stay untouched.
    dependency = f"""
import importlib.metadata as metadata
import sys
unavailable = {unavailable!r}
if unavailable != "none":
    sys.modules[unavailable] = None
original_version = metadata.version
def installed_version(package):
    if package == unavailable:
        raise metadata.PackageNotFoundError(package)
    return original_version(package)
metadata.version = installed_version
"""
    launch_import(dependency + _BASIC_IMPORT, tmp_path)


@pytest.mark.parametrize("first,second", [("adc", "evolution"), ("evolution", "adc")])
def test_numba_tools_share_one_namespace_in_either_import_order(
    tmp_path: Path, first: str, second: str
) -> None:
    launch_import(
        _BASIC_IMPORT
        + f"""
import importlib
from numba import config
importlib.import_module("zcu_tools.simulate.{first}")
directory = config.CACHE_DIR
assert Path(directory).parent == Path(parent)
assert Path(directory).name.startswith("zcu-simulate-")
assert config.CACHE_LOCATOR_CLASSES == ["UserProvidedCacheLocator"]
assert len(scans) == 1, scans
importlib.import_module("zcu_tools.simulate.{second}")
assert config.CACHE_DIR == directory
assert os.environ["NUMBA_CACHE_DIR"] == directory
assert len(scans) == 1, scans
""",
        tmp_path,
    )


def test_failed_numba_tool_initialization_is_retried_by_another_tool(
    tmp_path: Path,
) -> None:
    source = Path(__file__).resolve().parents[2] / "lib" / "zcu_tools"
    package = tmp_path / "lib" / "zcu_tools"
    shutil.copytree(source, package, ignore=shutil.ignore_patterns("__pycache__"))
    unreadable = package / "simulate" / "unreadable.py"
    unreadable.symlink_to(tmp_path / "missing-source")
    launch_import(
        _BASIC_IMPORT
        + f"""
import importlib
for tool in ("evolution", "adc"):
    try:
        importlib.import_module("zcu_tools.simulate." + tool)
    except FileNotFoundError as error:
        assert "unreadable.py" in str(error), error
    else:
        raise AssertionError("failed initialization must not count as success")
Path({str(unreadable)!r}).unlink()
importlib.import_module("zcu_tools.simulate.adc")
from numba import config
assert Path(config.CACHE_DIR).parent == Path(parent)
assert config.CACHE_LOCATOR_CLASSES == ["UserProvidedCacheLocator"]
importlib.import_module("zcu_tools.simulate.evolution")
assert os.environ["NUMBA_CACHE_DIR"] == config.CACHE_DIR
""",
        tmp_path,
        package_root=package.parent,
    )
