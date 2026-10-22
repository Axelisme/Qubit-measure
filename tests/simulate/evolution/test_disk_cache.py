"""Exercise machine-code cache loading and source evolution through public evolution."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_CHILD = """
import json
import os

# The package must also configure an already-imported Numba.
from numba import config
import numpy as np
from zcu_tools.simulate.evolution import RWACoefficients, interaction_hamiltonian

positive = np.array([[[0, 0.5], [0, 0]]], dtype=np.complex128)
coefficients = RWACoefficients(
    positive, positive.conj().transpose(0, 2, 1), np.zeros((1, 2))
)
result = interaction_hamiltonian(
    0.0, np.zeros(2), coefficients, np.ones(1, dtype=np.complex128)
)
print("RESULT=" + json.dumps({
    "matrix": result.real.tolist(),
    "cache": config.CACHE_DIR,
    "environment_cache": os.environ.get("NUMBA_CACHE_DIR"),
}))
"""


@pytest.fixture
def copied_package(tmp_path: Path) -> Path:
    source = Path(__file__).resolve().parents[3] / "lib" / "zcu_tools"
    target = tmp_path / "lib" / "zcu_tools"
    shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__"))
    return target


def launch(
    package: Path, cache: Path, *, numba_parent: Path | None = None
) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment.pop("NUMBA_CACHE_DIR", None)
    environment.update(
        PYTHONPATH=str(package.parent),
        ZCU_TOOLS_CACHE_DIR=str(cache),
        NUMBA_DEBUG_CACHE="1",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMBA_NUM_THREADS="1",
    )
    if numba_parent is not None:
        environment["NUMBA_CACHE_DIR"] = str(numba_parent)
    # Both interpreters finish a tiny 2x2 public calculation; timeout only guards hangs.
    return subprocess.run(
        [sys.executable, "-c", _CHILD],
        cwd=package.parent,
        env=environment,
        capture_output=True,
        text=True,
        timeout=25,
        check=False,
    )


def result(process: subprocess.CompletedProcess[str]) -> dict[str, object]:
    assert process.returncode == 0, process.stdout + process.stderr
    line = next(
        line for line in process.stdout.splitlines() if line.startswith("RESULT=")
    )
    return json.loads(line.removeprefix("RESULT="))


@pytest.mark.parametrize("override", [False, True], ids=["user-root", "numba-parent"])
def test_new_process_loads_machine_code_from_the_namespaced_cache(
    copied_package: Path, tmp_path: Path, override: bool
) -> None:
    root = tmp_path / "cache"
    parent = tmp_path / "external-numba" if override else None
    empty = launch(copied_package, root, numba_parent=parent)
    first = result(empty)
    np.testing.assert_array_equal(first["matrix"], [[0, 0.5], [0.5, 0]])
    destination = Path(str(first["cache"]))
    assert first["environment_cache"] == str(destination)
    assert destination.parent == (parent if override else root / "numba")
    prefix = "zcu-simulate-" if override else ""
    digest = destination.name.removeprefix(prefix)
    assert len(digest) == 64
    assert "[cache] data saved" in empty.stdout

    hit = launch(copied_package, root, numba_parent=parent)
    assert result(hit) == first
    assert "[cache] data loaded" in hit.stdout
    assert "[cache] data saved" not in hit.stdout


def test_callee_source_change_uses_new_machine_code_in_a_new_process(
    copied_package: Path, tmp_path: Path
) -> None:
    # This synthetic inline dependency tests cross-file cache invalidation,
    # not DOP853 physics. The public caller stays byte-for-byte unchanged
    # between processes, so Numba's own caller timestamp check cannot help.
    kernels = copied_package / "simulate" / "evolution" / "kernels"
    callee = kernels / "_disk_cache_fixture.py"
    callee.write_text(
        'from numba import njit\n@njit(inline="always")\n'
        "def disk_cache_scale():\n    return 1.0\n"
    )
    caller = kernels / "multitone_rwa.py"
    source = caller.read_text()
    source = source.replace(
        "from .._envelope import sample_value",
        "from ._disk_cache_fixture import disk_cache_scale\n"
        "from .._envelope import sample_value",
    )
    start = source.index("def interaction_hamiltonian(")
    source = source[:start] + source[start:].replace(
        "    return h\n", "    return h * disk_cache_scale()\n", 1
    )
    caller.write_text(source)

    root = tmp_path / "cache"
    first = result(launch(copied_package, root))
    np.testing.assert_array_equal(first["matrix"], [[0, 0.5], [0.5, 0]])
    # Preserve size and mtime too: correctness must depend on content, not stat.
    stamp = callee.stat()
    callee.write_text(callee.read_text().replace("return 1.0", "return 2.0"))
    os.utime(callee, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    for bytecode in kernels.glob("__pycache__/_disk_cache_fixture.*.pyc"):
        bytecode.unlink()
    changed = result(launch(copied_package, root))
    np.testing.assert_array_equal(changed["matrix"], [[0, 1.0], [1.0, 0]])
    assert changed["cache"] != first["cache"]


def test_unreadable_simulator_source_stops_import_before_using_disk_cache(
    copied_package: Path, tmp_path: Path
) -> None:
    missing = copied_package / "simulate" / "unreadable.py"
    missing.symlink_to(tmp_path / "missing-source")
    process = launch(copied_package, tmp_path / "cache")
    assert process.returncode != 0
    assert "FileNotFoundError" in process.stderr
    assert "unreadable.py" in process.stderr
