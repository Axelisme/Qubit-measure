"""Exercise the experiment-boundary contract through the external lint-imports CLI."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


def _lint_sources(
    root: Path, sources: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    for relative, source in sources.items():
        path = root / "zcu_lab" / "v2" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        for parent in path.parents:
            if parent == root:
                break
            (parent / "__init__.py").touch(exist_ok=True)
        path.write_text(source, encoding="utf-8")
    (root / ".importlinter").write_text(
        "[importlinter]\n"
        "root_package = zcu_lab\n"
        "contract_types =\n"
        "    experiment-imports: tools.check_experiment_imports.ExperimentImportsContract\n"
        "\n[importlinter:contract:experiments]\n"
        "name = Experiment directories are independent\n"
        "type = experiment-imports\n",
        encoding="utf-8",
    )
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2]))
    return subprocess.run(
        [
            str(Path(sys.executable).with_name("lint-imports")),
            "--no-cache",
            "--no-logo",
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )


@pytest.mark.parametrize(
    ("entry", "statement", "target"),
    [
        ("gui.py", "from ..second.analysis import result\n", "second.analysis"),
        ("__init__.py", "from ..second import core\n", "second.core"),
        ("nested/view.py", "from ... import second\n", "second"),
    ],
)
def test_cross_experiment_import_is_rejected_and_removal_passes(
    tmp_path: Path, entry: str, statement: str, target: str
) -> None:
    sources = {
        "family/first/core.py": "",
        f"family/first/{entry}": statement,
        "family/second/core.py": "",
        "family/second/analysis.py": "result = 1\n",
    }
    rejected = _lint_sources(tmp_path, sources)
    assert rejected.returncode == 1, rejected.stdout + rejected.stderr
    assert "zcu_lab.v2.family.first" in rejected.stdout
    assert f"zcu_lab.v2.family.{target}" in rejected.stdout

    sources[f"family/first/{entry}"] = ""
    allowed = _lint_sources(tmp_path, sources)
    assert allowed.returncode == 0, allowed.stdout + allowed.stderr


@pytest.mark.parametrize(
    ("statement", "target"),
    [
        ("from ...other.nested.helper import value\n", "other.nested.helper"),
        ("from ... import other\n", "other"),
    ],
)
def test_foreign_family_helper_is_rejected_and_removal_passes(
    tmp_path: Path, statement: str, target: str
) -> None:
    sources = {
        "family/first/core.py": statement,
        "other/second/core.py": "",
        "other/nested/helper.py": "value = 1\n",
    }
    rejected = _lint_sources(tmp_path, sources)
    assert rejected.returncode == 1, rejected.stdout + rejected.stderr
    assert "zcu_lab.v2.family.first.core" in rejected.stdout
    assert f"zcu_lab.v2.{target}" in rejected.stdout
    assert "l. 1" in rejected.stdout

    sources["family/first/core.py"] = ""
    allowed = _lint_sources(tmp_path, sources)
    assert allowed.returncode == 0, allowed.stdout + allowed.stderr


def test_own_family_support_and_family_exports_are_allowed(tmp_path: Path) -> None:
    allowed = _lint_sources(
        tmp_path,
        {
            "family/__init__.py": (
                "from .first.core import value\nfrom .nested.second.core import value\n"
            ),
            "family/first/__init__.py": "from .core import value\n",
            "family/first/core.py": (
                "from .._helper import value\n"
                "from ..nested.helper import value\n"
                "from ...support.numeric import value\n"
            ),
            "family/first/analysis.py": "from .core import value\n",
            "family/nested/second/core.py": "from ..helper import value\n",
            "family/_helper.py": "value = 1\n",
            "family/nested/helper.py": "value = 1\n",
            "support/numeric.py": "value = 1\n",
        },
    )
    assert allowed.returncode == 0, allowed.stdout + allowed.stderr
