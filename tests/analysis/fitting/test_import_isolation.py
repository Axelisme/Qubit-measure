"""Fitting remains usable when hardware and interactive tools are unavailable."""

import subprocess
from pathlib import Path
from textwrap import dedent

import pytest

pytestmark = pytest.mark.requires_subprocess


@pytest.mark.parametrize(
    "unavailable", ["qick", "zcu_tools.analysis.fluxdep.line_picker"]
)
def test_fitting_without_hardware_or_interactive_picker(unavailable: str) -> None:
    lane = Path(__file__).resolve().parents[3]
    source = dedent(
        f"""
        import importlib.abc
        import sys

        class UnavailableDependency(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == {unavailable!r} or fullname.startswith({unavailable!r} + "."):
                    raise ImportError("unrelated dependency requested: " + fullname)

        sys.meta_path.insert(0, UnavailableDependency())

        import numpy as np
        from zcu_tools.analysis.fitting import fitexp

        x = np.linspace(0.0, 10.0, 100)
        y = 0.2 + 1.3 * np.exp(-x / 2.5)
        parameters, covariance = fitexp(x, y)
        np.testing.assert_allclose(parameters, [0.2, 1.3, 2.5], rtol=1e-3)
        assert covariance.shape == (3, 3)
        """
    )
    result = subprocess.run(
        [
            "uv",
            "run",
            "--directory",
            str(lane),
            "--no-sync",
            "--",
            "python",
            "-c",
            source,
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
