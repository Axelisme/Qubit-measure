"""Verify first-import effects in a fresh interpreter, not a reused module cache."""

import os
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import pytest

pytestmark = pytest.mark.requires_subprocess


@pytest.mark.parametrize("backend", ["Agg", None])
@pytest.mark.parametrize("interactive", [False, True])
def test_notebook_import_is_lazy_and_first_presentation_selects_ipympl(
    backend: str | None, interactive: bool, tmp_path: Path
) -> None:
    source = dedent("""
        import sys
        import matplotlib
        import matplotlib.pyplot as plt

        if sys.argv[1] == "Agg":
            matplotlib.use("Agg")
        matplotlib.interactive(sys.argv[2] == "True")
        retained = plt.figure("caller-owned") if sys.argv[1] == "Agg" else None
        before_backend = matplotlib.get_backend(auto_select=False)
        before_figures = plt.get_fignums()
        before_interactive = matplotlib.is_interactive()
        assert before_backend == ("Agg" if retained is not None else None)
        assert "ipympl" not in sys.modules

        import zcu_tools.notebook
        from zcu_tools.notebook.plotting import NotebookPlotHost
        from zcu_tools.plotting.plots import Plots

        plots = Plots(NotebookPlotHost())
        try:
            figure, _ = plots.subplots(num="operation")
            assert "ipympl" not in sys.modules
            assert matplotlib.get_backend(auto_select=False) == before_backend
            figure.show()
            assert matplotlib.get_backend(auto_select=False) == "module://ipympl.backend_nbagg"
            assert plt.get_fignums() == before_figures
            assert matplotlib.is_interactive() == before_interactive
            if retained is not None:
                assert plt.figure("caller-owned") is retained
            plots.finish()
        finally:
            plots.finish(present=False)
            plots.release()
            if retained is not None:
                plt.close(retained)
    """)
    environment = {**os.environ, "MPLCONFIGDIR": str(tmp_path)}
    environment.pop("MPLBACKEND", None)
    result = subprocess.run(
        [sys.executable, "-c", source, str(backend), str(interactive)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
