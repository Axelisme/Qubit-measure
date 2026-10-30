"""Fresh-interpreter import boundary for NotebookPlotHost's optional backend."""

import os
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import pytest


@pytest.mark.parametrize("interactive", [False, True])
def test_first_notebook_presentation_preserves_selected_backend_and_pyplot_figures(
    interactive: bool, tmp_path: Path
) -> None:
    source = dedent("""
        import sys
        import matplotlib
        import matplotlib.pyplot as plt

        matplotlib.use("Agg")
        matplotlib.interactive(sys.argv[1] == "True")
        retained = plt.figure("caller-owned")
        before = (matplotlib.get_backend(auto_select=False),
                  plt.get_fignums(), matplotlib.is_interactive())
        assert "ipympl" not in sys.modules
        from zcu_tools.notebook.plotting import NotebookPlotHost
        from zcu_tools.plotting.plots import Plots

        plots = Plots(NotebookPlotHost())
        try:
            figure, _ = plots.subplots(num="operation")
            figure.show()
            after = (matplotlib.get_backend(auto_select=False),
                     plt.get_fignums(), matplotlib.is_interactive())
            assert after == before, (before, after)
            assert plt.figure("caller-owned") is retained
            plots.finish()
        finally:
            plots.finish(present=False)
            plots.release()
            plt.close(retained)
    """)
    result = subprocess.run(
        [sys.executable, "-c", source, str(interactive)],
        env={**os.environ, "MPLCONFIGDIR": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
