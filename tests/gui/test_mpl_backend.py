from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


def test_gui_package_import_is_matplotlib_clean():
    """``import zcu_tools.gui`` must not pull in matplotlib.

    This is the invariant that lets an entry script do
    ``from zcu_tools.gui import configure_gui_matplotlib_backend`` first and
    configure the backend before any pyplot import. Run in a subprocess because
    the pytest process has already imported matplotlib.
    """
    repo_root = Path(__file__).resolve().parents[2]
    script = textwrap.dedent(
        """
        import sys
        import zcu_tools.gui  # noqa: F401

        leaked = sorted(m for m in sys.modules if m.split(".")[0] == "matplotlib")
        assert not leaked, f"importing zcu_tools.gui leaked matplotlib: {leaked}"
        print("ok")
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


def test_configure_gui_matplotlib_backend_fails_after_pyplot_import():
    import matplotlib.pyplot as plt

    del plt

    from zcu_tools.gui.mpl_backend_setup import configure_gui_matplotlib_backend

    with pytest.raises(RuntimeError, match="already imported"):
        configure_gui_matplotlib_backend()


def test_custom_backend_supports_pyplot_in_active_container():
    repo_root = Path(__file__).resolve().parents[2]
    script = textwrap.dedent(
        """
        from zcu_tools.gui.mpl_backend_setup import (
            BACKEND_NAME,
            configure_gui_matplotlib_backend,
        )

        configure_gui_matplotlib_backend()

        from qtpy.QtWidgets import QApplication, QLabel, QStackedWidget
        from zcu_tools.gui.plot_host import FigureContainer
        from zcu_tools.gui.plot_routing import routing_scope

        import matplotlib
        import matplotlib.pyplot as plt

        app = QApplication.instance() or QApplication([])
        stack = QStackedWidget()
        placeholder = QLabel("(placeholder)")
        stack.addWidget(placeholder)
        container = FigureContainer(stack, placeholder)

        with routing_scope(container):
            fig, ax = plt.subplots()
            ax.plot([0, 1], [0, 1])
            plt.show()

        assert matplotlib.get_backend().lower() == BACKEND_NAME
        assert type(fig.canvas.manager).__name__ == "GuiFigureManager"
        assert stack.count() == 2
        assert stack.currentWidget() is fig.canvas

        plt.close(fig)
        print("ok")
        """
    )
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


def test_configure_gui_matplotlib_backend_is_idempotent_after_preconfigure():
    repo_root = Path(__file__).resolve().parents[2]
    script = textwrap.dedent(
        """
        from zcu_tools.gui.mpl_backend_setup import configure_gui_matplotlib_backend

        configure_gui_matplotlib_backend()

        import matplotlib.pyplot as plt

        configure_gui_matplotlib_backend()

        assert plt.get_backend().lower() == "module://zcu_tools.gui.mpl_backend"
        print("ok")
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


def test_make_empty_ctx_builds_runtime_context():
    from zcu_tools.gui.app import _make_empty_ctx

    ctx = _make_empty_ctx()

    assert ctx.soc is None
    assert ctx.soccfg is None
    assert ctx.md is not None
    assert ctx.ml is not None
