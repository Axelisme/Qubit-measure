"""Shared GUI plotting substrate: the embedded matplotlib backend + host.

App-agnostic. A ``module://`` backend (``backend.py``, the *client*) intercepts
pyplot figure creation and forwards to a single main-thread ``host`` that embeds
figures into ``FigureContainer`` widgets; ``routing`` picks which container new
figures go to (task-local ContextVar). Both measure-gui and fluxdep use this.

Import discipline (keeps the backend-select path import-clean):
- ``from zcu_tools.gui.plotting.setup import configure_matplotlib_backend`` —
  this module imports only ``sys`` (matplotlib loaded inside the function), so an
  entry script can select the backend *before* any pyplot import.
- The heavy names below pull in matplotlib + qtpy; import them only after the
  backend is selected (i.e. when building the UI). This package ``__init__`` does
  NOT import them at module scope — it re-exports lazily via ``__getattr__`` — so
  ``import zcu_tools.gui.plotting`` alone stays matplotlib-clean.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .container import FigureContainer
    from .host import (
        PlotStateSnapshot,
        activate_figure,
        assert_plot_invariants,
        attach_existing_figure_to_container,
        attach_figure_to_current_container,
        close_figure,
        dump_plot_state,
        ensure_host,
        get_figure_container,
        is_main_thread,
        refresh_figure_in_main_thread,
        remove_canvas,
        set_shutting_down,
    )
    from .mathtext_lock import (
        install_mathtext_lock,
        prewarm_mathtext,
    )
    from .routing import (
        require_current_container,
        routing_scope,
    )

# Public name → (submodule, attribute). Resolved lazily so importing this package
# does not drag in matplotlib/qtpy (those load only when a name is first used,
# which is after backend selection).
_LAZY: dict[str, str] = {
    "FigureContainer": "container",
    "PlotStateSnapshot": "host",
    "activate_figure": "host",
    "assert_plot_invariants": "host",
    "attach_existing_figure_to_container": "host",
    "attach_figure_to_current_container": "host",
    "close_figure": "host",
    "dump_plot_state": "host",
    "ensure_host": "host",
    "get_figure_container": "host",
    "install_mathtext_lock": "mathtext_lock",
    "is_main_thread": "host",
    "prewarm_mathtext": "mathtext_lock",
    "refresh_figure_in_main_thread": "host",
    "remove_canvas": "host",
    "set_shutting_down": "host",
    "require_current_container": "routing",
    "routing_scope": "routing",
}

__all__ = [
    "FigureContainer",
    "PlotStateSnapshot",
    "activate_figure",
    "assert_plot_invariants",
    "attach_existing_figure_to_container",
    "attach_figure_to_current_container",
    "close_figure",
    "dump_plot_state",
    "ensure_host",
    "get_figure_container",
    "install_mathtext_lock",
    "is_main_thread",
    "prewarm_mathtext",
    "refresh_figure_in_main_thread",
    "remove_canvas",
    "require_current_container",
    "routing_scope",
    "set_shutting_down",
]


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    submodule = importlib.import_module(f".{module}", __name__)
    return getattr(submodule, name)
