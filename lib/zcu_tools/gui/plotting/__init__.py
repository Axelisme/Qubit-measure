"""Explicit GUI figure containers, host bridge and rendering support.

Exports load lazily so importing this package does not initialize Qt or
Matplotlib. Callers own figures and select their presentation container.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .container import FigureContainer
    from .host import (
        PlotStateSnapshot,
        assert_plot_invariants,
        attach_existing_figure_to_container,
        dump_plot_state,
        ensure_host,
        get_figure_container,
        remove_canvas,
        set_shutting_down,
    )
    from .mathtext_lock import (
        install_mathtext_lock,
        prewarm_mathtext,
    )

# Public name → (submodule, attribute). Resolved lazily so importing this package
# does not drag in matplotlib/qtpy; those load only when a name is first used.
_LAZY: dict[str, str] = {
    "FigureContainer": "container",
    "PlotStateSnapshot": "host",
    "assert_plot_invariants": "host",
    "attach_existing_figure_to_container": "host",
    "dump_plot_state": "host",
    "ensure_host": "host",
    "get_figure_container": "host",
    "install_mathtext_lock": "mathtext_lock",
    "prewarm_mathtext": "mathtext_lock",
    "remove_canvas": "host",
    "set_shutting_down": "host",
}

__all__ = [
    "FigureContainer",
    "PlotStateSnapshot",
    "assert_plot_invariants",
    "attach_existing_figure_to_container",
    "dump_plot_state",
    "ensure_host",
    "get_figure_container",
    "install_mathtext_lock",
    "prewarm_mathtext",
    "remove_canvas",
    "set_shutting_down",
]


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    submodule = importlib.import_module(f".{module}", __name__)
    return getattr(submodule, name)
