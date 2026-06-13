"""Shared analyze-result shapes + helpers reused across adapters.

``FigureOnlyAnalyzeResult`` is the look-at-the-curve analyze result: a single
``figure`` field and nothing else (no fitted scalar, hence no writeback). The
reset length sweeps (single/dual/bath) all share this shape, so they subclass it
rather than redeclaring ``figure: Figure`` each.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

from matplotlib.figure import Figure

from zcu_tools.gui.app.main.adapter import AnalyzeResultBase

if TYPE_CHECKING:
    from zcu_tools.gui.app.main.adapter import AnalyzeRequest

_FigureOnlyT = TypeVar("_FigureOnlyT", bound="FigureOnlyAnalyzeResult")


@dataclass
class FigureOnlyAnalyzeResult(AnalyzeResultBase):
    """A look-at-the-curve analyze result: carries only the rendered figure.

    The analysis renders a trace for the Analyze tab but extracts no scalar, so
    there is no writeback. Adapters declare a named subclass (so the result type
    stays distinct) but inherit the single ``figure`` field.
    """

    figure: Figure


def run_figure_only_analyze(
    exp_cls: type[Any],
    result_cls: type[_FigureOnlyT],
    req: AnalyzeRequest[Any, Any],
) -> _FigureOnlyT:
    """Render ``exp_cls().analyze(run_result)`` into a figure-only result.

    Shared by the reset length sweeps whose ``analyze`` is exactly "run the
    experiment's figure-producing analyze, wrap the figure" — the only per-adapter
    variation is which ``exp_cls`` and which figure-only ``result_cls`` (the return
    type is exactly that subclass)."""
    return result_cls(figure=exp_cls().analyze(req.run_result))
