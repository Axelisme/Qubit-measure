"""Shared analyze-result shapes + helpers reused across adapters.

``FigureOnlyAnalyzeResult`` is the look-at-the-curve analyze result: a single
``figure`` field and nothing else (no fitted scalar, hence no writeback). The
reset length sweeps share this shape, so they subclass it rather than
redeclaring ``figure: Figure`` each.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, TypeVar

from matplotlib.figure import Figure
from zcu_tools.analysis.fitting import FitQuality, QualityIssue
from zcu_tools.gui.app.measure.adapter import AnalyzeResultBase

if TYPE_CHECKING:
    from zcu_tools.gui.app.measure.adapter import AnalyzeRequest

_FigureOnlyT = TypeVar("_FigureOnlyT", bound="FigureOnlyAnalyzeResult")


def fit_quality_models(
    qualities: Mapping[str, FitQuality] | None,
    *,
    summary_path: str = "summary.fit_quality",
) -> dict[str, FitQuality] | None:
    """Copy owner metrics, qualifying invalid paths without mutating core diagnostics.

    qualities maps fit names to core diagnostics; None remains None.
    summary_path is the JSON location of this map, including a mux row offset
    where applicable. Values and writeback eligibility remain unchanged.
    """
    if qualities is None:
        return None
    return {
        name: replace(
            quality,
            invalid=tuple(
                QualityIssue(
                    path=f"{summary_path}.{name}.{issue.path}", reason=issue.reason
                )
                for issue in quality.invalid
            ),
        )
        for name, quality in qualities.items()
    }


def fit_quality_summary(
    qualities: Mapping[str, FitQuality] | None,
    *,
    summary_path: str = "summary.fit_quality",
) -> dict[str, dict[str, object]] | None:
    """Serialize qualified owner diagnostics; None retains unavailable quality.

    summary_path locates the map in the returned JSON summary. No fit,
    acceptance decision or input mutation occurs.
    """
    models = fit_quality_models(qualities, summary_path=summary_path)
    return (
        None
        if models is None
        else {name: quality.to_summary_dict() for name, quality in models.items()}
    )


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

    Shared by reset visual checks whose ``analyze`` is exactly "run the
    experiment's figure-producing analyze, wrap the figure" — the only per-adapter
    variation is which ``exp_cls`` and which figure-only ``result_cls`` (the return
    type is exactly that subclass)."""
    return result_cls(figure=exp_cls().analyze(req.run_result))
