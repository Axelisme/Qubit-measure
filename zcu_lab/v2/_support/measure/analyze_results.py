"""Qualify and serialize core fit-quality diagnostics for adapter summaries."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from zcu_tools.analysis.fitting import FitQuality, QualityIssue


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
