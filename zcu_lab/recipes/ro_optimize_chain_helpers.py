"""Shared report fields and analysis-to-report copy for readout-chain ro_optimize recipes."""

from pydantic import Field
from typing_extensions import TypedDict
from zcu_tools.analysis.readout_chain import SettingFailure
from zcu_tools.mcp.measure.recipe_api import RecipeReport, report_number

from zcu_lab.v2.twotone.ro_optimize._chain_common import ChainAnalysis


class ChainSweepReport(RecipeReport):
    """Report fields of the best setting and of the settings left out.

    Each recipe adds the best setting itself.

    Fields stay null or empty when the analysis gave no result.
    """

    metric: float | None = Field(
        default=None,
        description="All-lag information per read at the best setting; null if unavailable.",
    )
    metric_err: float | None = Field(
        default=None,
        description="One-sigma error of that information; null if unavailable.",
    )
    qnd: float | None = Field(
        default=None,
        description="QND of the best setting, between 0 and 1; null if unavailable.",
    )
    qnd_err: float | None = Field(
        default=None, description="One-sigma error of that QND; null if unavailable."
    )
    separation: float | None = Field(
        default=None, description="d/sigma at the best setting; null if unavailable."
    )
    retention: list[float | None] = Field(
        default_factory=list,
        description="λ of its two most populated states; empty if unavailable.",
    )
    third_population: float | None = Field(
        default=None,
        description="Third-state population at the best setting; null if unavailable.",
    )
    no_signal: list[float] = Field(
        default_factory=list,
        description="Settings, in the swept unit, the no-signal test stopped before the fit.",
    )
    failures: list[SettingFailure] = Field(
        default_factory=list,
        description="Settings, in the swept unit, whose fit failed, with the reason.",
    )


class ChainReportFields(TypedDict):
    """ChainSweepReport field values, passed as keyword arguments."""

    metric: float | None
    metric_err: float | None
    qnd: float | None
    qnd_err: float | None
    separation: float | None
    retention: list[float | None]
    third_population: float | None
    no_signal: list[float]
    failures: list[SettingFailure]


def chain_report_fields(analysis: ChainAnalysis) -> ChainReportFields:
    """Copy the best fit, no-signal settings and failures into report fields.

    Numbers pass through report_number unchanged; non-finite values become null.
    """
    best = analysis.best
    return {
        "metric": report_number(best.metric),
        "metric_err": report_number(best.standard_error),
        "qnd": report_number(best.qnd),
        "qnd_err": report_number(best.qnd_error),
        "separation": report_number(best.separation),
        "retention": [report_number(value) for value in best.retention],
        "third_population": report_number(best.third_population),
        "no_signal": list(analysis.no_signal),
        "failures": list(analysis.failures),
    }
