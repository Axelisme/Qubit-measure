"""Common best-setting report fields for the three thermal-chain recipes."""

from pydantic import Field
from typing_extensions import TypedDict
from zcu_tools.mcp.measure.recipe_api import RecipeReport, report_number

from zcu_lab.v2.twotone.ro_optimize._chain_common import ChainAnalysis


class ChainSweepReport(RecipeReport):
    """Best-setting diagnostics; each recipe adds the swept setting and unit.

    The named metrics stay null when analysis gives no result.
    Errors are model-conditional approximate scales, not general one-sigma errors.
    """

    score: float | None = Field(
        default=None,
        description="Thermal-chain score at the selected setting; null if unavailable.",
    )
    score_error: float | None = Field(
        default=None,
        description="Model-conditional approximate score error scale at fixed M and main axis; not a general one-sigma error.",
    )
    qnd: float | None = Field(
        default=None,
        description="Estimated QND at the selected setting, without clipping; null if unavailable.",
    )
    qnd_error: float | None = Field(
        default=None,
        description="Model-conditional approximate QND error scale; not a general one-sigma error.",
    )
    correlation: float | None = Field(
        default=None,
        description="rho: first-pair covariance / R1 variance on the main axis.",
    )
    amplification: float | None = Field(
        default=None,
        description="G: restricted two-state 15-lag information amplification proxy.",
    )
    mist: float | None = Field(
        default=None,
        description="M: within-chain correlation growth, including two-state repopulation; not identifiable physical leakage.",
    )


class ChainReportFields(TypedDict):
    """Report the named diagnostics from the best SettingScore."""

    score: float | None
    score_error: float | None
    qnd: float | None
    qnd_error: float | None
    correlation: float | None
    amplification: float | None
    mist: float | None


def chain_report_fields(analysis: ChainAnalysis) -> ChainReportFields:
    """Copy the best score diagnostics; report_number maps non-finite values to null."""
    best = analysis.best
    return {
        "score": report_number(best.score),
        "score_error": report_number(best.score_error),
        "qnd": report_number(best.qnd),
        "qnd_error": report_number(best.qnd_error),
        "correlation": report_number(best.correlation),
        "amplification": report_number(best.amplification),
        "mist": report_number(best.mist),
    }
