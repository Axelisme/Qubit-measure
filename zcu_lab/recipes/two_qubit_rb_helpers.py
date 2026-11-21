"""Shared inputs and estimate fields for paired two-qubit RB recipes."""

from typing import Annotated

from pydantic import Field, StrictInt
from typing_extensions import TypedDict
from zcu_tools.mcp.measure.recipe_api import (
    KEEP,
    Keep,
    Overrides,
    RecipeInputs,
    RecipeReport,
    ReuseTabId,
    report_number,
)

from zcu_lab.v2.two_qubit.rb.core import RBEstimate

from .two_qubit_helpers import TwoQubitTarget


class RBRecipeInputs(RecipeInputs):
    """Paired RB scan inputs; overrides apply after the named fields."""

    target: TwoQubitTarget | None = Field(
        default=None,
        description="Role bindings for this Reset. Omitted or null roles keep the current aliases.",
    )
    reuse_tab_id: ReuseTabId
    overrides: Overrides
    depths: list[Annotated[StrictInt, Field(ge=0)]] | None | Keep = Field(
        KEEP, description="Random Clifford counts, written to Form depth."
    )
    seeds: list[StrictInt] | None | Keep = Field(
        KEEP,
        description="Integer sequence identities paired across arms, written to Form seed.",
    )
    reps: int | None | Keep = Field(
        KEEP, description="Shots per acquisition point per round."
    )
    rounds: int | None | Keep = Field(
        KEEP, description="Number of rounds to sum per seed and arm."
    )


class RBEstimateReport(RecipeReport):
    """Dimensionless owner estimates; all estimate fields are null without a fit."""

    p_reference: float | None = Field(
        default=None, description="Reference decay per random Clifford."
    )
    p_interleaved: float | None = Field(
        default=None, description="Interleaved decay per random Clifford plus CZ."
    )
    reference_epc: float | None = Field(
        default=None, description="Reference error per Clifford for dimension four."
    )
    gate_error: float | None = Field(
        default=None, description="Dimension-four CZ error estimate, not clipped."
    )
    error_ci_low: float | None = Field(
        default=None,
        description="Lower bound of the 95% paired-seed gate error interval.",
    )
    error_ci_high: float | None = Field(
        default=None,
        description="Upper bound of the 95% paired-seed gate error interval.",
    )
    n_paired_seeds: int | None = Field(
        default=None, description="Seed identities contributing to the paired fit."
    )
    bootstrap_successes: int | None = Field(
        default=None, description="Valid paired-seed bootstrap fits."
    )
    warning: str | None = Field(
        default=None, description="Owner warning about the ratio and model limitations."
    )
    reason: str | None = Field(
        default=None, description="Owner reason when analysis yields no estimate."
    )


class RBEstimateFields(TypedDict, total=False):
    """Flat estimate fields; no keys when the owner has no estimate."""

    p_reference: float | None
    p_interleaved: float | None
    reference_epc: float | None
    gate_error: float | None
    error_ci_low: float | None
    error_ci_high: float | None
    n_paired_seeds: int
    bootstrap_successes: int
    warning: str


def rb_estimate_fields(estimate: RBEstimate | None) -> RBEstimateFields:
    """Forward the owner estimate, mapping non-finite scalars to null."""
    if estimate is None:
        return {}
    return {
        "p_reference": report_number(estimate.p_reference),
        "p_interleaved": report_number(estimate.p_interleaved),
        "reference_epc": report_number(estimate.reference_epc),
        "gate_error": report_number(estimate.gate_error),
        "error_ci_low": report_number(estimate.error_ci_low),
        "error_ci_high": report_number(estimate.error_ci_high),
        "n_paired_seeds": estimate.n_paired_seeds,
        "bootstrap_successes": estimate.bootstrap_successes,
        "warning": estimate.warning,
    }
