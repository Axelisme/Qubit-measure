"""Shared presentation of one fitted K-cycle model; experiments own their figures."""

from typing import TypedDict

import numpy as np
from matplotlib.axes import Axes
from pydantic import BaseModel, Field
from zcu_tools.analysis.readout_chain import (
    CycleFit,
    CycleModel,
    CycleSums,
    RateModel,
    cycle_residual_projection,
)


class JointResidualFields(TypedDict):
    """Flat scalar diagnostics from the last joint stage, never a residual series."""

    J: float
    """Sum of squares of the joint whitened residuals."""
    df: int
    """Nominal residual degrees of freedom, 65 per point minus joint parameters."""
    j_per_df: float
    """J divided by df, or NaN if df is nonpositive."""
    expected_band_low: float
    """Lower endpoint of the approximate chi-square reference band for J/df."""
    expected_band_high: float
    """Upper endpoint of the approximate chi-square reference band for J/df."""
    tail_fraction: float
    """Fraction (0 to 1) of whitened residual components with magnitude above 3."""


class JointResidualSummary(BaseModel):
    """Typed JSON fields for joint residuals; null means unavailable."""

    J: float | None = Field(description="Sum of squared joint whitened residuals.")
    df: int | None = Field(description="Nominal joint residual degrees of freedom.")
    j_per_df: float | None = Field(description="Joint residual sum J divided by df.")
    expected_band_low: float | None = Field(
        description="Lower endpoint of the approximate chi-square J/df reference band."
    )
    expected_band_high: float | None = Field(
        description="Upper endpoint of the approximate chi-square J/df reference band."
    )
    tail_fraction: float | None = Field(
        description="Fraction (0 to 1) of joint whitened residual components with |r| > 3."
    )


def joint_residual_fields(fit: CycleFit) -> JointResidualFields:
    """Copy the fit's diagnostics to flat summary fields without recomputing them."""
    low, high = fit.expected_band
    return {
        "J": fit.J,
        "df": fit.df,
        "j_per_df": fit.j_per_df,
        "expected_band_low": low,
        "expected_band_high": high,
        "tail_fraction": fit.tail_fraction,
    }


def joint_residuals_text(fit: CycleFit) -> str:
    """Format two lines of diagnostics for a figure, without acceptance labels."""
    low, high = fit.expected_band
    return (
        f"J = {fit.J:.5g}    df = {fit.df}    J/df = {fit.j_per_df:.5g}\n"
        f"χ² reference = [{low:.4f}, {high:.4f}]    Fraction |r| > 3: {fit.tail_fraction:.3%}"
    )


def plot_joint_residuals(
    projection_ax: Axes,
    residual_ax: Axes,
    sums: CycleSums,
    model: CycleModel | RateModel,
    fit: CycleFit,
) -> None:
    """Add live-direction points, a dense joint curve and signed residual points.

    Pass the same sums and model as in fit_cycle. Both axes use the sweep
    coordinates; y values use the live projection's point-mean sigma units.
    The residual is measured minus predicted at the acquired points, not the
    whitened 65-feature vector. Callers own figure layout, x labels, annotations
    and legend placement. Propagate the projection's input errors.
    """
    axis = sums.axis
    grid = np.linspace(axis.min(), axis.max(), 400)
    view = cycle_residual_projection(sums, model, fit, model_axis=grid)
    projection_ax.plot(
        axis,
        view.measured_sigma,
        linestyle="none",
        marker="o",
        markersize=4,
        label="Measured · all 65 numbers",
    )
    projection_ax.plot(
        grid, view.model_sigma, linestyle="-", marker="none", label="Joint model"
    )
    projection_ax.set_ylabel("Projected signal (sigma)")

    residual_ax.axhline(0, color="0.4", linewidth=1)
    residual_ax.plot(
        axis,
        view.measured_sigma - view.predicted_sigma,
        linestyle="none",
        marker="o",
        markersize=4,
    )
    residual_ax.set_ylabel("Measured − model (sigma)")
