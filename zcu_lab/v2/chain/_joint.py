"""Scalar publication and unlabeled trace artists for K-cycle diagnostics."""

from dataclasses import dataclass

import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray
from pydantic import BaseModel, Field
from zcu_tools.analysis.readout_chain import CycleResidualProjection


@dataclass(frozen=True, kw_only=True)
class JointResidualDiagnostics:
    """Flat scalar diagnostics from the last joint stage, never a residual series."""

    J: float
    """Sum of squares of the joint whitened residuals."""
    df: int
    """Nominal residual degrees of freedom, 65 per point minus joint parameters."""
    j_per_df: float
    """J divided by df, or NaN if df is nonpositive."""
    expected_band_low: float
    """Lower endpoint of the approximate chi-square reference band, or NaN if df <= 0."""
    expected_band_high: float
    """Upper endpoint of the approximate chi-square reference band, or NaN if df <= 0."""
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


def plot_joint_residuals(
    projection_ax: Axes,
    residual_ax: Axes,
    axis: NDArray[np.float64],
    grid: NDArray[np.float64],
    view: CycleResidualProjection,
    labels: tuple[str, str],
) -> None:
    """Draw measured and fitted projections and measured-minus-fitted residual points.

    axis and grid are the acquired and model coordinates used to compute view.
    labels gives the experiment's legend text for measured points and the fitted
    curve, in that order. The caller owns all axis labels and figure layout.
    """
    projection_ax.plot(
        axis,
        view.measured_sigma,
        linestyle="none",
        marker="o",
        markersize=4,
        label=labels[0],
    )
    projection_ax.plot(
        grid, view.model_sigma, linestyle="-", marker="none", label=labels[1]
    )
    residual_ax.axhline(0, color="0.4", linewidth=1)
    residual_ax.plot(
        axis,
        view.measured_sigma - view.predicted_sigma,
        linestyle="none",
        marker="o",
        markersize=4,
    )
