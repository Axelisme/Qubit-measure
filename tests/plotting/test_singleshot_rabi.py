"""Joint Rabi charts consume numeric fit data and caller-owned display values."""

from typing import Literal

import numpy as np
import pytest
from zcu_tools.analysis.fitting.singleshot_rabi import (
    RabiBackendResult,
    RabiJointFitResult,
    RabiProjection,
)
from zcu_tools.plotting.plots import NonPresentingHost, Plots
from zcu_tools.plotting.singleshot_rabi import plot_rabi_joint


def _fit(*, valid: bool) -> RabiJointFitResult:
    populations = np.array([[0.5, 0.5, 0.0], [0.25, 0.75, 0.0]])
    return RabiJointFitResult(
        initial_populations=populations[0],
        p_inf=0.5,
        t_r=None,
        omega=1.0,
        projected_g_center=-1.0,
        projected_e_center=1.0,
        sigma=0.3,
        p_avg=0.2,
        length_ratio=0.5,
        g_center=-1.0,
        e_center=1.0,
        radius=0.5,
        confusion_matrix=np.eye(3),
        condition_number=1.0,
        measured_populations=populations,
        fitted_populations=populations,
        projection=RabiProjection(
            projected=np.array([[-1.0, 1.0], [-1.0, 1.0]]),
            bin_edges=np.array([-2.0, 0.0, 2.0]),
            counts=np.ones((2, 2), dtype=np.int64),
            origin=0j,
            axis=1 + 0j,
            perpendicular_offset=0.0,
        ),
        backend=RabiBackendResult(
            parameter_names=(),
            values=np.empty(0),
            covariance=np.empty((0, 0)),
            valid=valid,
            covariance_accurate=valid,
            reached_call_limit=False,
            hesse_failed=False,
            edm=0.0,
            nll=0.0,
            calls=0,
        ),
    )


@pytest.mark.parametrize("sweep", ["length", "gain"])
@pytest.mark.parametrize("ro_length_us", [None, 2.0])
def test_chart_uses_explicit_duration_size_and_coordinate_labels(
    sweep: Literal["length", "gain"], ro_length_us: float | None
) -> None:
    plots = Plots(NonPresentingHost())
    try:
        plot_rabi_joint(
            np.array([0.0, 1.0]),
            np.array([[-1.0, 1.0], [-1.0, 1.0]], dtype=np.complex128),
            _fit(valid=True),
            ro_length_us,
            sweep=sweep,
            plots=plots,
            figsize=(6.0, 4.0),
        )
        assert tuple(plots) == ("fit",)
        fig = plots["fit"]
        np.testing.assert_allclose(fig.get_size_inches(), [6.0, 6.4])
        population, histogram, confusion = fig.axes
        assert population.get_xlabel() == (
            "Pulse length (μs)" if sweep == "length" else "Pulse gain (a.u.)"
        )
        assert population.get_ylabel() == "Population"
        assert "Fitted confusion matrix" in confusion.get_title()
        assert ("Effective $T_1$" in histogram.get_title()) is (
            ro_length_us is not None
        )
        if ro_length_us is not None:
            assert "4 μs" in histogram.get_title()
    finally:
        plots.finish()
        plots.release()


def test_invalid_fit_retains_measured_histogram_and_unavailable_calibration() -> None:
    plots = Plots(NonPresentingHost())
    try:
        plot_rabi_joint(
            np.array([0.0, 1.0]),
            np.array([[-1.0, 1.0], [-1.0, 1.0]], dtype=np.complex128),
            _fit(valid=False),
            2.0,
            sweep="length",
            plots=plots,
            figsize=(6.0, 4.0),
        )
        population, histogram, confusion = plots["fit"].axes
        assert population.get_title() == "Rabi joint fit invalid"
        assert histogram.get_title().endswith("Joint fit invalid")
        assert [artist.get_label() for artist in histogram.patches] == [
            "Observed counts"
        ]
        assert confusion.get_title() == "Confusion matrix unavailable"
    finally:
        plots.finish()
        plots.release()
