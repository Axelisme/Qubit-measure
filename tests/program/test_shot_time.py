"""Acquisition shot seconds through the public program interface."""

import numpy as np
import pytest
from numpy.typing import ArrayLike
from zcu_tools import program


def test_nonuniform_point_lengths_include_relaxation_on_every_shot() -> None:
    # One round/repetition takes 39 us; two rounds of three reps take 234 us.
    assert program.shot_time_s(
        rounds=2, reps=3, shot_lengths_us=[2.0, 5.0, 11.0], relax_delay_us=7.0
    ) == pytest.approx(0.000234)


def test_zero_duration_numpy_points_need_no_shot_time() -> None:
    assert (
        program.shot_time_s(
            rounds=1, reps=1, shot_lengths_us=np.zeros(2), relax_delay_us=0.0
        )
        == 0.0
    )


@pytest.mark.parametrize(
    "rounds, reps", [(0, 1), (1, 0), (-1, 1), (1, -1), (True, 1), (1, False)]
)
def test_rejects_invalid_acquisition_counts(rounds: int, reps: int) -> None:
    with pytest.raises(ValueError, match="rounds|reps"):
        program.shot_time_s(
            rounds=rounds, reps=reps, shot_lengths_us=[2.0], relax_delay_us=0.0
        )


@pytest.mark.parametrize("lengths", [[], [[1.0]], 1.0, [-1.0], [np.nan], [np.inf]])
def test_rejects_invalid_point_lengths(lengths: ArrayLike) -> None:
    with pytest.raises(ValueError, match="shot_lengths_us"):
        program.shot_time_s(
            rounds=1, reps=1, shot_lengths_us=lengths, relax_delay_us=0.0
        )


@pytest.mark.parametrize("relax_us", [-1.0, np.nan, np.inf])
def test_rejects_invalid_relaxation(relax_us: float) -> None:
    with pytest.raises(ValueError, match="relax_delay_us"):
        program.shot_time_s(
            rounds=1, reps=1, shot_lengths_us=[2.0], relax_delay_us=relax_us
        )
