"""Physical seconds for one round of a simulation request."""

from fractions import Fraction

import numpy as np
import pytest
from zcu_tools.program.v2.simulation.round_time import request_round_time_s
from zcu_tools.simulate.boundary import AcquireRequest, TimeBase, TimelineTemplate


@pytest.mark.parametrize("reps", [3, 10**12])
@pytest.mark.parametrize("relax_delay", [1, np.int64(1)])
def test_integer_request_round_seconds_preserve_exact_long_totals(
    reps: int, relax_delay
) -> None:
    # Each point is within the ns bound; the whole round may exceed int64.
    request = AcquireRequest(
        "decimated",
        (reps, 3),
        0,
        (),
        1,
        0,
        "test",
        time_base=TimeBase(Fraction(3, 2)),
        point_dims=(3,),
        point_names=("scan",),
        reps=reps,
        templates=(
            TimelineTemplate(
                np.array([0, 2]), (), (), np.array([2, 5], dtype=np.int64)
            ),
            TimelineTemplate(
                np.array([1]), (), (), np.array([4_000_000_000_000], dtype=np.int64)
            ),
        ),
        relax_delay_ticks=relax_delay,
    )
    expected = float(Fraction(reps * 4_000_000_000_010, 1_500_000))
    assert request_round_time_s(request).hex() == expected.hex()
