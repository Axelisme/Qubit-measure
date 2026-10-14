import numpy as np
from zcu_tools.analysis.fluxdep.search_njit import (
    candidate_breakpoint_search,
    eval_dist_bounded,
)

from ._synthetic import synth_ABC as _synth

# -------------------- eval_dist_bounded -----------------------------


def test_eval_dist_bounded_prunes_when_threshold_low():
    A, B, C = _synth(a_true=1.7)
    # a=0 gives a large mean distance; threshold=0 forces immediate prune.
    assert not np.isfinite(eval_dist_bounded(A, 0.0, B, C, 0.0))


def test_eval_dist_bounded_returns_finite_at_loose_threshold():
    A, B, C = _synth(a_true=1.7)
    assert np.isfinite(eval_dist_bounded(A, 1.7, B, C, 1e9))


# ----------------- candidate_breakpoint_search ----------------------


def test_cbs_recovers_known_a():
    a_true = 1.4
    A, B, C = _synth(N=20, K=5, a_true=a_true)
    dist, a = candidate_breakpoint_search(A, B, C, 0.5, 3.0)
    assert np.isclose(a, a_true, rtol=1e-6)
    assert dist < 1e-10


def test_cbs_out_of_range_returns_midpoint_nonzero_dist():
    a_true = 5.0
    A, B, C = _synth(a_true=a_true)
    dist, a = candidate_breakpoint_search(A, B, C, 0.5, 2.0)
    assert 0.5 <= a <= 2.0
    assert dist > 0.0


def test_cbs_handles_zero_B_column():
    a_true = 1.2
    A, B, C = _synth(a_true=a_true)
    B[:, 1] = 0.0  # Should be skipped without crashing
    dist, a = candidate_breakpoint_search(A, B, C, 0.5, 3.0)
    assert np.isclose(a, a_true, rtol=1e-6)
    assert dist < 1e-10


def test_cbs_equal_bounds_returns_midpoint_with_inf_when_no_breakpoint_hits():
    # Contract: when no candidate breakpoint lies inside [a_min, a_max],
    # the function returns (inf, midpoint).
    A, B, C = _synth()
    dist, a = candidate_breakpoint_search(A, B, C, 2.0, 2.0)
    assert a == 2.0
    assert not np.isfinite(dist)


# ------------------- search_in_database -----------------------------
