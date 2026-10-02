from types import SimpleNamespace

import numpy as np
import pytest
from zcu_tools.analysis.fluxdep import search
from zcu_tools.analysis.fluxdep.search_njit import (
    candidate_breakpoint_search,
    eval_dist_bounded,
)
from zcu_tools.notebook.analysis.fluxdep import fitting
from zcu_tools.notebook.analysis.fluxdep.fitting import (
    fit_spectrum,
    search_in_database,
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


def test_search_warns_when_interrupted_after_best_so_far(monkeypatch) -> None:
    import zcu_tools.analysis.fluxdep.search_njit as njit

    monkeypatch.setattr(
        search,
        "load_database",
        lambda _path: (
            np.array([0.0, 1.0], dtype=np.float64),
            np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=np.float64),
            np.zeros((2, 2, 1), dtype=np.float64),
        ),
    )
    monkeypatch.setattr(
        njit,
        "_interp_weights",
        lambda fluxs, _f_fluxs: (
            np.zeros(len(fluxs), dtype=np.int64),
            np.zeros(len(fluxs), dtype=np.float64),
        ),
    )
    monkeypatch.setattr(njit, "_apply_interp", lambda energies, _idxs, _ws: energies)
    monkeypatch.setattr(
        njit,
        "_lower_bound_kernel",
        lambda *_args: np.array([0.0, 0.0], dtype=np.float64),
    )

    calls = 0

    def fake_search_one_entry(*_args):
        nonlocal calls
        calls += 1
        if calls == 1:
            return 0.25, 1.0
        raise KeyboardInterrupt

    monkeypatch.setattr(njit, "search_one_entry", fake_search_one_entry)

    with pytest.warns(RuntimeWarning, match="best-so-far"):
        params, fig = search_in_database(
            np.array([0.1], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
            "unused.h5",
            {},
            (0.5, 3.0),
            (0.5, 3.0),
            (0.5, 3.0),
            plot=False,
        )

    assert params == (1.0, 1.0, 1.0)
    assert fig is None


def test_fit_spectrum_reads_scipy_result_x(monkeypatch) -> None:
    def fake_least_squares(*_args, **_kwargs):
        return SimpleNamespace(x=np.array([1.1, 2.2, 3.3], dtype=np.float64))

    monkeypatch.setattr(fitting, "least_squares", fake_least_squares)

    params = fit_spectrum(
        np.array([0.0], dtype=np.float64),
        np.array([1.0], dtype=np.float64),
        (1.0, 2.0, 3.0),
        {},
        ((0.5, 2.0), (1.0, 3.0), (2.0, 4.0)),
        maxfun=1,
    )

    assert params == (1.1, 2.2, 3.3)
