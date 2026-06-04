import os

import numpy as np
import pytest
from zcu_tools.notebook.analysis.fluxdep.fitting import (
    load_database,
    search_in_database,
)
from zcu_tools.notebook.analysis.fluxdep.models import energy2transition
from zcu_tools.notebook.analysis.fluxdep.njit import (
    candidate_breakpoint_search,
    eval_dist_bounded,
    smart_fuzzy_search,
)
from zcu_tools.notebook.persistance import TransitionDict

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


# ---------------------- smart_fuzzy_search --------------------------


def test_sfs_matches_cbs_on_small_input():
    # N*K = 40 < DOWNSAMPLE_THRESHOLD (1000): both evaluate all candidates.
    A, B, C = _synth(N=10, K=4, a_true=1.1)
    d_cbs, a_cbs = candidate_breakpoint_search(A, B, C, 0.5, 3.0)
    d_sfs, a_sfs = smart_fuzzy_search(A, B, C, 0.5, 3.0)
    assert np.isclose(a_sfs, a_cbs)
    assert np.isclose(d_sfs, d_cbs)


def test_sfs_large_input_triggers_downsample_and_returns_in_range():
    # N*K = 2000 > DOWNSAMPLE_THRESHOLD (1000): fuzzy path activates.
    # Signal is only in column 0 (noise-heavy input), so we only require
    # the result to be in-range and finite — not parameter-recoverable.
    a_true = 1.5
    A, B, C = _synth(N=50, K=40, a_true=a_true)
    dist, a = smart_fuzzy_search(A, B, C, 0.5, 3.0)
    assert 0.5 <= a <= 3.0
    assert np.isfinite(dist)


def test_sfs_large_input_with_strong_signal_recovers_a():
    # Multiple columns encode a_true so the density peak is detectable.
    a_true = 1.5
    rng = np.random.default_rng(0)
    N, K = 60, 40
    B = rng.standard_normal((N, K))
    C = rng.standard_normal((N, K))
    # Make every row's A match half the columns at a_true.
    # For each row i, pick column 0 as the signal column that A matches.
    A = np.abs(a_true * B[:, 0] + C[:, 0])
    # Replicate the signal column into many columns so density search sees
    # a clear peak at a_true.
    for k in range(1, K // 2):
        B[:, k] = B[:, 0] + rng.standard_normal(N) * 1e-3
        C[:, k] = C[:, 0] + rng.standard_normal(N) * 1e-3
    dist, a = smart_fuzzy_search(A, B, C, 0.5, 3.0)
    assert abs(a - a_true) / a_true < 0.05
    assert dist < 0.1


# ------------------- search_in_database (full path) -----------------
# These exercise the real database search end-to-end. The reduced-level
# interpolation + file-load cache must NOT change the result, so we pin the
# recovered (EJ, EC, EL) to a recorded baseline (bit-identical) and check that
# transitions referencing NON-prefix levels (the level remapping's edge case)
# recover the same true parameters they were synthesised from.

_DB = "Database/simulation/fluxonium_1.h5"
_HAS_DB = os.path.exists(_DB)
_EJb, _ECb, _ELb = (3.0, 15.0), (0.2, 2.0), (0.5, 2.0)
_skip_db = pytest.mark.skipif(not _HAS_DB, reason=f"database {_DB} not present")


def _clean_observation(db_path, idx, transitions, n_fluxs=128):
    """A clean (fluxs, freqs) cloud from one DB entry's first transition line."""
    f_fluxs, f_params, f_energies = load_database(db_path)
    fluxs = np.linspace(0.05, 0.95, n_fluxs)
    energies = np.empty((n_fluxs, f_energies.shape[2]))
    for m in range(f_energies.shape[2]):
        energies[:, m] = np.interp(fluxs, f_fluxs, f_energies[idx, :, m])
    fs, _ = energy2transition(energies, transitions)
    return fluxs, fs[:, 0], tuple(float(x) for x in f_params[idx])


@_skip_db
def test_search_matches_recorded_baseline():
    # The reduced-level / load-cache optimisation must be bit-identical to the
    # pre-optimisation result for the canonical 0->1/0->2/1->2 transition set.
    transitions: TransitionDict = {"transitions": [(0, 1), (0, 2), (1, 2)]}  # type: ignore[typeddict-unknown-key]
    fluxs, freqs, _ = _clean_observation(_DB, 100, transitions)
    baseline = (4.8467594372, 0.2941233938, 1.8697251889)
    for fuzzy in (True, False):
        params, _ = search_in_database(
            fluxs, freqs, _DB, transitions, _EJb, _ECb, _ELb, fuzzy=fuzzy, plot=False
        )
        np.testing.assert_allclose(params, baseline, atol=1e-9)


@_skip_db
def test_search_non_prefix_levels_remap_correctly():
    # Transitions touching non-contiguous, non-prefix levels (0, 4, 7) stress the
    # used-levels slicing + pair remapping; the search must still recover the
    # entry it was synthesised from to good accuracy.
    transitions: TransitionDict = {"transitions": [(0, 1), (0, 4), (1, 7)]}  # type: ignore[typeddict-unknown-key]
    fluxs, freqs, true_params = _clean_observation(_DB, 100, transitions)
    params, _ = search_in_database(
        fluxs, freqs, _DB, transitions, _EJb, _ECb, _ELb, fuzzy=True, plot=False
    )
    rel = max(abs(p - t) / t for p, t in zip(params, true_params))
    assert rel < 0.05, f"non-prefix-level search drifted: {params} vs {true_params}"


@_skip_db
def test_load_database_is_cached_by_file():
    # Two loads of the same unchanged file return the cached (identical-object)
    # arrays — the GUI re-runs the search against the same DB many times.
    a = load_database(_DB)
    b = load_database(_DB)
    assert all(x is y for x, y in zip(a, b))
