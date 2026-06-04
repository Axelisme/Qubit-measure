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
    # Both search paths recover the recorded result for the canonical
    # 0->1/0->2/1->2 transition set on a clean (noiseless) cloud — where the exact
    # (default, LB-pruned) and fuzzy heuristic paths agree. This pins the value
    # across the load-cache / reduced-level / LB-prune optimisations.
    transitions: TransitionDict = {"transitions": [(0, 1), (0, 2), (1, 2)]}  # type: ignore[typeddict-unknown-key]
    fluxs, freqs, _ = _clean_observation(_DB, 100, transitions)
    baseline = (4.8467594372, 0.2941233938, 1.8697251889)
    for fuzzy in (False, True):
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


# --- exact lower-bound prune (the default exact path) ----------------------
# The prune searches entries in increasing-lower-bound order and stops once the
# bound exceeds the incumbent. Skipped entries provably cannot win, so the result
# must be IDENTICAL to a full exact scan over every entry. We verify this directly
# against an unpruned full exact search, on noisy clouds where many entries are
# near-degenerate (the case where the prune searches the most entries).


def _full_exact_scan(db_path, fluxs, freqs, transitions, EJb, ECb, ELb):
    """A reference full exact scan: search EVERY feasible entry (no prune)."""
    from zcu_tools.notebook.analysis.fluxdep.models import compile_transitions
    from zcu_tools.notebook.analysis.fluxdep.njit import (
        _apply_interp,
        _interp_weights,
        search_one_entry,
    )

    f_fluxs, f_params, f_energies = load_database(db_path)
    M = f_energies.shape[2]
    pairs, coeffs, offsets = compile_transitions(transitions, M)
    used = np.unique(pairs.reshape(-1)) if pairs.size else np.arange(M)
    pos = np.full(M, -1, dtype=np.int64)
    pos[used] = np.arange(used.shape[0])
    pr = pos[pairs].astype(np.int32)
    idxs, ws = _interp_weights(
        np.ascontiguousarray(np.mod(fluxs, 1.0)), np.ascontiguousarray(f_fluxs)
    )
    sf = np.ascontiguousarray(
        _apply_interp(np.ascontiguousarray(f_energies[:, :, used]), idxs, ws)
    )
    freqs_c = np.ascontiguousarray(freqs, dtype=np.float64)

    best_d, best_params = np.inf, None
    for i in range(f_params.shape[0]):
        p0, p1, p2 = f_params[i]
        am = max(EJb[0] / p0, ECb[0] / p1, ELb[0] / p2)
        aM = min(EJb[1] / p0, ECb[1] / p1, ELb[1] / p2)
        if am > aM:
            continue
        d, a = search_one_entry(sf[i], pr, coeffs, offsets, freqs_c, am, aM)
        if d < best_d:
            best_d, best_params = d, tuple(f_params[i] * a)
    return best_params


@_skip_db
@pytest.mark.parametrize("seed", [0, 1, 5])
def test_prune_is_identical_to_full_exact_scan(seed):
    # Build a noisy cloud (the prune searches more entries here) and assert the
    # default (pruned exact) result matches a full unpruned exact scan exactly.
    transitions: TransitionDict = {"transitions": [(0, 1), (0, 2), (1, 2)]}  # type: ignore[typeddict-unknown-key]
    f_fluxs, f_params, f_energies = load_database(_DB)
    rng = np.random.RandomState(seed)
    src = rng.randint(0, f_params.shape[0])
    fs, _ = energy2transition(f_energies[src], transitions)  # (n_flux, K)
    fl, fr = [], []
    for _ in range(400):
        fi = rng.randint(0, len(f_fluxs))
        ki = rng.randint(0, fs.shape[1])
        fl.append(f_fluxs[fi])
        fr.append(fs[fi, ki])
    fluxs = np.asarray(fl)
    freqs = np.asarray(fr) + rng.standard_normal(len(fr)) * 0.02  # noise

    pruned, _ = search_in_database(
        fluxs, freqs, _DB, transitions, _EJb, _ECb, _ELb, fuzzy=False, plot=False
    )
    full = _full_exact_scan(_DB, fluxs, freqs, transitions, _EJb, _ECb, _ELb)
    assert full is not None
    np.testing.assert_allclose(pruned, full, atol=1e-9)


@_skip_db
def test_entry_lower_bound_is_a_valid_floor():
    # entry_lower_bound(...) must never exceed the entry's true best distance
    # (else the prune could wrongly skip the winner).
    from zcu_tools.notebook.analysis.fluxdep.models import compile_transitions
    from zcu_tools.notebook.analysis.fluxdep.njit import (
        _apply_interp,
        _interp_weights,
        candidate_breakpoint_search,
        energy2linearform_nb,
        entry_lower_bound,
    )

    transitions: TransitionDict = {"transitions": [(0, 1), (0, 2), (1, 2)]}  # type: ignore[typeddict-unknown-key]
    fluxs, freqs, _ = _clean_observation(_DB, 100, transitions)
    f_fluxs, f_params, f_energies = load_database(_DB)
    M = f_energies.shape[2]
    pairs, coeffs, offsets = compile_transitions(transitions, M)
    used = np.unique(pairs.reshape(-1))
    pos = np.full(M, -1, dtype=np.int64)
    pos[used] = np.arange(used.shape[0])
    pr = pos[pairs].astype(np.int32)
    idxs, ws = _interp_weights(
        np.ascontiguousarray(np.mod(fluxs, 1.0)), np.ascontiguousarray(f_fluxs)
    )
    sf = np.ascontiguousarray(
        _apply_interp(np.ascontiguousarray(f_energies[:, :, used]), idxs, ws)
    )
    freqs_c = np.ascontiguousarray(freqs)

    for i in range(0, f_params.shape[0], 250):  # sample entries
        p0, p1, p2 = f_params[i]
        am = max(_EJb[0] / p0, _ECb[0] / p1, _ELb[0] / p2)
        aM = min(_EJb[1] / p0, _ECb[1] / p1, _ELb[1] / p2)
        if am > aM:
            continue
        Bs, Cs = energy2linearform_nb(sf[i], pr, coeffs, offsets)
        lb = entry_lower_bound(freqs_c, Bs, Cs, am, aM)
        d, _ = candidate_breakpoint_search(freqs_c, Bs, Cs, am, aM)
        assert lb <= d + 1e-9, (
            f"lower bound {lb} exceeds true distance {d} at entry {i}"
        )
