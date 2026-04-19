"""Performance benchmark for `zcu_tools.notebook.analysis.fluxdep.fitting`.

Not a pytest test — run manually:

    .venv/bin/python tests/notebook/analysis/fluxdep/bench_fitting.py --quick
    .venv/bin/python tests/notebook/analysis/fluxdep/bench_fitting.py --n-fluxs 128
    .venv/bin/python tests/notebook/analysis/fluxdep/bench_fitting.py \
        --db Database/simulation/fluxonium_1.h5 \
        --out bench_results/baseline.json

Use `--compare <file.json>` to print side-by-side percentage diffs vs a
previously-saved run.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from statistics import median

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))
_LIB = os.path.join(_REPO_ROOT, "lib")
if _LIB not in sys.path:
    sys.path.insert(0, _LIB)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from zcu_tools.notebook.analysis.fluxdep.fitting import (  # noqa: E402
    candidate_breakpoint_search,
    eval_dist_bounded,
    fit_spectrum,
    search_in_database,
    smart_fuzzy_search,
)
from zcu_tools.simulate.fluxonium import calculate_energy_vs_flux  # noqa: E402

from tests.notebook.analysis.fluxdep._synthetic import synth_ABC  # noqa: E402


def _timed(fn, *, repeat=5):
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return {"median": median(times), "min": min(times), "n": repeat}


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=_REPO_ROOT
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def warm_up():
    """Trigger JIT compilation so it doesn't skew timing."""
    A, B, C = synth_ABC(N=10, K=4, a_true=1.5)
    eval_dist_bounded(A, 1.5, B, C, 1.0)
    candidate_breakpoint_search(A, B, C, 0.5, 3.0)
    smart_fuzzy_search(A, B, C, 0.5, 3.0)


def micro_benches(quick: bool) -> dict:
    shapes = [(240, 6), (240, 12), (240, 40)]
    if quick:
        shapes = [(60, 6)]

    repeat = 3 if quick else 11
    out: dict = {}

    for N, K in shapes:
        A, B, C = synth_ABC(N=N, K=K, a_true=1.5, seed=1)
        key = f"N{N}_K{K}"
        d_true = eval_dist_bounded(A, 1.5, B, C, 1e9)

        out[f"eval_dist_bounded_loose/{key}"] = _timed(
            lambda: eval_dist_bounded(A, 1.5, B, C, 1e9), repeat=repeat
        )
        out[f"eval_dist_bounded_tight/{key}"] = _timed(
            lambda: eval_dist_bounded(A, 1.5, B, C, d_true + 1e-9), repeat=repeat
        )
        out[f"candidate_breakpoint_search/{key}"] = _timed(
            lambda: candidate_breakpoint_search(A, B, C, 0.5, 3.0), repeat=repeat
        )
        out[f"smart_fuzzy_search/{key}"] = _timed(
            lambda: smart_fuzzy_search(A, B, C, 0.5, 3.0), repeat=repeat
        )

        # Correctness assert for regressions.
        d_cbs, a_cbs = candidate_breakpoint_search(A, B, C, 0.5, 3.0)
        assert abs(a_cbs - 1.5) < 1e-6, f"cbs regression at {key}: a={a_cbs}"
        assert d_cbs < 1e-8, f"cbs regression at {key}: d={d_cbs}"

    return out


def _synth_observation(db_path: str, idx: int, n_fluxs: int, transitions):
    """Generate a synthetic (fluxs, freqs) observation using one DB entry."""
    import h5py

    with h5py.File(db_path, "r") as f:
        f_fluxs = f["fluxs"][:]
        f_params = f["params"][:]
        f_energies = f["energies"][:]

    fluxs = np.linspace(0.05, 0.95, n_fluxs)
    params = tuple(float(x) for x in f_params[idx])
    # Interp energies at chosen fluxs.
    energies = np.empty((n_fluxs, f_energies.shape[2]))
    for m in range(f_energies.shape[2]):
        energies[:, m] = np.interp(fluxs, f_fluxs, f_energies[idx, :, m])

    from zcu_tools.notebook.analysis.fluxdep.models import energy2transition

    fs, _ = energy2transition(energies, transitions)
    # Pick one observed transition per flux (column 0) — clean data.
    freqs = fs[:, 0]
    return fluxs, freqs, params


def macro_benches(db_path: str, quick: bool, n_fluxs: int | None = None) -> dict:
    if not os.path.exists(db_path):
        print(f"[skip macro] DB not found: {db_path}")
        return {}

    transitions = {"transitions": [(0, 1), (0, 2), (1, 2)]}
    if n_fluxs is None:
        n_fluxs = 30 if quick else 128
    fluxs, freqs, true_params = _synth_observation(
        db_path, idx=100, n_fluxs=n_fluxs, transitions=transitions
    )

    EJb = (3.0, 15.0)
    ECb = (0.2, 2.0)
    ELb = (0.5, 2.0)

    out: dict = {"true_params": true_params}

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cases = [
        ("search_njobs1_fuzzy", dict(n_jobs=1, fuzzy=True)),
        ("search_njobsN_fuzzy", dict(n_jobs=-1, fuzzy=True)),
        ("search_njobs1_exact", dict(n_jobs=1, fuzzy=False)),
        ("search_njobsN_exact", dict(n_jobs=-1, fuzzy=False)),
    ]
    if quick:
        cases = cases[:2]

    for name, kwargs in cases:
        t0 = time.perf_counter()
        best_params, fig = search_in_database(
            fluxs, freqs, db_path, transitions, EJb, ECb, ELb, **kwargs
        )
        dt = time.perf_counter() - t0
        plt.close(fig)
        err = max(abs(bp - tp) / tp for bp, tp in zip(best_params, true_params))
        out[name] = {"time": dt, "best": list(best_params), "max_rel_err": err}

    # fit_spectrum benchmark.
    if not quick:
        init = tuple(p * 1.02 for p in true_params)
        t0 = time.perf_counter()
        fit = fit_spectrum(
            fluxs,
            freqs,
            init,
            transitions,
            (EJb, ECb, ELb),
            maxfun=80,
        )
        dt = time.perf_counter() - t0
        err = max(abs(f - tp) / tp for f, tp in zip(fit, true_params))
        out["fit_spectrum"] = {"time": dt, "best": list(fit), "max_rel_err": err}

    return out


def run(args) -> dict:
    warm_up()
    results = {
        "git_sha": _git_sha(),
        "quick": args.quick,
        "micro": micro_benches(args.quick),
    }
    if args.db:
        results["macro"] = macro_benches(args.db, args.quick, args.n_fluxs)
    return results


def print_summary(r: dict) -> None:
    print(f"git: {r.get('git_sha')}  quick={r.get('quick')}")
    print("-- micro (median seconds) --")
    for k, v in sorted(r.get("micro", {}).items()):
        print(f"  {k:45s} {v['median'] * 1000:8.3f} ms  (min {v['min'] * 1000:7.3f})")
    if "macro" in r:
        print("-- macro --")
        for k, v in r["macro"].items():
            if isinstance(v, dict) and "time" in v:
                print(f"  {k:25s} {v['time']:7.2f} s  err={v['max_rel_err']:.2e}")
            else:
                print(f"  {k}: {v}")


def print_compare(cur: dict, base: dict) -> None:
    print("-- compare vs baseline (neg = faster) --")
    for section in ("micro", "macro"):
        for k, v in cur.get(section, {}).items():
            if not isinstance(v, dict):
                continue
            bv = base.get(section, {}).get(k)
            if not isinstance(bv, dict):
                continue
            cur_t = v.get("median", v.get("time"))
            base_t = bv.get("median", bv.get("time"))
            if cur_t is None or base_t is None:
                continue
            pct = (cur_t - base_t) / base_t * 100.0
            print(
                f"  {section}/{k:40s} {base_t * 1000:9.2f} -> {cur_t * 1000:9.2f} ms  ({pct:+6.1f}%)"
            )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="Small shapes / fewer reps")
    p.add_argument("--db", type=str, default="Database/simulation/fluxonium_1.h5")
    p.add_argument(
        "--n-fluxs",
        type=int,
        default=128,
        help="Number of observed flux points for macro benchmark (default: 128)",
    )
    p.add_argument("--out", type=str, default=None, help="Write JSON to path")
    p.add_argument("--compare", type=str, default=None, help="Compare against JSON")
    args = p.parse_args()

    r = run(args)
    print_summary(r)

    if args.compare:
        with open(args.compare) as f:
            base = json.load(f)
        print_compare(r, base)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(r, f, indent=2)
        print(f"[wrote] {args.out}")


if __name__ == "__main__":
    main()
