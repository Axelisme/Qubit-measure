"""Generate a precomputed fluxonium energy database for fluxdep database search.

For each sampled ``(EJ, EC, EL)`` point this computes the lowest ``evals_count``
energy levels versus external flux (0 → 0.5, then mirrored to 0 → 1 since the
fluxonium spectrum is symmetric about half-flux) and stores them. The result
feeds ``zcu_tools.notebook.analysis.fluxdep.search_in_database`` (and the
fluxdep-gui v2 search): each row is a candidate, and the search scales it to
cover a continuous neighbourhood of parameter space.

Parameters are sampled as rays from the origin through the ``EJb × ECb × ELb``
bounding box (Fibonacci-lattice directions, filtered to those that intersect the
box). Rays — rather than a uniform grid — match the search's ``params[i] * scale``
design: one direction covers a continuous line of parameters via the scale.

Usage (the energy computation is SLOW — minutes to hours for a real run):

    # a real "all"-range database with 10k samples (serial — scqubits already
    # multithreads each diagonalisation, so --n-jobs > 1 usually doesn't help)
    .venv/bin/python script/generate_fluxonium_sample.py \
        --output Database/simulation/fluxonium_all.h5 \
        --preset all --num-samples 10000

    # a tiny dry run (random energies, no scqubits) to sanity-check plumbing
    .venv/bin/python script/generate_fluxonium_sample.py \
        --output /tmp/db.h5 --num-samples 8 --dry-run

The output path must not already exist unless ``--overwrite`` is given (so a real
run never clobbers an existing database by accident). ``--dry-run`` always writes
to a ``*_dryrun.h5`` sibling instead of the given path.

WARNING: a real run takes a long time. Back up any existing database first
(``cp Database/simulation/fluxonium_all.h5 ~/backup/``) before regenerating.
"""

from __future__ import annotations

import argparse
import os

import h5py as h5
import numpy as np
from numpy.typing import NDArray
from typing_extensions import Callable, Optional

# Preset (EJb, ECb, ELb) bounding boxes — the notebook's named ranges (GHz).
PRESETS: dict[
    str, tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
] = {
    "normal": ((3.0, 15.0), (0.2, 2.0), (0.5, 2.0)),
    "integer": ((2.0, 6.0), (0.8, 2.0), (0.01, 0.2)),
    "all": ((1.0, 20.0), (0.1, 4.0), (0.01, 3.0)),
}

Bound = tuple[float, float]
EnergyFn = Callable[[tuple[float, float, float]], NDArray[np.float64]]


# ---------------------------------------------------------------------------
# Pure sampling geometry (no scqubits, no IO — unit-testable)
# ---------------------------------------------------------------------------


def fibonacci_lattice(K: int) -> NDArray[np.float64]:
    """K approximately-uniform unit directions on the sphere (Fibonacci lattice).

    Returns an array of shape ``(K, 3)`` of unit vectors.
    """
    phi = (1 + np.sqrt(5)) / 2
    indices = np.arange(K)
    z = 1 - (2 * indices + 1) / K
    theta = 2 * np.pi * indices / phi
    r = np.sqrt(1 - z**2)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.stack([x, y, z], axis=-1)


def fibonacci_lattice_positive(K: int) -> NDArray[np.float64]:
    """At least ``K`` Fibonacci-lattice directions with all components > 0.

    Oversamples (8·K) and filters to the positive octant; grows K if too few
    survive (parameter bounds are all positive, so only this octant matters).
    """
    while True:
        directions = fibonacci_lattice(8 * K)
        x, y, z = directions.T
        mask = (x > 0) & (y > 0) & (z > 0)
        valid = directions[mask]
        if valid.shape[0] >= K:
            return valid
        K = int(K * 1.1)


def ray_intersects_box(
    direction: NDArray[np.float64],
    x_range: Bound,
    y_range: Bound,
    z_range: Bound,
) -> bool:
    """Whether a ray from the origin along ``direction`` enters the box.

    Slab method: the ray (t ≥ 0) intersects iff the per-axis entry/exit interval
    overlaps. ``direction`` components are assumed > 0 (positive octant).
    """
    t_min = 0.0
    t_max = np.inf
    for d, rng in zip(direction, (x_range, y_range, z_range)):
        t1 = rng[0] / d
        t2 = rng[1] / d
        t_min = max(t_min, t1)
        t_max = min(t_max, t2)
    return t_min <= t_max


def sample_intersecting_rays(
    x_range: Bound,
    y_range: Bound,
    z_range: Bound,
    n_samples: int,
    *,
    max_attempts: int = 10,
    plot: bool = False,
) -> NDArray[np.float64]:
    """Sample ``n_samples`` ray directions (scaled by the box) that hit the box.

    Each returned row is a ``(EJ, EC, EL)`` parameter point on a ray through the
    bounding box. Grows the candidate count until enough rays intersect; raises
    if it cannot reach ``n_samples`` within ``max_attempts``. With ``plot=True``
    shows a 3D debug scatter of the accepted directions vs the box corners.
    """
    K = n_samples
    intersecting: list[NDArray[np.float64]] = []
    for attempt in range(max_attempts):
        directions = fibonacci_lattice_positive(K)
        directions[:, 0] *= x_range[1]
        directions[:, 1] *= y_range[1]
        directions[:, 2] *= z_range[1]

        intersecting = [
            d for d in directions if ray_intersects_box(d, x_range, y_range, z_range)
        ]
        if len(intersecting) >= n_samples:
            params = np.array(intersecting)
            if plot:
                _plot_directions(params, x_range, y_range, z_range)
            return params

        orig_K = K
        K = int(K * min(max(n_samples / max(len(intersecting), 1), 1.01), 100))
        print(
            f"Attempt {attempt + 1}: {len(intersecting)} intersecting rays "
            f"< {n_samples}; increasing candidates {orig_K} -> {K}."
        )

    raise ValueError(
        f"Unable to find {n_samples} intersecting rays after {max_attempts} "
        f"attempts (found {len(intersecting)})."
    )


def mirror_expand(
    fluxs: NDArray[np.float64], energies: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Mirror a 0→0.5 half-period sweep to a full 0→1 sweep (fluxonium symmetry).

    ``fluxs`` is ``(F,)`` over [0, 0.5]; ``energies`` is ``(N, F, L)``. The
    fluxonium spectrum is symmetric about flux = 0.5, so each is reflected and
    concatenated to span [0, 1] with ``2F`` flux points.
    """
    full_fluxs = np.concatenate([fluxs, 1.0 - fluxs[::-1]])
    full_energies = np.concatenate([energies, energies[:, ::-1, :]], axis=1)
    return full_fluxs, full_energies


def _plot_directions(params, x_range, y_range, z_range) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(*np.array(params).T)
    for x in x_range:
        for y in y_range:
            for z in z_range:
                ax.scatter(x, y, z, color="r")
    ax.scatter(0, 0, 0, color="b")
    plt.show()


# ---------------------------------------------------------------------------
# Energy computation + database assembly
# ---------------------------------------------------------------------------


def _make_energy_fn(
    fluxs: NDArray[np.float64], cutoff: int, evals_count: int
) -> EnergyFn:
    """A per-parameter energy function bound to the flux grid + truncation."""
    from zcu_tools.simulate.fluxonium import calculate_energy_vs_flux

    def _fn(params: tuple[float, float, float]) -> NDArray[np.float64]:
        return calculate_energy_vs_flux(params, fluxs, cutoff, evals_count)[1]

    return _fn


def build_database(
    params: NDArray[np.float64],
    *,
    energy_fn: EnergyFn,
    n_jobs: int = 1,
    progress: bool = True,
) -> NDArray[np.float64]:
    """Compute the ``(N, F, L)`` energy table for every parameter row.

    ``energy_fn`` maps one ``(EJ, EC, EL)`` to its ``(F, L)`` energies (it closes
    over the flux grid + truncation; injected so tests can pass a cheap stub
    instead of scqubits). With ``n_jobs != 1`` the rows run in parallel via joblib
    — but scqubits already multithreads each diagonalisation, so row-level workers
    usually just oversubscribe the cores; the default ``n_jobs=1`` (serial) is the
    right choice unless a parallel win has been measured.
    """
    rows = [tuple(float(v) for v in p) for p in params]

    if n_jobs == 1:
        iterator = rows
        if progress:
            from tqdm.auto import tqdm

            iterator = tqdm(rows, desc="Calculating")
        energies = [energy_fn(p) for p in iterator]  # type: ignore[arg-type]
    else:
        from joblib import Parallel, delayed

        energies = Parallel(n_jobs=n_jobs)(  # type: ignore[assignment]
            delayed(energy_fn)(p) for p in rows
        )

    return np.asarray(energies, dtype=np.float64)


def dump_data(
    filepath: str,
    fluxs: NDArray[np.float64],
    params: NDArray[np.float64],
    energies: NDArray[np.float64],
    Ebounds: NDArray[np.float64],
) -> None:
    """Write the database datasets (fluxs / params / energies / Ebounds)."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with h5.File(filepath, "w") as f:
        f.create_dataset("Ebounds", data=Ebounds)
        f.create_dataset("fluxs", data=fluxs)
        f.create_dataset("params", data=params)
        f.create_dataset("energies", data=energies)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_bound(text: str) -> Bound:
    parts = text.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"bound must be 'min,max', got {text!r}")
    lo, hi = float(parts[0]), float(parts[1])
    if lo >= hi:
        raise argparse.ArgumentTypeError(f"bound min must be < max, got {text!r}")
    return (lo, hi)


def _resolve_bounds(args: argparse.Namespace) -> tuple[Bound, Bound, Bound]:
    """Bounds come from --preset, then any explicit --EJb/--ECb/--ELb override."""
    if args.preset is not None:
        EJb, ECb, ELb = PRESETS[args.preset]
    else:
        EJb = ECb = ELb = None  # require explicit bounds below
    EJb = args.EJb if args.EJb is not None else EJb
    ECb = args.ECb if args.ECb is not None else ECb
    ELb = args.ELb if args.ELb is not None else ELb
    if EJb is None or ECb is None or ELb is None:
        raise SystemExit("bounds required: pass --preset, or all of --EJb/--ECb/--ELb")
    return EJb, ECb, ELb


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate a fluxonium energy database for fluxdep search.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--output", required=True, help="Output .h5 path")
    p.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default=None,
        help="Named (EJb, ECb, ELb) bounding box; overridable per-axis below",
    )
    p.add_argument("--EJb", type=_parse_bound, default=None, help="EJ bound 'min,max'")
    p.add_argument("--ECb", type=_parse_bound, default=None, help="EC bound 'min,max'")
    p.add_argument("--ELb", type=_parse_bound, default=None, help="EL bound 'min,max'")
    p.add_argument("--num-samples", type=int, default=10000, help="Parameter samples")
    p.add_argument("--cutoff", type=int, default=40, help="Fluxonium charge cutoff")
    p.add_argument(
        "--evals-count", type=int, default=15, help="Energy levels per point"
    )
    p.add_argument(
        "--num-flux", type=int, default=120, help="Flux points over [0, 0.5] (mirrored)"
    )
    p.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Joblib workers over parameter rows (default 1 = serial). scqubits "
        "already multithreads each diagonalisation, so spawning workers usually "
        "just oversubscribes the cores; leave at 1 unless you've measured a win.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Random energies (skip scqubits); writes to *_dryrun.h5",
    )
    p.add_argument(
        "--plot", action="store_true", help="Show a 3D scatter of sampled directions"
    )
    p.add_argument(
        "--overwrite", action="store_true", help="Allow overwriting an existing output"
    )
    p.add_argument("--seed", type=int, default=0, help="RNG seed (dry-run energies)")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    EJb, ECb, ELb = _resolve_bounds(args)

    output = args.output
    if args.dry_run:
        output = output.replace(".h5", "_dryrun.h5")
    if os.path.exists(output) and not args.overwrite:
        raise SystemExit(
            f"output already exists: {output} (pass --overwrite to replace it). "
            "Back up an existing database before regenerating."
        )

    fluxs = np.linspace(0.0, 0.5, args.num_flux)

    params = sample_intersecting_rays(EJb, ECb, ELb, args.num_samples, plot=args.plot)
    print(f"Generated {len(params)} samples.")

    if args.dry_run:
        rng = np.random.RandomState(args.seed)
        energies = np.stack(
            [rng.randn(len(fluxs), args.evals_count) for _ in range(len(params))]
        )
    else:
        import scqubits.settings as scq_settings

        scq_settings.PROGRESSBAR_DISABLED = True
        try:
            energy_fn = _make_energy_fn(fluxs, args.cutoff, args.evals_count)
            energies = build_database(params, energy_fn=energy_fn, n_jobs=args.n_jobs)
        finally:
            scq_settings.PROGRESSBAR_DISABLED = False

    full_fluxs, full_energies = mirror_expand(fluxs, energies)
    Ebounds = np.array((EJb, ECb, ELb))

    dump_data(output, full_fluxs, params, full_energies, Ebounds)
    print(
        f"Wrote {len(params)} params x {len(full_fluxs)} fluxs x "
        f"{args.evals_count} levels -> {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
