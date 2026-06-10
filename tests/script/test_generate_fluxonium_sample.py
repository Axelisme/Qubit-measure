"""Tests for the fluxonium-database generator's pure pieces.

The script is not a package, so it is loaded by file path. Only the cheap, pure
parts are exercised — sampling geometry, mirror expansion, database assembly with
a stub energy function, CLI bound resolution, and overwrite protection. The real
energy computation (scqubits) is never run; ``build_database`` takes an injected
``energy_fn`` so a stub stands in.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[2] / "script" / "generate_fluxonium_sample.py"
)
_spec = importlib.util.spec_from_file_location("gen_fluxonium", _SCRIPT)
assert _spec is not None and _spec.loader is not None
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)


# --- sampling geometry -----------------------------------------------------


def test_fibonacci_lattice_unit_vectors():
    dirs = gen.fibonacci_lattice(200)
    assert dirs.shape == (200, 3)
    norms = np.linalg.norm(dirs, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)


def test_fibonacci_lattice_positive_octant():
    dirs = gen.fibonacci_lattice_positive(50)
    assert dirs.shape[0] >= 50
    assert np.all(dirs > 0)


def test_ray_intersects_box_true_and_false():
    box = ((1.0, 2.0), (1.0, 2.0), (1.0, 2.0))
    # a ray toward the box centre intersects
    centre_dir = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
    assert gen.ray_intersects_box(centre_dir, *box)
    # a ray steeply along z (x,y tiny) exits the x/y slabs before the z slab
    skew = np.array([0.01, 0.01, 1.0])
    skew = skew / np.linalg.norm(skew)
    assert not gen.ray_intersects_box(skew, *box)


def test_sample_intersecting_rays_count_and_in_octant():
    box = ((1.0, 20.0), (0.1, 4.0), (0.01, 3.0))
    params = gen.sample_intersecting_rays(*box, 64, plot=False)
    assert params.shape[0] >= 64
    assert params.shape[1] == 3
    assert np.all(params > 0)


# --- mirror expansion ------------------------------------------------------


def test_mirror_expand_doubles_flux_and_reflects():
    fluxs = np.linspace(0.0, 0.5, 6)
    energies = np.arange(2 * 6 * 3, dtype=np.float64).reshape(2, 6, 3)
    full_fluxs, full_energies = gen.mirror_expand(fluxs, energies)

    assert full_fluxs.shape == (12,)
    assert full_energies.shape == (2, 12, 3)
    # the flux axis spans [0, 1] and is symmetric about 0.5
    np.testing.assert_allclose(full_fluxs[:6], fluxs)
    np.testing.assert_allclose(full_fluxs[6:], 1.0 - fluxs[::-1])
    # the energies mirror: row n's second half is the reverse of its first half
    np.testing.assert_allclose(full_energies[0, 6:], energies[0, ::-1])


# --- database assembly (stub energy_fn — no scqubits) ----------------------


def test_build_database_serial_shape():
    params = np.array([[3.0, 1.0, 0.5], [5.0, 1.2, 0.4]])

    def stub(p):
        EJ = p[0]
        return np.full((6, 4), EJ)  # (F, L)

    energies = gen.build_database(params, energy_fn=stub, n_jobs=1, progress=False)
    assert energies.shape == (2, 6, 4)
    np.testing.assert_allclose(energies[0], 3.0)
    np.testing.assert_allclose(energies[1], 5.0)


# --- CLI bound resolution + overwrite protection ---------------------------


def test_resolve_bounds_from_preset():
    args = gen.build_parser().parse_args(["--output", "x.h5", "--preset", "all"])
    EJb, ECb, ELb = gen._resolve_bounds(args)
    assert EJb == gen.PRESETS["all"][0]


def test_resolve_bounds_per_axis_override():
    args = gen.build_parser().parse_args(
        ["--output", "x.h5", "--preset", "all", "--EJb", "2,9"]
    )
    EJb, _, _ = gen._resolve_bounds(args)
    assert EJb == (2.0, 9.0)


def test_resolve_bounds_requires_bounds():
    args = gen.build_parser().parse_args(["--output", "x.h5"])
    with pytest.raises(SystemExit):
        gen._resolve_bounds(args)


def test_parse_bound_rejects_bad():
    with pytest.raises(Exception):
        gen._parse_bound("5")  # not 'min,max'
    with pytest.raises(Exception):
        gen._parse_bound("9,2")  # min >= max


def test_main_refuses_existing_output(tmp_path):
    existing = tmp_path / "db.h5"
    existing.write_bytes(b"x")
    with pytest.raises(SystemExit):
        gen.main(["--output", str(existing), "--preset", "all", "--num-samples", "4"])


def test_main_dry_run_writes_dryrun_file(tmp_path):
    out = tmp_path / "db.h5"
    rc = gen.main(
        [
            "--output",
            str(out),
            "--preset",
            "all",
            "--num-samples",
            "4",
            "--num-flux",
            "5",
            "--evals-count",
            "3",
            "--dry-run",
        ]
    )
    assert rc == 0
    # dry-run writes to the *_dryrun.h5 sibling, never the given path
    assert not out.exists()
    dryrun = tmp_path / "db_dryrun.h5"
    assert dryrun.exists()

    import h5py

    with h5py.File(dryrun, "r") as f:
        fluxs = np.asarray(f["fluxs"])
        params = np.asarray(f["params"])
        energies = np.asarray(f["energies"])
    # mirrored flux: 2 * num_flux
    assert fluxs.shape == (10,)
    assert params.shape[1] == 3
    assert energies.shape[1] == 10  # mirrored flux axis
    assert energies.shape[2] == 3  # evals_count
