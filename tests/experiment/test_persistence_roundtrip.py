"""Regression test for the ADR-0027 PersistableExperiment save/load round-trip.

Exercises the *base-level* persistence mechanism end-to-end against a real
on-disk Labber HDF5 file (no labber_io mocking): a minimal in-test
``PersistableExperiment`` subclass declares an ``AXES_SPEC``, we build a Result
with known numpy arrays, ``save()`` it to ``tmp_path``, ``load()`` it back, and
assert axis values, the complex z array, the cfg snapshot, and the inner-first
shape invariant all round-trip.

ADR-0027 invariants under test:
- axes are inner-first: ``z.shape == tuple(len(ax) for ax in reversed(axes))``;
- ``load`` is the exact inverse of ``save`` (zero caller-side transpose);
- per-axis ``scale`` is applied on save (disk = memory * scale) and undone on
  load (memory = disk / scale);
- Fast-Fail: a z whose shape disagrees with the declared axis lengths makes
  ``save`` raise rather than silently transpose.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import pytest
from zcu_tools.experiment.axes_spec import (
    MHZ_TO_HZ,
    US_TO_S,
    AxesSpec,
    Axis,
    ZSpec,
)
from zcu_tools.experiment.base import PersistableExperiment
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.utils.datasaver import LabberData, save_labber_data

# --------------------------------------------------------------------------- #
# Minimal cfg / Result / experiment fixtures matching the AxesSpec contract.
# --------------------------------------------------------------------------- #


class _TinyCfg(ExpCfgModel):
    """Minimal cfg the spec references as ``cfg_type`` (round-trips via comment)."""

    name: str = "roundtrip"
    reps: int = 100


@dataclass(frozen=True)
class _Result2D:
    """Frozen Result for the 2-axis spec: freq axis (MHz), length axis (us)."""

    freqs: np.ndarray  # inner axis, in MHz (memory units)
    lengths: np.ndarray  # outer axis, in us (memory units)
    signals: np.ndarray  # complex z, shape (Nlength, Nfreq) -> inner-first
    cfg_snapshot: _TinyCfg | None


@dataclass(frozen=True)
class _Result1D:
    """Frozen Result for the 1-axis spec (proves N-D generality at N=1)."""

    freqs: np.ndarray  # inner axis, in MHz
    signals: np.ndarray  # complex z, shape (Nfreq,)
    cfg_snapshot: _TinyCfg | None


class _Exp2D(PersistableExperiment[_Result2D, _TinyCfg]):
    AXES_SPEC: ClassVar[AxesSpec[Any, Any] | None] = AxesSpec(
        axes=(
            Axis("freqs", "Frequency", "Hz", MHZ_TO_HZ, np.float64),  # inner
            Axis("lengths", "Pulse length", "s", US_TO_S, np.float64),  # outer
        ),
        z=ZSpec("signals", "S21", "", np.complex128),
        result_type=_Result2D,
        cfg_type=_TinyCfg,
        tag="test/roundtrip2d",
    )


class _Exp1D(PersistableExperiment[_Result1D, _TinyCfg]):
    AXES_SPEC: ClassVar[AxesSpec[Any, Any] | None] = AxesSpec(
        axes=(Axis("freqs", "Frequency", "Hz", MHZ_TO_HZ, np.float64),),
        z=ZSpec("signals", "S21", "", np.complex128),
        result_type=_Result1D,
        cfg_type=_TinyCfg,
        tag="test/roundtrip1d",
    )


class _Exp1DReal(PersistableExperiment[_Result1D, _TinyCfg]):
    AXES_SPEC: ClassVar[AxesSpec[Any, Any] | None] = AxesSpec(
        axes=(Axis("freqs", "Frequency", "Hz", MHZ_TO_HZ, np.float64),),
        z=ZSpec("signals", "Population", "a.u.", np.float64),
        result_type=_Result1D,
        cfg_type=_TinyCfg,
        tag="test/roundtrip1d-real",
    )


def _saved_path(tmp_path: Any, base: str) -> str:
    """save() writes the exact formatted path; reservation belongs to callers."""
    return os.path.join(str(tmp_path), f"{base}.hdf5")


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


def test_roundtrip_2d_inner_first(tmp_path: Any) -> None:
    """2-D round-trip: axes (scaled), complex z, cfg, and shape invariant."""
    freqs = np.linspace(4000.0, 5000.0, 7)  # MHz, inner axis (Nx = 7)
    lengths = np.linspace(0.1, 2.0, 4)  # us, outer axis (Ny = 4)
    # inner-first: z.shape == (len(lengths), len(freqs)) == (Ny, Nx)
    rng = np.random.default_rng(0)
    signals = (rng.standard_normal((4, 7)) + 1j * rng.standard_normal((4, 7))).astype(
        np.complex128
    )

    spec = _Exp2D.AXES_SPEC
    assert spec is not None
    # the inner-first invariant the data itself must obey
    assert signals.shape == (len(lengths), len(freqs))

    cfg = _TinyCfg(name="twotone-len", reps=512)
    result = _Result2D(freqs=freqs, lengths=lengths, signals=signals, cfg_snapshot=cfg)

    exp = _Exp2D()
    base = os.path.join(str(tmp_path), "scan2d")
    exp.save(base, result)

    path = _saved_path(tmp_path, "scan2d")
    assert os.path.exists(path)

    loaded = exp.load(path)

    # axis values round-trip within scale tolerance (memory units restored)
    np.testing.assert_allclose(loaded.freqs, freqs, rtol=0, atol=1e-6)
    np.testing.assert_allclose(loaded.lengths, lengths, rtol=0, atol=1e-9)
    assert loaded.freqs.dtype == np.float64
    assert loaded.lengths.dtype == np.float64

    # z array matches exactly (dtype + shape + values), zero transpose
    assert loaded.signals.shape == signals.shape
    assert loaded.signals.dtype == np.complex128
    np.testing.assert_allclose(loaded.signals, signals, rtol=0, atol=0)

    # inner-first shape invariant holds on the loaded data
    inner_first = tuple(len(getattr(loaded, ax.field_name)) for ax in spec.axes)
    assert loaded.signals.shape == inner_first[::-1]

    # cfg snapshot round-trips through the comment channel
    assert loaded.cfg_snapshot is not None
    assert loaded.cfg_snapshot.name == "twotone-len"
    assert loaded.cfg_snapshot.reps == 512

    # last_result bookkeeping (@record_result on load)
    assert exp.last_result is loaded


def test_save_applies_scale_on_disk(tmp_path: Any) -> None:
    """Disk values carry the SI scale (Hz / s), proving load divides it back."""
    from zcu_tools.utils.datasaver import load_labber_data

    freqs = np.array([4000.0, 4500.0, 5000.0])  # MHz
    lengths = np.array([1.0, 2.0])  # us
    signals = np.ones((2, 3), dtype=np.complex128)
    cfg = _TinyCfg()
    result = _Result2D(freqs, lengths, signals, cfg)

    exp = _Exp2D()
    exp.save(os.path.join(str(tmp_path), "scaled"), result)
    ld = load_labber_data(_saved_path(tmp_path, "scaled"))

    # on disk the inner axis is in Hz (MHz * 1e6), outer in s (us * 1e-6)
    np.testing.assert_allclose(ld.axes[0].values, freqs * MHZ_TO_HZ, rtol=0, atol=1e-3)
    np.testing.assert_allclose(ld.axes[1].values, lengths * US_TO_S, rtol=0, atol=1e-15)


def test_roundtrip_1d(tmp_path: Any) -> None:
    """1-D round-trip proves the same mechanism at N=1 (z.shape == (Nx,))."""
    freqs = np.linspace(4000.0, 6000.0, 11)  # MHz
    rng = np.random.default_rng(1)
    signals = (rng.standard_normal(11) + 1j * rng.standard_normal(11)).astype(
        np.complex128
    )

    spec = _Exp1D.AXES_SPEC
    assert spec is not None
    assert signals.shape == (len(freqs),)

    result = _Result1D(freqs=freqs, signals=signals, cfg_snapshot=_TinyCfg(reps=7))
    exp = _Exp1D()
    exp.save(os.path.join(str(tmp_path), "scan1d"), result)

    loaded = exp.load(_saved_path(tmp_path, "scan1d"))

    np.testing.assert_allclose(loaded.freqs, freqs, rtol=0, atol=1e-6)
    assert loaded.signals.shape == (len(freqs),)
    assert loaded.signals.dtype == np.complex128
    np.testing.assert_allclose(loaded.signals, signals, rtol=0, atol=0)

    inner_first = tuple(len(getattr(loaded, ax.field_name)) for ax in spec.axes)
    assert loaded.signals.shape == inner_first[::-1]

    assert loaded.cfg_snapshot is not None
    assert loaded.cfg_snapshot.reps == 7


def test_experiment_save_rejects_existing_exact_path(tmp_path: Any) -> None:
    freqs = np.array([4000.0, 5000.0])
    signals = np.ones(2, dtype=np.complex128)
    result = _Result1D(freqs=freqs, signals=signals, cfg_snapshot=_TinyCfg(reps=3))
    exp = _Exp1D()
    base = os.path.join(str(tmp_path), "scan1d")

    exp.save(base, result)
    with pytest.raises(FileExistsError):
        exp.save(base, result)

    assert os.path.exists(_saved_path(tmp_path, "scan1d"))
    assert not os.path.exists(os.path.join(str(tmp_path), "scan1d_1.hdf5"))


def test_real_z_roundtrip_does_not_warn_on_complex_container(tmp_path: Any) -> None:
    freqs = np.array([4000.0, 5000.0, 6000.0], dtype=np.float64)
    signals = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    path = os.path.join(str(tmp_path), "real_z_complex_container.hdf5")
    save_labber_data(
        path,
        z=("Population", "a.u.", signals.astype(np.complex128)),
        axes=[("Frequency", "Hz", freqs * MHZ_TO_HZ)],
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loaded = _Exp1DReal().load(path)

    assert caught == []
    assert loaded.signals.dtype == np.float64
    np.testing.assert_allclose(loaded.signals, signals, rtol=0, atol=0)


@pytest.mark.parametrize("imaginary", [0.25, 1e-12])
def test_real_z_load_rejects_nonzero_imaginary_component(
    tmp_path: Any,
    imaginary: float,
) -> None:
    freqs = np.array([4000.0, 5000.0, 6000.0], dtype=np.float64)
    signals = np.array([0.1, 0.2, 0.3], dtype=np.float64) + 1j * imaginary
    path = os.path.join(str(tmp_path), f"real_z_nonzero_imag_{imaginary}.hdf5")
    save_labber_data(
        path,
        z=("Population", "a.u.", signals.astype(np.complex128)),
        axes=[("Frequency", "Hz", freqs * MHZ_TO_HZ)],
    )

    with pytest.raises(ValueError, match="z channel.*imaginary component"):
        _Exp1DReal().load(path)


def test_save_fast_fails_on_shape_mismatch(tmp_path: Any) -> None:
    """Fast-Fail: z whose shape disagrees with the axis lengths -> save raises.

    The 2-D spec is inner-first ``(Nlength, Nfreq)``; a transposed z
    ``(Nfreq, Nlength)`` (3, 4) disagrees with both axis lengths -> labber_io's
    save validation raises rather than silently transposing.
    """
    freqs = np.linspace(4000.0, 5000.0, 3)  # Nfreq = 3 (inner)
    lengths = np.linspace(0.1, 2.0, 4)  # Nlength = 4 (outer)
    # WRONG orientation: (Nfreq, Nlength) instead of (Nlength, Nfreq)
    bad_signals = np.ones((3, 4), dtype=np.complex128)

    result = _Result2D(
        freqs=freqs, lengths=lengths, signals=bad_signals, cfg_snapshot=_TinyCfg()
    )
    exp = _Exp2D()

    with pytest.raises(ValueError):
        exp.save(os.path.join(str(tmp_path), "bad"), result)


def test_save_fast_fails_without_cfg_snapshot(tmp_path: Any) -> None:
    """cfg_snapshot=None -> save raises (cfg is required to build the comment)."""
    result = _Result1D(
        freqs=np.array([4000.0, 5000.0]),
        signals=np.ones(2, dtype=np.complex128),
        cfg_snapshot=None,
    )
    exp = _Exp1D()
    with pytest.raises(ValueError):
        exp.save(os.path.join(str(tmp_path), "nocfg"), result)


@pytest.mark.parametrize(
    ("axes", "match"),
    [
        (
            [
                ("Wrong Frequency", "Hz", np.array([4000.0, 5000.0]) * MHZ_TO_HZ),
                ("Pulse length", "s", np.array([1.0, 2.0]) * US_TO_S),
            ],
            "axis 0 label",
        ),
        (
            [
                ("Frequency", "wrong", np.array([4000.0, 5000.0]) * MHZ_TO_HZ),
                ("Pulse length", "s", np.array([1.0, 2.0]) * US_TO_S),
            ],
            "axis 0 unit",
        ),
    ],
)
def test_load_rejects_wrong_axis_metadata(
    tmp_path: Any, axes: list[tuple[str, str, np.ndarray]], match: str
) -> None:
    signals = np.ones((2, 2), dtype=np.complex128)
    path = os.path.join(str(tmp_path), "wrong_axis.hdf5")
    save_labber_data(path, z=("S21", "", signals), axes=axes)

    with pytest.raises(ValueError, match=match):
        _Exp2D().load(path)


@pytest.mark.parametrize(
    ("z", "match"),
    [
        (("Wrong S21", "", np.ones((2, 2), dtype=np.complex128)), "z channel label"),
        (("S21", "wrong", np.ones((2, 2), dtype=np.complex128)), "z channel unit"),
    ],
)
def test_load_rejects_wrong_z_channel_metadata(
    tmp_path: Any, z: tuple[str, str, np.ndarray], match: str
) -> None:
    freqs = np.array([4000.0, 5000.0])
    lengths = np.array([1.0, 2.0])
    path = os.path.join(str(tmp_path), "wrong_z.hdf5")
    save_labber_data(
        path,
        z=z,
        axes=[
            ("Frequency", "Hz", freqs * MHZ_TO_HZ),
            ("Pulse length", "s", lengths * US_TO_S),
        ],
    )

    with pytest.raises(ValueError, match=match):
        _Exp2D().load(path)


def test_load_rejects_wrong_z_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = LabberData(
        data=("S21", "", np.ones((2, 3), dtype=np.complex128)),
        axes=[
            ("Frequency", "Hz", np.array([4000.0, 5000.0]) * MHZ_TO_HZ),
            ("Pulse length", "s", np.array([1.0, 2.0]) * US_TO_S),
        ],
    )

    monkeypatch.setattr(
        "zcu_tools.utils.datasaver.load_labber_data",
        lambda _path: payload,
    )

    with pytest.raises(ValueError, match="z shape"):
        _Exp2D().load("fake.hdf5")
