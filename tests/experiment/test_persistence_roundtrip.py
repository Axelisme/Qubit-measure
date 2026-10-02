"""Regression test for the PersistableExperiment save/load round-trip (ADR-0063).

Exercises the *base-level* persistence mechanism end-to-end against a real
on-disk Labber HDF5 file (no labber_io mocking): a minimal in-test
``PersistableExperiment`` subclass declares an ``AXES_SPEC``, we pair pure data
and configuration in a RunRecord, save it to ``tmp_path``, and load it back.
We assert axis values, the complex z array, the record cfg, and the inner-first
shape invariant all round-trip.

Persistence invariants under test (ADR-0063; formerly ADR-0027):
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
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest
from zcu_tools.datafile import LabberData, save_labber_data
from zcu_tools.experiment.axes_spec import (
    MHZ_TO_HZ,
    US_TO_S,
    AxesSpec,
    Axis,
    ZSpec,
)
from zcu_tools.experiment.base import PersistableExperiment
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.records import RunRecord

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


@dataclass(frozen=True)
class _Result1D:
    """Frozen Result for the 1-axis spec (proves N-D generality at N=1)."""

    freqs: np.ndarray  # inner axis, in MHz
    signals: np.ndarray  # complex z, shape (Nfreq,)


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


@dataclass(frozen=True)
class _RecordData1D:
    freqs: np.ndarray
    signals: np.ndarray


class _RecordExp1D(PersistableExperiment[_RecordData1D, _TinyCfg]):
    AXES_SPEC: ClassVar[AxesSpec[Any, Any] | None] = AxesSpec(
        axes=(Axis("freqs", "Frequency", "Hz", MHZ_TO_HZ, np.float64),),
        z=ZSpec("signals", "S21", "", np.complex128),
        result_type=_RecordData1D,
        cfg_type=_TinyCfg,
        tag="test/record",
    )


def test_record_roundtrip_keeps_config_with_its_data(tmp_path: Path) -> None:
    cfg = _TinyCfg(name="source-A", reps=7)
    data = _RecordData1D(
        freqs=np.array([400.0, 420.0]),
        signals=np.array([1.0 + 2.0j, 3.0 + 4.0j]),
    )
    source = RunRecord(cfg=cfg, result=data)
    cfg.name = "later-input"
    exp = _RecordExp1D()
    destination = tmp_path / "source-A.hdf5"

    exp.save(source, destination)
    restored = exp.load(destination)

    assert restored.cfg is not None
    assert restored.cfg.name == "source-A"
    assert restored.cfg.reps == 7
    np.testing.assert_array_equal(restored.result.freqs, data.freqs)
    np.testing.assert_array_equal(restored.result.signals, data.signals)


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
    result = _Result2D(freqs=freqs, lengths=lengths, signals=signals)

    exp = _Exp2D()
    base = tmp_path / "scan2d"
    exp.save(RunRecord(cfg=cfg, result=result), base)

    path = _saved_path(tmp_path, "scan2d")
    assert os.path.exists(path)

    loaded = exp.load(Path(path))

    # axis values round-trip within scale tolerance (memory units restored)
    np.testing.assert_allclose(loaded.result.freqs, freqs, rtol=0, atol=1e-6)
    np.testing.assert_allclose(loaded.result.lengths, lengths, rtol=0, atol=1e-9)
    assert loaded.result.freqs.dtype == np.float64
    assert loaded.result.lengths.dtype == np.float64

    # z array matches exactly (dtype + shape + values), zero transpose
    assert loaded.result.signals.shape == signals.shape
    assert loaded.result.signals.dtype == np.complex128
    np.testing.assert_allclose(loaded.result.signals, signals, rtol=0, atol=0)

    # inner-first shape invariant holds on the loaded data
    inner_first = tuple(len(getattr(loaded.result, ax.field_name)) for ax in spec.axes)
    assert loaded.result.signals.shape == inner_first[::-1]

    # Record configuration round-trips through the comment channel
    assert loaded.cfg is not None
    assert loaded.cfg.name == "twotone-len"
    assert loaded.cfg.reps == 512


def test_save_applies_scale_on_disk(tmp_path: Any) -> None:
    """Disk values carry the SI scale (Hz / s), proving load divides it back."""
    from zcu_tools.datafile import load_labber_data

    freqs = np.array([4000.0, 4500.0, 5000.0])  # MHz
    lengths = np.array([1.0, 2.0])  # us
    signals = np.ones((2, 3), dtype=np.complex128)
    cfg = _TinyCfg()
    result = _Result2D(freqs, lengths, signals)

    exp = _Exp2D()
    exp.save(RunRecord(cfg=cfg, result=result), tmp_path / "scaled")
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

    result = _Result1D(freqs=freqs, signals=signals)
    exp = _Exp1D()
    exp.save(RunRecord(cfg=_TinyCfg(reps=7), result=result), tmp_path / "scan1d")

    loaded = exp.load(Path(_saved_path(tmp_path, "scan1d")))

    np.testing.assert_allclose(loaded.result.freqs, freqs, rtol=0, atol=1e-6)
    assert loaded.result.signals.shape == (len(freqs),)
    assert loaded.result.signals.dtype == np.complex128
    np.testing.assert_allclose(loaded.result.signals, signals, rtol=0, atol=0)

    inner_first = tuple(len(getattr(loaded.result, ax.field_name)) for ax in spec.axes)
    assert loaded.result.signals.shape == inner_first[::-1]

    assert loaded.cfg is not None
    assert loaded.cfg.reps == 7


def test_experiment_save_rejects_existing_exact_path(tmp_path: Any) -> None:
    freqs = np.array([4000.0, 5000.0])
    signals = np.ones(2, dtype=np.complex128)
    source = RunRecord(cfg=_TinyCfg(reps=3), result=_Result1D(freqs, signals))
    exp = _Exp1D()
    base = tmp_path / "scan1d"

    exp.save(source, base)
    with pytest.raises(FileExistsError):
        exp.save(source, base)

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
        loaded = _Exp1DReal().load(Path(path))

    assert caught == []
    assert loaded.cfg is None
    assert loaded.result.signals.dtype == np.float64
    np.testing.assert_allclose(loaded.result.signals, signals, rtol=0, atol=0)


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
        _Exp1DReal().load(Path(path))


def test_save_uses_explicit_result_after_loading_another(tmp_path: Path) -> None:
    exp = _Exp1D()
    first = RunRecord(
        cfg=_TinyCfg(name="first"),
        result=_Result1D(
            np.array([4000.0, 5000.0]),
            np.array([1.0 + 2.0j, 3.0 + 4.0j]),
        ),
    )
    second = RunRecord(
        cfg=_TinyCfg(name="second"),
        result=_Result1D(
            np.array([6000.0, 7000.0]),
            np.array([5.0 + 6.0j, 7.0 + 8.0j]),
        ),
    )
    exp.save(second, tmp_path / "second.hdf5")
    loaded_second = exp.load(tmp_path / "second.hdf5")
    exp.save(first, tmp_path / "first.hdf5")
    loaded_first = exp.load(tmp_path / "first.hdf5")

    np.testing.assert_array_equal(loaded_first.result.signals, first.result.signals)
    np.testing.assert_array_equal(loaded_first.result.freqs, first.result.freqs)
    np.testing.assert_array_equal(loaded_second.result.signals, second.result.signals)
    assert loaded_first.cfg is not None
    assert loaded_first.cfg.name == "first"
    assert loaded_second.cfg is not None
    assert loaded_second.cfg.name == "second"


@pytest.mark.parametrize("tag", [None, "custom/t1"])
def test_save_preserves_comment_and_tag(tmp_path: Path, tag: str | None) -> None:
    from zcu_tools.datafile import load_labber_data
    from zcu_tools.experiment.utils import parse_comment

    result = _Result1D(
        np.array([4000.0, 5000.0]),
        np.ones(2, dtype=np.complex128),
    )
    path = tmp_path / "metadata.hdf5"
    _Exp1D().save(
        RunRecord(cfg=_TinyCfg(reps=17), result=result),
        path,
        comment="T1 observation",
        tag=tag,
    )
    data = load_labber_data(str(path))
    cfg, comment, timestamp = parse_comment(data.comment)

    assert cfg is not None
    assert cfg["reps"] == 17
    assert comment == "T1 observation"
    assert timestamp is not None
    assert data.tags == [tag or "test/roundtrip1d"]


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

    result = _Result2D(freqs=freqs, lengths=lengths, signals=bad_signals)
    exp = _Exp2D()

    with pytest.raises(ValueError, match="axis.*length.*z dim"):
        exp.save(RunRecord(cfg=_TinyCfg(), result=result), tmp_path / "bad")


def test_save_fast_fails_without_record_cfg(tmp_path: Any) -> None:
    """A data-only record cannot be saved without comment configuration."""
    result = _Result1D(
        freqs=np.array([4000.0, 5000.0]),
        signals=np.ones(2, dtype=np.complex128),
    )
    exp = _Exp1D()
    with pytest.raises(ValueError, match=r"RunRecord\.cfg is None"):
        exp.save(
            RunRecord[_TinyCfg, _Result1D](cfg=None, result=result), tmp_path / "nocfg"
        )


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
        _Exp2D().load(Path(path))


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
        _Exp2D().load(Path(path))


def test_load_rejects_wrong_z_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = LabberData(
        data=("S21", "", np.ones((2, 3), dtype=np.complex128)),
        axes=[
            ("Frequency", "Hz", np.array([4000.0, 5000.0]) * MHZ_TO_HZ),
            ("Pulse length", "s", np.array([1.0, 2.0]) * US_TO_S),
        ],
    )

    monkeypatch.setattr(
        "zcu_tools.datafile.load_labber_data",
        lambda _path: payload,
    )

    with pytest.raises(ValueError, match="z shape"):
        _Exp2D().load(Path("fake.hdf5"))
