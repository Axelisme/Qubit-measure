"""Explicit single-file Labber export contracts (ADR-0063).

Native load contracts live in test_native_persistence and Notebook load tests.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest
from zcu_tools.datafile import decode_labber_comment
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


def _saved_path(tmp_path: Any, base: str) -> str:
    """save() writes the exact formatted path; reservation belongs to callers."""
    return os.path.join(str(tmp_path), f"{base}.hdf5")


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


def test_save_applies_scale_on_disk(tmp_path: Any) -> None:
    """Labber exports carry the declared SI axis scales (Hz / s)."""
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


@pytest.mark.parametrize("tag", [None, "custom/t1"])
def test_save_preserves_comment_and_tag(tmp_path: Path, tag: str | None) -> None:
    from zcu_tools.datafile import load_labber_data

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
    decoded = decode_labber_comment(data.comment)
    cfg, comment, timestamp = decoded.cfg, decoded.comment, decoded.timestamp

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
