from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from zcu_tools.experiment.v2.twotone.reset.rabi_check import (
    RabiCheckCfg,
    RabiCheckExp,
    RabiCheckModuleCfg,
    RabiCheckResult,
    RabiCheckSweepCfg,
)
from zcu_tools.program.v2 import (
    DirectReadoutCfg,
    PulseCfg,
    PulseResetCfg,
    SweepCfg,
)
from zcu_tools.program.v2.modules import ConstWaveformCfg
from zcu_tools.utils.datasaver import load_labber_data


def _pulse(*, freq: float = 100.0, gain: float = 0.1) -> PulseCfg:
    return PulseCfg(
        type="pulse",
        waveform=ConstWaveformCfg(style="const", length=1.0),
        ch=0,
        nqz=1,
        freq=freq,
        gain=gain,
    )


def _cfg() -> RabiCheckCfg:
    return RabiCheckCfg(
        reps=1,
        rounds=1,
        modules=RabiCheckModuleCfg(
            rabi_pulse=_pulse(freq=3000.0, gain=0.0),
            tested_reset=PulseResetCfg(pulse_cfg=_pulse(freq=1500.0, gain=0.2)),
            pi_pulse=_pulse(freq=3000.0, gain=0.4),
            readout=DirectReadoutCfg(
                type="readout/direct",
                ro_ch=0,
                ro_length=1.0,
                ro_freq=6000.0,
            ),
        ),
        sweep=RabiCheckSweepCfg(gain=SweepCfg(start=0.0, stop=0.5, step=0.25, expts=3)),
    )


def _sample_result(*, with_cfg: bool = True) -> RabiCheckResult:
    gains = np.array([0.0, 0.25, 0.5], dtype=np.float64)
    values = np.arange(9, dtype=np.float64).reshape(3, 3)
    signals = (values + 1j * (values + 0.5)).astype(np.complex128)
    return RabiCheckResult(
        gains=gains,
        signals=signals,
        cfg_snapshot=_cfg() if with_cfg else None,
    )


def _saved_path(tmp_path: Path, base: str) -> Path:
    return tmp_path / f"{base}.hdf5"


def test_rabi_check_save_load_roundtrip(tmp_path: Path) -> None:
    result = _sample_result()
    exp = RabiCheckExp()

    exp.save(str(tmp_path / "rabi_check"), result=result, comment="note")
    path = _saved_path(tmp_path, "rabi_check")
    assert path.exists()

    load_exp = RabiCheckExp()
    loaded = load_exp.load(str(path))

    np.testing.assert_allclose(loaded.gains, result.gains, rtol=0, atol=0)
    np.testing.assert_array_equal(
        loaded.reset_states, np.array([0, 1, 2], dtype=np.int64)
    )
    assert loaded.reset_states.dtype == np.int64
    assert loaded.signals.shape == (3, len(result.gains))
    assert loaded.signals.dtype == np.complex128
    np.testing.assert_allclose(loaded.signals, result.signals, rtol=0, atol=0)
    assert load_exp.last_result is loaded
    assert loaded.cfg_snapshot is not None
    assert loaded.cfg_snapshot.sweep.gain.expts == 3


def test_rabi_check_raw_labber_axes_and_tag(tmp_path: Path) -> None:
    result = _sample_result()
    RabiCheckExp().save(str(tmp_path / "raw"), result=result)

    raw = load_labber_data(str(_saved_path(tmp_path, "raw")))

    assert [axis.name for axis in raw.axes] == ["Amplitude", "Reset"]
    assert [axis.unit for axis in raw.axes] == ["a.u.", "None"]
    np.testing.assert_allclose(raw.axes[0].values, result.gains, rtol=0, atol=0)
    np.testing.assert_allclose(raw.axes[1].values, result.reset_states, rtol=0, atol=0)
    assert raw.data.name == "Signal"
    assert raw.data.unit == "a.u."
    assert raw.z.shape == (3, len(result.gains))
    assert raw.tags == ["twotone/reset/rabi_check"]


def test_rabi_check_save_fast_fails_on_shape_mismatch(tmp_path: Path) -> None:
    gains = np.array([0.0, 0.25, 0.5, 0.75], dtype=np.float64)
    bad_signals = np.ones((len(gains), 3), dtype=np.complex128)
    result = RabiCheckResult(gains=gains, signals=bad_signals, cfg_snapshot=_cfg())

    with pytest.raises(ValueError):
        RabiCheckExp().save(str(tmp_path / "bad_shape"), result=result)


def test_rabi_check_save_fast_fails_without_cfg_snapshot(tmp_path: Path) -> None:
    result = _sample_result(with_cfg=False)

    with pytest.raises(ValueError, match="cfg_snapshot"):
        RabiCheckExp().save(str(tmp_path / "no_cfg"), result=result)


def test_rabi_check_result_default_reset_states() -> None:
    first = _sample_result()
    second = _sample_result()

    expected = np.array([0, 1, 2], dtype=np.int64)
    np.testing.assert_array_equal(first.reset_states, expected)
    assert first.reset_states.dtype == np.int64
    assert first.reset_states is not second.reset_states
