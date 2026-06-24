from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from zcu_tools.experiment.v2.twotone.reset.bath.freq import (
    FreqGainCfg,
    FreqGainExp,
    FreqGainModuleCfg,
    FreqGainResult,
    FreqGainSweepCfg,
)
from zcu_tools.program.v2 import BathResetCfg, DirectReadoutCfg, PulseCfg, SweepCfg
from zcu_tools.program.v2.modules import ConstWaveformCfg
from zcu_tools.utils.datasaver import load_labber_data


def _pulse(
    *,
    freq: float = 100.0,
    gain: float = 0.1,
    phase: float = 0.0,
) -> PulseCfg:
    return PulseCfg(
        type="pulse",
        waveform=ConstWaveformCfg(style="const", length=1.0),
        ch=0,
        nqz=1,
        freq=freq,
        gain=gain,
        phase=phase,
    )


def _cfg() -> FreqGainCfg:
    return FreqGainCfg(
        reps=1,
        rounds=1,
        modules=FreqGainModuleCfg(
            reset=None,
            init_pulse=None,
            tested_reset=BathResetCfg(
                cavity_tone_cfg=_pulse(freq=5500.0, gain=0.0),
                qubit_tone_cfg=_pulse(freq=3000.0, gain=0.1),
                pi2_cfg=_pulse(freq=3000.0, gain=0.2, phase=90.0),
            ),
            readout=DirectReadoutCfg(
                type="readout/direct",
                ro_ch=0,
                ro_length=1.0,
                ro_freq=6000.0,
            ),
        ),
        sweep=FreqGainSweepCfg(
            gain=SweepCfg(start=0.4, stop=0.7, step=0.3, expts=2),
            freq=SweepCfg(start=5480.0, stop=5520.0, step=20.0, expts=3),
        ),
    )


def _sample_result(
    *,
    with_cfg: bool = True,
    omit_phases: bool = False,
) -> FreqGainResult:
    gains = np.array([0.4, 0.7], dtype=np.float64)
    freqs = np.array([5480.0, 5500.0, 5520.0], dtype=np.float64)
    values = np.arange(24, dtype=np.float64).reshape(4, 2, 3)
    signals = (values + 1j * (values + 0.5)).astype(np.complex128)
    cfg_snapshot = _cfg() if with_cfg else None
    if omit_phases:
        return FreqGainResult(
            gains=gains,
            freqs=freqs,
            signals=signals,
            cfg_snapshot=cfg_snapshot,
        )
    return FreqGainResult(
        gains=gains,
        freqs=freqs,
        phases=np.array([0.0, 90.0, 180.0, 270.0], dtype=np.float64),
        signals=signals,
        cfg_snapshot=cfg_snapshot,
    )


def _saved_path(tmp_path: Path, base: str) -> Path:
    return tmp_path / f"{base}_1.hdf5"


def test_bath_freq_gain_raw_labber_axes_tag_shape_and_no_sidecar(
    tmp_path: Path,
) -> None:
    result = _sample_result()

    FreqGainExp().save(str(tmp_path / "bath_freq_gain"), result=result)
    raw = load_labber_data(str(_saved_path(tmp_path, "bath_freq_gain")))

    assert [axis.name for axis in raw.axes] == [
        "Cavity Frequency",
        "Cavity drive Gain",
        "Pi2 Phase",
    ]
    assert [axis.unit for axis in raw.axes] == ["Hz", "a.u.", "deg"]
    np.testing.assert_allclose(raw.axes[0].values, result.freqs * 1e6, rtol=0, atol=0)
    np.testing.assert_allclose(raw.axes[1].values, result.gains, rtol=0, atol=0)
    np.testing.assert_allclose(raw.axes[2].values, result.phases, rtol=0, atol=0)
    assert raw.data.name == "Signal"
    assert raw.data.unit == "a.u."
    assert raw.z.shape == (len(result.phases), len(result.gains), len(result.freqs))
    assert raw.tags == ["twotone/reset/bath/freq_gain"]
    assert not any(tmp_path.glob("*_0deg*"))
    assert not any(tmp_path.glob("*_90deg*"))
    assert not any(tmp_path.glob("*_180deg*"))
    assert not any(tmp_path.glob("*_270deg*"))


def test_bath_freq_gain_save_load_roundtrip(tmp_path: Path) -> None:
    result = _sample_result()
    FreqGainExp().save(str(tmp_path / "roundtrip"), result=result, comment="note")

    load_exp = FreqGainExp()
    loaded = load_exp.load(str(_saved_path(tmp_path, "roundtrip")))

    np.testing.assert_allclose(loaded.freqs, result.freqs, rtol=0, atol=0)
    np.testing.assert_allclose(loaded.gains, result.gains, rtol=0, atol=0)
    np.testing.assert_allclose(loaded.phases, result.phases, rtol=0, atol=0)
    assert loaded.freqs.dtype == np.float64
    assert loaded.gains.dtype == np.float64
    assert loaded.phases.dtype == np.float64
    assert loaded.signals.shape == result.signals.shape == (4, 2, 3)
    assert loaded.signals.dtype == np.complex128
    np.testing.assert_allclose(loaded.signals, result.signals, rtol=0, atol=0)
    assert load_exp.last_result is loaded
    assert loaded.cfg_snapshot is not None
    assert loaded.cfg_snapshot.sweep.freq.expts == 3
    assert loaded.cfg_snapshot.sweep.gain.expts == 2


def test_bath_freq_gain_save_fast_fails_on_shape_mismatch(tmp_path: Path) -> None:
    result = _sample_result()
    bad_signals = np.ones(
        (len(result.gains), len(result.freqs), len(result.phases)),
        dtype=np.complex128,
    )
    bad_result = FreqGainResult(
        gains=result.gains,
        freqs=result.freqs,
        phases=result.phases,
        signals=bad_signals,
        cfg_snapshot=_cfg(),
    )

    with pytest.raises(ValueError):
        FreqGainExp().save(str(tmp_path / "bad_shape"), result=bad_result)


def test_bath_freq_gain_save_fast_fails_without_cfg_snapshot(
    tmp_path: Path,
) -> None:
    result = _sample_result(with_cfg=False)

    with pytest.raises(ValueError, match="cfg_snapshot"):
        FreqGainExp().save(str(tmp_path / "no_cfg"), result=result)


def test_bath_freq_gain_result_default_phases_are_isolated() -> None:
    first = _sample_result(with_cfg=False, omit_phases=True)
    second = _sample_result(with_cfg=False, omit_phases=True)

    expected = np.array([0.0, 90.0, 180.0, 270.0], dtype=np.float64)
    np.testing.assert_array_equal(first.phases, expected)
    np.testing.assert_array_equal(second.phases, expected)
    assert first.phases.dtype == np.float64
    assert second.phases.dtype == np.float64
    assert first.phases is not second.phases

    first.phases[0] = -1.0
    assert second.phases[0] == 0.0
