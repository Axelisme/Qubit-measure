from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from zcu_tools.device import FakeDeviceInfo
from zcu_tools.experiment.v2.jpa.jpa_check import (
    CheckCfg,
    CheckExp,
    CheckModuleCfg,
    CheckResult,
    CheckSweepCfg,
)
from zcu_tools.experiment.v2.jpa.jpa_flux import (
    FluxCfg,
    FluxExp,
    FluxModuleCfg,
    FluxResult,
    FluxSweepCfg,
)
from zcu_tools.experiment.v2.jpa.jpa_flux_onetone import (
    OneToneFluxCfg,
    OneToneFluxExp,
    OneToneFluxModuleCfg,
    OneToneFluxResult,
    OneToneFluxSweepCfg,
)
from zcu_tools.experiment.v2.jpa.jpa_freq import (
    FreqCfg,
    FreqExp,
    FreqModuleCfg,
    FreqResult,
    FreqSweepCfg,
)
from zcu_tools.experiment.v2.jpa.jpa_power import (
    PowerCfg,
    PowerExp,
    PowerModuleCfg,
    PowerResult,
    PowerSweepCfg,
)
from zcu_tools.program.v2 import (
    DirectReadoutCfg,
    PulseCfg,
    PulseReadoutCfg,
    SweepCfg,
)
from zcu_tools.program.v2.modules import ConstWaveformCfg
from zcu_tools.utils.datasaver import load_labber_data


def _pulse(*, ch: int = 0, freq: float = 6000.0, gain: float = 0.3) -> PulseCfg:
    return PulseCfg(
        waveform=ConstWaveformCfg(length=1.0),
        ch=ch,
        nqz=1,
        freq=freq,
        gain=gain,
    )


def _direct_readout(*, ch: int = 0) -> DirectReadoutCfg:
    return DirectReadoutCfg(ro_ch=0, ro_length=1.0, ro_freq=6000.0, gen_ch=ch)


def _pulse_readout(*, ch: int = 0) -> PulseReadoutCfg:
    return PulseReadoutCfg(
        pulse_cfg=_pulse(ch=ch, gain=0.2), ro_cfg=_direct_readout(ch=ch)
    )


def _dev() -> dict[str, FakeDeviceInfo]:
    return {
        "jpa_flux_dev": FakeDeviceInfo(address="fake_flux", label="jpa_flux_dev"),
        "jpa_rf_dev": FakeDeviceInfo(address="fake_rf", label="jpa_rf_dev"),
    }


def _flux_onetone_cfg(*, n_flux: int = 3, n_freq: int = 5) -> OneToneFluxCfg:
    return OneToneFluxCfg(
        reps=1,
        rounds=1,
        dev=_dev(),
        modules=OneToneFluxModuleCfg(reset=None, readout=_pulse_readout()),
        sweep=OneToneFluxSweepCfg(
            jpa_flux=SweepCfg(
                start=0.1, stop=0.1 + 0.1 * (n_flux - 1), expts=n_flux, step=0.1
            ),
            freq=SweepCfg(
                start=7000.0, stop=7000.0 + (n_freq - 1), expts=n_freq, step=1.0
            ),
        ),
    )


def _flux_cfg() -> FluxCfg:
    return FluxCfg(
        reps=1,
        rounds=1,
        modules=FluxModuleCfg(
            reset=None,
            pi_pulse=_pulse(ch=0, freq=3000.0, gain=0.3),
            readout=_direct_readout(),
        ),
        sweep=FluxSweepCfg(jpa_flux=SweepCfg(start=0.1, stop=0.4, expts=4, step=0.1)),
    )


def _freq_cfg() -> FreqCfg:
    return FreqCfg(
        reps=1,
        rounds=1,
        modules=FreqModuleCfg(
            reset=None,
            pi_pulse=_pulse(ch=0, freq=3000.0, gain=0.3),
            readout=_direct_readout(),
        ),
        sweep=FreqSweepCfg(
            jpa_freq=SweepCfg(start=7200.0, stop=7230.0, expts=4, step=10.0)
        ),
    )


def _power_cfg() -> PowerCfg:
    return PowerCfg(
        reps=1,
        rounds=1,
        dev=_dev(),
        modules=PowerModuleCfg(
            reset=None,
            pi_pulse=_pulse(ch=0, freq=3000.0, gain=0.3),
            readout=_direct_readout(),
        ),
        sweep=PowerSweepCfg(
            jpa_power=SweepCfg(start=-20.0, stop=-14.0, expts=4, step=2.0)
        ),
    )


def _check_cfg() -> CheckCfg:
    return CheckCfg(
        reps=1,
        rounds=1,
        dev=_dev(),
        modules=CheckModuleCfg(reset=None, readout=_pulse_readout()),
        sweep=CheckSweepCfg(
            freq=SweepCfg(start=7000.0, stop=7002.0, expts=3, step=1.0)
        ),
    )


def _saved_path(tmp_path: Path, base: str) -> Path:
    return tmp_path / f"{base}.hdf5"


@pytest.mark.parametrize(("n_flux", "n_freq"), [(3, 5), (5, 3), (3, 3)])
def test_flux_onetone_save_load_roundtrip_preserves_axes_and_signal(
    tmp_path: Path, n_flux: int, n_freq: int
) -> None:
    fluxes = np.linspace(0.1, 0.3, n_flux, dtype=np.float64)
    freqs = np.linspace(7000.0, 7000.0 + (n_freq - 1), n_freq, dtype=np.float64)
    real = np.arange(n_flux * n_freq, dtype=np.float64).reshape(n_flux, n_freq)
    signals = (real + 1j * (real + 0.5)).astype(np.complex128)
    result = OneToneFluxResult(
        fluxes=fluxes,
        freqs=freqs,
        signals=signals,
        cfg_snapshot=_flux_onetone_cfg(n_flux=n_flux, n_freq=n_freq),
    )

    OneToneFluxExp().save(str(tmp_path / "flux_onetone"), result=result, comment="note")

    raw = load_labber_data(str(_saved_path(tmp_path, "flux_onetone")))
    # On-disk axes are inner-first: inner frequency axis first, outer flux axis last.
    assert [axis.name for axis in raw.axes] == ["Readout frequency", "JPA Flux value"]
    assert [axis.unit for axis in raw.axes] == ["Hz", "a.u."]
    np.testing.assert_array_equal(raw.axes[0].values, freqs * 1e6)
    np.testing.assert_array_equal(raw.axes[1].values, fluxes)
    assert raw.z.shape == (n_flux, n_freq)
    np.testing.assert_allclose(raw.z, signals, rtol=0, atol=0)

    loaded = OneToneFluxExp().load(str(_saved_path(tmp_path, "flux_onetone")))
    np.testing.assert_array_equal(loaded.fluxes, fluxes)
    np.testing.assert_array_equal(loaded.freqs, freqs)
    assert loaded.fluxes.dtype == np.float64
    assert loaded.freqs.dtype == np.float64
    assert loaded.signals.shape == (n_flux, n_freq)
    assert loaded.signals.dtype == np.complex128
    np.testing.assert_allclose(loaded.signals, signals, rtol=0, atol=0)
    assert loaded.cfg_snapshot is not None
    assert loaded.cfg_snapshot.sweep.freq.expts == n_freq
    assert loaded.cfg_snapshot.sweep.jpa_flux.expts == n_flux


def test_flux_save_load_roundtrip(tmp_path: Path) -> None:
    result = FluxResult(
        fluxes=np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64),
        signals=np.array([1.0, 2.0, 3.0, 2.5], dtype=np.float64),
        cfg_snapshot=_flux_cfg(),
    )

    FluxExp().save(str(tmp_path / "flux"), result=result, comment="note")

    raw = load_labber_data(str(_saved_path(tmp_path, "flux")))
    assert [axis.name for axis in raw.axes] == ["JPA Flux value"]
    assert [axis.unit for axis in raw.axes] == ["a.u."]
    np.testing.assert_array_equal(raw.axes[0].values, result.fluxes)
    np.testing.assert_allclose(raw.z, result.signals, rtol=0, atol=0)

    loaded = FluxExp().load(str(_saved_path(tmp_path, "flux")))
    np.testing.assert_array_equal(loaded.fluxes, result.fluxes)
    np.testing.assert_allclose(loaded.signals, result.signals, rtol=0, atol=0)
    assert loaded.signals.dtype == np.float64
    assert loaded.cfg_snapshot is not None
    assert loaded.cfg_snapshot.skew_penalty == 0.0


def test_freq_save_load_roundtrip(tmp_path: Path) -> None:
    freqs = np.array([7200.0, 7210.0, 7220.0, 7230.0], dtype=np.float64)
    result = FreqResult(
        freqs=freqs,
        signals=np.array([1.0, 4.0, 5.0, 2.0], dtype=np.float64),
        cfg_snapshot=_freq_cfg(),
    )

    FreqExp().save(str(tmp_path / "freq"), result=result, comment="note")

    raw = load_labber_data(str(_saved_path(tmp_path, "freq")))
    assert [axis.name for axis in raw.axes] == ["JPA Frequency"]
    assert [axis.unit for axis in raw.axes] == ["Hz"]
    np.testing.assert_array_equal(raw.axes[0].values, freqs * 1e6)
    np.testing.assert_allclose(raw.z, result.signals, rtol=0, atol=0)

    loaded = FreqExp().load(str(_saved_path(tmp_path, "freq")))
    np.testing.assert_array_equal(loaded.freqs, freqs)
    np.testing.assert_allclose(loaded.signals, result.signals, rtol=0, atol=0)
    assert loaded.freqs.dtype == np.float64
    assert loaded.signals.dtype == np.float64
    assert loaded.cfg_snapshot is not None
    assert loaded.cfg_snapshot.sweep.jpa_freq.expts == 4


def test_power_save_load_roundtrip(tmp_path: Path) -> None:
    result = PowerResult(
        powers=np.array([-20.0, -18.0, -16.0, -14.0], dtype=np.float64),
        signals=np.array([0.5, 1.5, 2.5, 1.0], dtype=np.float64),
        cfg_snapshot=_power_cfg(),
    )

    PowerExp().save(str(tmp_path / "power"), result=result, comment="note")

    raw = load_labber_data(str(_saved_path(tmp_path, "power")))
    assert [axis.name for axis in raw.axes] == ["JPA Power"]
    assert [axis.unit for axis in raw.axes] == ["dBm"]
    np.testing.assert_array_equal(raw.axes[0].values, result.powers)
    np.testing.assert_allclose(raw.z, result.signals, rtol=0, atol=0)

    loaded = PowerExp().load(str(_saved_path(tmp_path, "power")))
    np.testing.assert_array_equal(loaded.powers, result.powers)
    np.testing.assert_allclose(loaded.signals, result.signals, rtol=0, atol=0)
    assert loaded.powers.dtype == np.float64
    assert loaded.signals.dtype == np.float64
    assert loaded.cfg_snapshot is not None
    assert loaded.cfg_snapshot.sweep.jpa_power.expts == 4


def test_check_save_load_roundtrip(tmp_path: Path) -> None:
    freqs = np.array([7000.0, 7001.0, 7002.0], dtype=np.float64)
    outputs = np.array([0.0, 1.0], dtype=np.float64)
    real = np.arange(6, dtype=np.float64).reshape(2, 3)
    signals = (real + 1j * (real + 1.0)).astype(np.complex128)
    result = CheckResult(
        outputs=outputs,
        freqs=freqs,
        signals=signals,
        cfg_snapshot=_check_cfg(),
    )

    CheckExp().save(str(tmp_path / "check"), result=result, comment="note")

    raw = load_labber_data(str(_saved_path(tmp_path, "check")))
    assert [axis.name for axis in raw.axes] == ["Frequency", "JPA Output"]
    assert [axis.unit for axis in raw.axes] == ["Hz", "a.u."]
    np.testing.assert_array_equal(raw.axes[0].values, freqs * 1e6)
    np.testing.assert_array_equal(raw.axes[1].values, outputs)
    assert raw.z.shape == (2, 3)
    np.testing.assert_allclose(raw.z, signals, rtol=0, atol=0)

    loaded = CheckExp().load(str(_saved_path(tmp_path, "check")))
    np.testing.assert_array_equal(loaded.outputs, outputs)
    np.testing.assert_array_equal(loaded.freqs, freqs)
    assert np.issubdtype(loaded.outputs.dtype, np.integer)
    assert loaded.freqs.dtype == np.float64
    assert loaded.signals.shape == (2, 3)
    assert loaded.signals.dtype == np.complex128
    np.testing.assert_allclose(loaded.signals, signals, rtol=0, atol=0)
    assert loaded.cfg_snapshot is not None
    assert loaded.cfg_snapshot.sweep.freq.expts == 3
