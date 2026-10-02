"""Public mock acquisition and explicit live flux-source contracts."""

from __future__ import annotations

import numpy as np
import pytest
from zcu_tools.device import FakeDevice
from zcu_tools.program.v2.base import ProgramV2Cfg
from zcu_tools.program.v2.mocksoc import make_mock_soc
from zcu_tools.program.v2.modular import ModularProgramV2
from zcu_tools.program.v2.modules.readout import DirectReadoutCfg
from zcu_tools.program.v2.sim import SimParams
from zcu_tools.program.v2.sim.engine import SimEngine
from zcu_tools.program.v2.sweep import SweepCfg
from zcu_tools.program.v2.utils import sweep2param
from zcu_tools.simulate.fluxonium.predict import FluxoniumPredictor

_SIM = SimParams(
    EJ=8.5,
    EC=1.0,
    EL=0.5,
    flux_period=1.0,
    flux_half=0.0,
    flux_bias=0.2,
    T1=20.0,
    T2=10.0,
    T2_star=10.0,
    bare_rf=7.2,
    g=0.08,
    Ql=5000.0,
    Qi=50000.0,
    snr=200.0,
    pi_gain_len=0.4,
    seed=12345,
)


def _predictor() -> FluxoniumPredictor:
    return FluxoniumPredictor(
        params=(_SIM.EJ, _SIM.EC, _SIM.EL),
        flux_half=_SIM.flux_half,
        flux_period=_SIM.flux_period,
        flux_bias=_SIM.flux_bias,
    )


def _compiled_onetone(soccfg) -> ModularProgramV2:
    sw = SweepCfg(start=7000.0, stop=7200.0, expts=11, step=20.0)
    readout = DirectReadoutCfg(
        ro_ch=0, ro_length=1.0, ro_freq=sweep2param("ro_freq", sw)
    ).build("ro")
    prog = ModularProgramV2(
        soccfg,
        ProgramV2Cfg(reps=20, rounds=1),
        modules=[readout],
        sweep=[("ro_freq", sw)],
    )
    prog.compile()
    return prog


def test_flux_binding_is_local_to_soc_and_preserves_parameter_values():
    before = _SIM.model_dump()
    first, _ = make_mock_soc(sim=_SIM)
    second, _ = make_mock_soc(sim=_SIM)
    device = FakeDevice(fast_mode=True)
    first.set_flux_source(device.get_value)
    assert first.flux_source is not None
    assert first.flux_source() == device.get_value()
    assert second.flux_source is None
    assert first.sim_params is not _SIM
    assert first.sim_params is not None
    assert first.sim_params.model_dump() == before
    assert _SIM.model_dump() == before


def test_white_noise_soc_rejects_flux_binding():
    soc, _ = make_mock_soc()
    with pytest.raises(RuntimeError, match="requires a SimParams"):
        soc.set_flux_source(lambda: 0.0)


def test_invalid_binding_preserves_previous_source():
    soc, _ = make_mock_soc(sim=_SIM)

    def source() -> float:
        return 0.5

    soc.set_flux_source(source)
    with pytest.raises(TypeError, match="callable"):
        soc.set_flux_source("unresolved-device")  # type: ignore[arg-type]
    assert soc.flux_source is source


def test_unbinding_restores_fixed_flux_acquisition():
    soc, soccfg = make_mock_soc(sim=_SIM)

    def source() -> float:
        return _predictor().flux_to_value(1.0)

    prog = _compiled_onetone(soccfg)
    soc.set_flux_source(source)
    bound = SimEngine(prog, _SIM, flux_source=soc.flux_source).compute_round(0)
    soc.set_flux_source(None)
    unbound = SimEngine(prog, _SIM, flux_source=soc.flux_source).compute_round(0)
    np.testing.assert_array_equal(bound[0], unbound[0])


def test_engine_reads_source_once_per_acquire_not_per_round():
    _, soccfg = make_mock_soc(sim=_SIM)
    prog = _compiled_onetone(soccfg)
    values = iter([0.0, 0.5])
    reads = []

    def read_value():
        value = next(values)
        reads.append(value)
        return value

    first = SimEngine(prog, _SIM, flux_source=read_value)
    first.compute_round(0)
    first.compute_round(1)
    assert reads == [0.0]
    second = SimEngine(prog, _SIM, flux_source=read_value)
    second.compute_round(0)
    assert reads == [0.0, 0.5]


def test_source_failure_propagates_without_fixed_flux_fallback():
    soc, soccfg = make_mock_soc(sim=_SIM)

    def failed_source():
        raise RuntimeError("flux source unavailable")

    soc.set_flux_source(failed_source)
    prog = _compiled_onetone(soccfg)
    with pytest.raises(RuntimeError, match="flux source unavailable"):
        prog.acquire(soc, progress=False)


def test_acquire_dip_tracks_flux():
    """A full acquire's resonator dip moves when the bound device value changes.

    Each acquire builds a fresh SimEngine that reads the live device value, so two
    acquires at two flux values put the dip at two different dressed resonator
    frequencies (the runner's software-per-acquire coupling, end to end).
    """

    from zcu_tools.program.v2.sim.readout import resonator_freqs

    dev = FakeDevice(fast_mode=True)

    soc, soccfg = make_mock_soc(sim=_SIM)
    soc.set_flux_source(dev.get_value)
    pred = _predictor()

    def _dip_freq(device_value: float) -> float:
        dev.set_value(device_value)
        rf_g, _ = resonator_freqs(_SIM, pred.value_to_flux(device_value))
        rf_g_mhz = rf_g * 1e3
        sw = SweepCfg(
            start=rf_g_mhz - 100.0, stop=rf_g_mhz + 100.0, expts=81, step=200.0 / 80
        )
        ro_param = sweep2param("ro_freq", sw)
        readout = DirectReadoutCfg(ro_ch=0, ro_length=1.0, ro_freq=ro_param).build("ro")
        prog = ModularProgramV2(
            soccfg,
            ProgramV2Cfg(reps=80, rounds=1),
            modules=[readout],
            sweep=[("ro_freq", sw)],
        )
        result = prog.acquire(soc, progress=False)
        iq = result[0][0]
        amp = np.abs(iq[:, 0] + 1j * iq[:, 1])
        freqs = np.linspace(sw.start, sw.stop, sw.expts)
        return float(freqs[int(np.argmin(amp))])

    # Two flux points (reduced flux 0.7 and 1.2) whose dressed resonator
    # frequencies differ measurably (not mirror images about a sweet spot).
    dip_lo = _dip_freq(0.0)
    dip_hi = _dip_freq(0.5)

    rf_g_lo = resonator_freqs(_SIM, pred.value_to_flux(0.0))[0] * 1e3
    rf_g_hi = resonator_freqs(_SIM, pred.value_to_flux(0.5))[0] * 1e3

    # The two operating points have distinct dressed resonator frequencies, so the
    # flux binding actually moves the readout dip (not a degenerate no-op).
    assert abs(rf_g_lo - rf_g_hi) > 1.0

    # Each dip lands near its own flux-dependent dressed resonator frequency.
    assert abs(dip_lo - rf_g_lo) < 30.0
    assert abs(dip_hi - rf_g_hi) < 30.0
