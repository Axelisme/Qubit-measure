"""Tests for the FLUX-AWARE-MOCK runtime flux binding.

The mock soc can read its operating flux live from a connected ``FakeDevice``
instead of pinning it at reduced flux = 1.0.  This is the per-acquire coupling
that mirrors a real software flux sweep (set device value -> run one acquire ->
qubit frequency follows).  Covers:

- copy-on-input: ``set_flux_device`` on one mock soc never mutates a shared
  SimParams (e.g. the GUI singleton), so two socs built from the same params are
  independent.
- engine reads flux: a bound FakeDevice's value sets the operating point, so the
  engine's ``f_qubit`` / dressed resonator track ``value_to_flux(value)``.
- per-acquire reread: changing the FakeDevice value and acquiring again moves the
  resonator dip (each acquire builds a fresh engine that reads the live value).
- fallback: ``flux_device=None`` reproduces the fixed reduced flux = 1.0 path.
- fail-fast: an unregistered device or a non-FakeDevice raises at acquire time;
  ``set_flux_device`` on a sim-less white-noise soc raises.

``GlobalDeviceManager`` is a class-level singleton, so an autouse fixture clears
its registry around every test (mirrors tests/device/test_manager_lock.py).
"""

from __future__ import annotations

import numpy as np
import pytest
from zcu_tools.device import FakeDevice, GlobalDeviceManager
from zcu_tools.program.v2.base import ProgramV2Cfg
from zcu_tools.program.v2.mocksoc import make_mock_soc
from zcu_tools.program.v2.modular import ModularProgramV2
from zcu_tools.program.v2.modules.readout import DirectReadoutCfg
from zcu_tools.program.v2.sim import SimParams
from zcu_tools.program.v2.sim.engine import SimEngine
from zcu_tools.program.v2.sweep import SweepCfg
from zcu_tools.program.v2.utils import sweep2param
from zcu_tools.simulate.fluxonium.predict import FluxoniumPredictor

# Same operating regime as test_engine: a finite gap + clear dispersive shift.
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


@pytest.fixture(autouse=True)
def _clean_registry():
    """Keep GlobalDeviceManager's singleton registry empty around each test."""

    GlobalDeviceManager._devices.clear()
    yield
    GlobalDeviceManager._devices.clear()


def _predictor() -> FluxoniumPredictor:
    return FluxoniumPredictor(
        params=(_SIM.EJ, _SIM.EC, _SIM.EL),
        flux_half=_SIM.flux_half,
        flux_period=_SIM.flux_period,
        flux_bias=_SIM.flux_bias,
    )


def _engine(prog: ModularProgramV2, soc) -> SimEngine:
    """Build a SimEngine off the soc's internal SimParams copy (narrowed not-None)."""

    sim = soc._sim_params
    assert sim is not None
    return SimEngine(prog, sim)


def _compiled_onetone(soc, soccfg) -> ModularProgramV2:
    """A minimal compiled onetone program usable to build a SimEngine directly."""

    sw = SweepCfg(start=7000.0, stop=7200.0, expts=11, step=20.0)
    ro_param = sweep2param("ro_freq", sw)
    readout = DirectReadoutCfg(ro_ch=0, ro_length=1.0, ro_freq=ro_param).build("ro")
    prog = ModularProgramV2(
        soccfg,
        ProgramV2Cfg(reps=20, rounds=1),
        modules=[readout],
        sweep=[("ro_freq", sw)],
    )
    prog.compile()
    return prog


# ----------------------------------------------------------- copy-on-input


def test_set_flux_device_does_not_mutate_shared_params():
    """copy-on-input: binding flux on one soc leaves the shared SimParams untouched."""

    shared = _SIM  # stand-in for the shared GUI DEFAULT_SIMPARAM singleton
    soc_a, _ = make_mock_soc(sim=shared)
    soc_b, _ = make_mock_soc(sim=shared)

    soc_a.set_flux_device("flux")

    # The caller's SimParams and the other soc's internal copy are unaffected.
    assert shared.flux_device is None
    assert soc_b._sim_params is not None
    assert soc_b._sim_params.flux_device is None
    assert soc_a._sim_params is not None
    assert soc_a._sim_params.flux_device == "flux"


def test_make_mock_soc_copies_params():
    """The soc holds its own SimParams copy, not the caller's instance."""

    soc, _ = make_mock_soc(sim=_SIM)
    assert soc._sim_params is not None
    assert soc._sim_params is not _SIM


def test_set_flux_device_without_sim_raises():
    """A white-noise soc (no SimParams) rejects a flux_device binding (fast-fail)."""

    soc, _ = make_mock_soc()  # sim=None
    assert soc._sim_params is None
    with pytest.raises(RuntimeError, match="requires a SimParams"):
        soc.set_flux_device("flux")


# ----------------------------------------------------------- engine reads flux


def test_engine_operating_flux_tracks_device_value():
    """A bound FakeDevice's value sets the engine's reduced operating flux."""

    device_value = 0.8  # value_to_flux -> (0.8 + 0.2 - 0.0)/1.0 + 0.5 = 1.5
    dev = FakeDevice(fast_mode=True)
    dev.set_value(device_value)
    GlobalDeviceManager.register_device("flux", dev)

    soc, soccfg = make_mock_soc(sim=_SIM)
    soc.set_flux_device("flux")
    prog = _compiled_onetone(soc, soccfg)

    engine = _engine(prog, soc)
    f_qubit_ghz, _rf_g, _rf_e = engine._operating_signal()

    # The engine must land on f_qubit at value_to_flux(device_value), i.e. the same
    # number predict_freq gives for that raw device value.
    pred = _predictor()
    expected_mhz = float(pred.predict_freq(device_value))
    assert f_qubit_ghz * 1e3 == pytest.approx(expected_mhz, rel=1e-9)


def test_engine_fallback_matches_fixed_flux():
    """flux_device=None reproduces the fixed reduced flux = 1.0 operating point."""

    soc, soccfg = make_mock_soc(sim=_SIM)  # no flux_device binding
    prog = _compiled_onetone(soc, soccfg)

    engine = _engine(prog, soc)
    f_qubit_ghz, _rf_g, _rf_e = engine._operating_signal()

    pred = _predictor()
    fixed_value = pred.flux_to_value(1.0)
    expected_mhz = float(pred.predict_freq(fixed_value))
    assert f_qubit_ghz * 1e3 == pytest.approx(expected_mhz, rel=1e-9)


def test_engine_flux_change_moves_f_qubit_per_acquire():
    """Changing the device value between engine builds moves f_qubit (reread)."""

    dev = FakeDevice(fast_mode=True)
    GlobalDeviceManager.register_device("flux", dev)

    soc, soccfg = make_mock_soc(sim=_SIM)
    soc.set_flux_device("flux")
    prog = _compiled_onetone(soc, soccfg)

    # Two values whose reduced fluxes (0.7 and 1.2) are NOT mirror images about an
    # integer/half-integer sweet spot, so f_qubit genuinely differs between them.
    dev.set_value(0.0)  # value_to_flux -> 0.7
    f1 = _engine(prog, soc)._operating_signal()[0]

    dev.set_value(0.5)  # value_to_flux -> 1.2
    f2 = _engine(prog, soc)._operating_signal()[0]

    # Different flux -> different qubit frequency (the binding is read live).
    assert abs(f1 - f2) > 1e-3


# ----------------------------------------------------------- fail-fast


def test_engine_unregistered_device_raises():
    """A flux_device naming an unregistered device fails at resolution time."""

    soc, soccfg = make_mock_soc(sim=_SIM)
    soc.set_flux_device("missing")
    prog = _compiled_onetone(soc, soccfg)

    engine = _engine(prog, soc)
    with pytest.raises(ValueError, match="not found"):
        engine._operating_signal()


def test_engine_non_fake_device_raises():
    """A non-FakeDevice flux source fails fast (only FakeDevice is supported)."""

    class _NotFake:
        def get_value(self) -> float:
            return 0.0

    GlobalDeviceManager.register_device("flux", _NotFake())  # type: ignore[arg-type]

    soc, soccfg = make_mock_soc(sim=_SIM)
    soc.set_flux_device("flux")
    prog = _compiled_onetone(soc, soccfg)

    engine = _engine(prog, soc)
    with pytest.raises(TypeError, match="must be a FakeDevice"):
        engine._operating_signal()


# ----------------------------------------------------------- end-to-end acquire


def test_acquire_dip_tracks_flux():
    """A full acquire's resonator dip moves when the bound device value changes.

    Each acquire builds a fresh SimEngine that reads the live device value, so two
    acquires at two flux values put the dip at two different dressed resonator
    frequencies (the runner's software-per-acquire coupling, end to end).
    """

    from zcu_tools.program.v2.sim.readout import resonator_freqs

    dev = FakeDevice(fast_mode=True)
    GlobalDeviceManager.register_device("flux", dev)

    soc, soccfg = make_mock_soc(sim=_SIM)
    soc.set_flux_device("flux")
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
