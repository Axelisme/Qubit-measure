"""Shared physical parameters and program builders for simulator tests."""

from __future__ import annotations

from zcu_tools.program.v2.base import ProgramV2Cfg
from zcu_tools.program.v2.mocksoc import make_mock_soc
from zcu_tools.program.v2.modular import ModularProgramV2
from zcu_tools.program.v2.modules.base import Module
from zcu_tools.program.v2.modules.pulse import PulseCfg
from zcu_tools.program.v2.modules.readout import DirectReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg
from zcu_tools.program.v2.sim import SimParams
from zcu_tools.program.v2.sim.readout import resonator_freqs
from zcu_tools.simulate.fluxonium.predict import FluxoniumPredictor

# Operating point: the engine pins reduced flux = 1.0 (R-3).  At EJ/EC/EL =
# 8.5/1.0/0.5 the qubit sits a few GHz above the sweet spot there, giving a finite
# gap and a clear dispersive shift.  T1/T2 are short so decay/dephasing are visible
# over modest sweeps; snr is generous so the physical structure dominates the noise.
# These tests drive ModularProgramV2 directly with a *raw* (un-folded) sweep axis,
# so the engine's true f_qubit (~7391 MHz) is the same axis the peak is read on —
# Nyquist folding only affects the analyzer's axis, which this layer does not use.
OPERATING_FLUX = 1.0

SIM = SimParams(
    EJ=8.5,
    EC=1.0,
    EL=0.5,
    flux_period=1.0,
    flux_half=0.0,
    flux_bias=0.2,
    T1=20.0,
    T2=10.0,
    T2_star=10.0,  # T2_star == T2 => gamma=0 (pure homogeneous; preserves existing physics)
    bare_rf=7.2,
    g=0.08,
    Ql=5000.0,
    Qi=50000.0,
    snr=200.0,
    pi_gain_len=0.4,
    seed=12345,
)

RESET_RELAX_DELAY = 10.0 * SIM.T1


def qubit_frequency_mhz() -> float:
    """The qubit 0->1 frequency (MHz) the engine sees at the fixed operating flux.

    The engine pins reduced flux = 1.0 (R-3) and feeds ``predict_freq`` a *device
    value*, so map the fixed flux back through ``flux_to_value`` exactly as the
    engine does — this is the true (un-folded) f_qubit the engine drives at.
    """

    predictor = FluxoniumPredictor(
        params=(SIM.EJ, SIM.EC, SIM.EL),
        flux_half=SIM.flux_half,
        flux_period=SIM.flux_period,
        flux_bias=SIM.flux_bias,
    )
    return float(predictor.predict_freq(predictor.flux_to_value(OPERATING_FLUX)))


def ground_resonator_frequency_mhz() -> float:
    """The ground-state dressed resonator frequency (MHz) at the fixed operating flux.

    Reading out near rf_g maximizes the |g>/|e> contrast, which is what makes
    the T1 decay and Rabi oscillation visible in the readout magnitude.
    """

    rf_g, _rf_e = resonator_freqs(SIM, OPERATING_FLUX)
    return rf_g * 1e3


def readout(ro_freq_mhz: float) -> Module:
    return DirectReadoutCfg(ro_ch=0, ro_length=1.0, ro_freq=ro_freq_mhz).build("ro")


def pi_pulse_program(relax_delay: float, *, reps: int = 4) -> ModularProgramV2:
    _soc, soccfg = make_mock_soc(sim=SIM)
    f_qubit = qubit_frequency_mhz()
    pulse = PulseCfg(
        ch=0,
        nqz=1,
        gain=1.0,
        freq=f_qubit,
        phase=0.0,
        waveform=ConstWaveformCfg(length=SIM.pi_gain_len),
    ).build("pi")
    prog = ModularProgramV2(
        soccfg,
        ProgramV2Cfg(reps=reps, rounds=1, relax_delay=relax_delay),
        modules=[pulse, readout(ground_resonator_frequency_mhz())],
    )
    prog.compile()
    return prog
