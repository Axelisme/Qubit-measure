"""Simulator inputs shared by the native workflow seam tests."""

from dataclasses import dataclass
from uuid import UUID

from qick import QickConfig
from zcu_tools.datafile import RunSnapshot
from zcu_tools.program.v2.mocksoc import MockQickSoc, make_mock_soc
from zcu_tools.program.v2.modules import DirectReadoutCfg, PulseCfg, PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg
from zcu_tools.program.v2.sim import SimParams
from zcu_tools.program.v2.sim.readout import resonator_freqs
from zcu_tools.program.v2.sweep import SweepCfg
from zcu_tools.simulate.fluxonium import FluxoniumPredictor

from zcu_lab.v2.twotone.freq.core import FreqCfg, FreqModuleCfg, FreqSweepCfg
from zcu_lab.v2.twotone.rabi.len_rabi.core import (
    LenRabiCfg,
    LenRabiModuleCfg,
    LenRabiSweepCfg,
)
from zcu_lab.v2.twotone.ro_optimize.freq_gain.core import (
    FreqGainCfg,
    FreqGainModuleCfg,
    FreqGainSweepCfg,
)
from zcu_lab.v2.twotone.time_domain.t1.core import T1Cfg, T1ModuleCfg, T1SweepCfg
from zcu_lab.workflows.calibration import Calibrated
from zcu_lab.workflows.t1 import T1Plan


@dataclass
class Inputs:
    """Isolated board/config and the four validated native callback inputs.

    soc/soccfg form one physical simulator. freq/rabi/readout/t1 contain native
    cfg for their respective callbacks, calibrated to that simulator's qubit
    and ground-state resonator at reduced flux 1.0. They have no device setup.
    flux_value is the corresponding absolute native device value, not reduced flux.
    """

    soc: MockQickSoc
    soccfg: QickConfig
    freq: FreqCfg
    rabi: LenRabiCfg
    readout: FreqGainCfg
    t1: T1Cfg
    flux_value: float


def make_inputs() -> Inputs:
    sim = SimParams(
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
        poll_latency=0.0,
    )
    soc, soccfg = make_mock_soc(sim=sim)
    predictor = FluxoniumPredictor(
        params=(sim.EJ, sim.EC, sim.EL),
        flux_half=sim.flux_half,
        flux_period=sim.flux_period,
        flux_bias=sim.flux_bias,
    )
    qubit_freq = float(predictor.predict_freq(predictor.flux_to_value(1.0)))
    ro_freq = resonator_freqs(sim, 1.0)[0] * 1000.0
    pi_pulse = PulseCfg(
        ch=0,
        nqz=1,
        gain=1.0,
        freq=qubit_freq,
        waveform=ConstWaveformCfg(length=sim.pi_gain_len),
    )
    readout = DirectReadoutCfg(ro_ch=0, ro_length=1.0, ro_freq=ro_freq)
    weak_pulse = pi_pulse.model_copy(deep=True)
    weak_pulse.gain = 0.05
    weak_pulse.waveform = ConstWaveformCfg(length=2.0)
    pulse_readout = PulseReadoutCfg(
        pulse_cfg=PulseCfg(
            ch=1, nqz=2, gain=0.1, freq=ro_freq, waveform=ConstWaveformCfg(length=1.0)
        ),
        ro_cfg=DirectReadoutCfg(ro_ch=0, gen_ch=1, ro_length=1.0, ro_freq=ro_freq),
    )
    return Inputs(
        soc,
        soccfg,
        FreqCfg(
            reps=100,
            rounds=2,
            relax_delay=200.0,
            modules=FreqModuleCfg(qub_pulse=weak_pulse, readout=readout),
            sweep=FreqSweepCfg(
                freq=SweepCfg(
                    start=qubit_freq - 125.0,
                    stop=qubit_freq + 125.0,
                    expts=31,
                    step=250.0 / 30,
                )
            ),
        ),
        LenRabiCfg(
            reps=100,
            rounds=2,
            relax_delay=200.0,
            modules=LenRabiModuleCfg(qub_pulse=pi_pulse, readout=readout),
            sweep=LenRabiSweepCfg(
                length=SweepCfg(start=0.05, stop=1.2, expts=9, step=1.15 / 8)
            ),
        ),
        FreqGainCfg(
            reps=100,
            rounds=2,
            relax_delay=200.0,
            modules=FreqGainModuleCfg(qub_pulse=pi_pulse, readout=pulse_readout),
            sweep=FreqGainSweepCfg(
                freq=SweepCfg(
                    start=ro_freq - 2.0, stop=ro_freq + 2.0, expts=9, step=0.5
                ),
                gain=SweepCfg(start=0.05, stop=0.15, expts=3, step=0.05),
            ),
        ),
        T1Cfg(
            reps=100,
            rounds=2,
            relax_delay=200.0,
            modules=T1ModuleCfg(pi_pulse=pi_pulse, readout=readout),
            sweep=T1SweepCfg(length=SweepCfg(start=0.0, stop=80.0, expts=21, step=4.0)),
        ),
        float(predictor.flux_to_value(1.0)),
    )


def make_snapshot() -> RunSnapshot:
    """Return explicit historical metadata for these isolated offline acquisitions."""
    return RunSnapshot(
        entry_id=UUID("78fa0941-f9ce-49d4-9665-4eb9302b8162"),
        entry_name="offline",
        point="native",
        description="simulator workflow seam",
        roles={"qubit": "Q1"},
    )


def make_plan(inputs: Inputs) -> T1Plan:
    """Build complete native inputs without replacing acquisition logic."""
    return T1Plan(
        flux_dev="flux",
        fluxes=(1.0, 1.1),
        points=21,
        qub_pulse=inputs.rabi.modules.qub_pulse,
        readout=inputs.readout.modules.readout,
        snapshot=make_snapshot(),
    )


def seed_from_inputs(inputs: Inputs, *, pi_length_us: float = 0.4) -> Calibrated:
    """Create accepted scalars matching this simulator's native drive/readout."""
    frequency = inputs.rabi.modules.qub_pulse.freq
    readout = inputs.readout.modules.readout
    assert isinstance(frequency, float)
    assert isinstance(readout.pulse_cfg.freq, float)
    return Calibrated(frequency, pi_length_us, readout.pulse_cfg.freq, 0.15)
