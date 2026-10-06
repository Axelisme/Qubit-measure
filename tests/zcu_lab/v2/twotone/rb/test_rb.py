"""RB acquisition reproducibility and inverse recovery regression checks."""

import numpy as np
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.stop_signal import StopSignal
from zcu_tools.plotting.plots import NonPresentingHost, Plots
from zcu_tools.program.v2.mocksoc import make_mock_soc
from zcu_tools.program.v2.modules.pulse import PulseCfg
from zcu_tools.program.v2.modules.readout import DirectReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg

from zcu_lab.v2.twotone.rb.core import (
    CAYLEY,
    GATE_EFFECT_MAP,
    INVERSE_INDEX,
    BasicGate,
    RB_Exp,
    RBCfg,
    RBModuleCfg,
    RBSweepCfg,
    build_seed_program_tables,
)


def test_acquisition_records_distinct_reproducible_subseeds() -> None:
    soc, soccfg = make_mock_soc()
    pulse = PulseCfg(
        ch=0, nqz=1, freq=1000.0, gain=0.2, waveform=ConstWaveformCfg(length=0.1)
    )
    cfg = RBCfg(
        modules=RBModuleCfg(
            X90_pulse=pulse,
            X180_pulse=pulse.with_updates(gain=0.4),
            readout=DirectReadoutCfg(ro_ch=0, ro_length=1.0, ro_freq=1000.0),
        ),
        sweep=RBSweepCfg(depth=[0, 1, 3]),
        seed=42,
        n_seeds=3,
        reps=2,
        rounds=1,
        relax_delay=1.0,
    )
    seeds = []
    for _ in range(2):
        plots = Plots(NonPresentingHost())
        stop = StopSignal()
        try:
            result = RB_Exp().run(
                cfg,
                context=RunContext(
                    soc=soc, soccfg=soccfg, plots=plots, devices={}, cancel_signal=stop
                ),
            )
            stop.raise_if_error()
        finally:
            plots.finish(present=False)
            plots.release()
        assert result.signals2D.shape == (3, 3)
        assert np.isfinite(result.signals2D).all()
        assert len(np.unique(result.sub_seeds)) == 3
        seeds.append(result.sub_seeds)
    np.testing.assert_array_equal(*seeds)


def test_recovery_returns_all_two_clifford_prefixes_to_ground() -> None:
    names = {
        BasicGate.Id: "Id",
        BasicGate.X90: "X90",
        BasicGate.X180: "X180",
        BasicGate.MX90: "-X90",
        BasicGate.Y90: "Y90",
        BasicGate.Y180: "Y180",
        BasicGate.MY90: "-Y90",
    }
    for first in range(24):
        for second in range(24):
            inverse = [0, INVERSE_INDEX[first], INVERSE_INDEX[CAYLEY[second][first]]]
            gates, lengths, recovery0, recovery1 = build_seed_program_tables(
                [first, second], inverse, np.arange(3, dtype=np.int64)
            )
            for length, r0, r1 in zip(lengths, recovery0, recovery1, strict=True):
                state = 4
                for gate in [*gates[:length], r0, r1]:
                    state = GATE_EFFECT_MAP[names[BasicGate(gate)]][state]
                assert state == 4
