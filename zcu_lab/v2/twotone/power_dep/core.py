from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar

import numpy as np
from numpy.typing import NDArray
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.program.v2 import Pulse, Readout, Reset, SweepCfg, sweep2param
from zcu_tools.utils.process import minus_background

from zcu_lab.v2.support import live_heatmap_with_line


@dataclass(frozen=True)
class PowerResult:
    """Drive gains, frequencies in MHz, and mean IQ shaped (Ngain, Nfreq)."""

    gains: NDArray[np.float64]
    freqs: NDArray[np.float64]
    signals: NDArray[np.complex128]


def gain_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return np.abs(minus_background(signals, axis=1))


class PowerCfg(ConfigBase):
    reps: Annotated[
        int, FieldMeta(label="Reps", description="Number of repetitions per round.")
    ] = 1
    rounds: Annotated[
        int,
        FieldMeta(label="Rounds", description="Number of software acquisition rounds."),
    ] = 1
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relax delay",
            unit="us",
            decimals=3,
            description="Relaxation delay after each repetition.",
        ),
    ] = 1.0
    reset: Annotated[Reset.Cfg | None, FieldMeta(label="Reset")] = None
    init_pulse: Annotated[Pulse.Cfg | None, FieldMeta(label="Init Pulse")] = None
    qub_pulse: Annotated[Pulse.Cfg, FieldMeta(label="Probe Pulse")]
    readout: Annotated[Readout.Cfg, FieldMeta(label="Readout")]
    gain: Annotated[SweepCfg, FieldMeta(label="Gain", unit="a.u.")]
    freq: Annotated[SweepCfg, FieldMeta(label="Freq", unit="MHz")]


class PowerExp(Experiment[PowerResult, PowerCfg]):
    # inner freqs stores MHz on disk (disk Hz) -> scale=1e6; outer gains -> IDENTITY
    AXES_SPEC: ClassVar[AxesSpec[PowerResult, PowerCfg]] = AxesSpec(
        axes=(
            AxesSpec.Axis("freqs", "Frequency", "Hz", scale=1e6),
            AxesSpec.Axis("gains", "Power", "a.u."),
        ),
        z=AxesSpec.Z("signals", "Signal", "a.u."),
        result_type=PowerResult,
        cfg_type=PowerCfg,
        cfg_schema_version="3.0",
        tag="twotone/power_dep",
    )

    def run(
        self,
        cfg: PowerCfg,
        *,
        context: RunContext,
    ) -> PowerResult:
        """Map the qubit spectrum as a function of drive gain.

        For each host-selected gain, reset, apply the optional init pulse,
        sweep the qubit probe frequency on the FPGA, then read out. Return
        mean complex IQ on the (gain, frequency) grid. Gain is the configured
        drive amplitude, not a calibrated power in dBm.
        """
        # Prepare drive-gain rows and the probe's hardware frequency grid.
        cfg = deepcopy(cfg)
        soc, soccfg = context.soc, context.soccfg
        modules = cfg

        gain_sweep = cfg.gain
        freq_sweep = cfg.freq

        gains = sweep2array(
            gain_sweep,
            "gain",
            {"soccfg": soccfg, "gen_ch": modules.qub_pulse.ch},
            allow_array=True,
        )
        freqs = sweep2array(
            freq_sweep,
            "freq",
            {"soccfg": soccfg, "gen_ch": modules.qub_pulse.ch},
        )

        viewer = live_heatmap_with_line(
            context.plots,
            "measurement",
            "Pulse Gain (a.u.)",
            "Frequency (MHz)",
            line_axis=1,
            num_lines=2,
        )
        signals_buffer = SignalBuffer(
            (len(gains), len(freqs)),
            on_update=lambda data: viewer.update(gains, freqs, gain_signal2real(data)),
        )

        # Acquire a complete spectrum at each fixed drive gain.
        with Schedule(cfg, signals_buffer, scope=context.scope) as sched:
            for _, step in sched.scan("gain", gains.tolist()):
                cfg = step.cfg
                modules = cfg

                modules.qub_pulse.set_param("gain", step.value)
                freq_sweep = cfg.freq
                modules.qub_pulse.set_param("freq", sweep2param("freq", freq_sweep))

                # Reset and initialize before each probe; read out its response.
                _ = (
                    step.prog_builder(
                        soc,
                        soccfg,
                        reps=step.cfg.reps,
                        relax_delay=step.cfg.relax_delay,
                    )
                    .add_reset("reset", modules.reset)
                    .add_pulse("init_pulse", modules.init_pulse)
                    .add_pulse("qubit_pulse", modules.qub_pulse)
                    .add_readout("readout", modules.readout)
                    .declare_sweep("freq", freq_sweep)
                    .build_and_acquire(rounds=cfg.rounds)
                )

        return PowerResult(
            gains=gains,
            freqs=freqs,
            signals=signals_buffer.array,
        )
