from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar

import numpy as np
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from numpy.typing import NDArray
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.program.v2 import (
    GainSweep,
    Pulse,
    Readout,
    Reset,
    SweepCfg,
    TimelineBuilder,
)
from zcu_tools.utils.process import minus_background

from zcu_lab.v2.support import live_heatmap, live_line


@dataclass(frozen=True)
class PowerResult:
    """Drive gains, frequencies in MHz, and mean IQ shaped (Ngain, Nfreq)."""

    gains: NDArray[np.float64]
    freqs: NDArray[np.float64]
    signals: NDArray[np.complex128]


def gain_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return np.abs(minus_background(signals, axis=1))


class PowerCfg(ConfigBase):
    reset: Annotated[Reset.Cfg | None, FieldMeta(label="Reset")] = None
    init_pulse: Annotated[Pulse.Cfg | None, FieldMeta(label="Init Pulse")] = None
    qub_pulse: Annotated[Pulse.Cfg, FieldMeta(label="Probe Pulse")]
    readout: Annotated[Readout.Cfg, FieldMeta(label="Readout")]
    gain: Annotated[GainSweep, FieldMeta(label="Gain", unit="a.u.")]
    freq: Annotated[SweepCfg, FieldMeta(label="Freq", unit="MHz")]
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


class PowerExp(Experiment[PowerResult, PowerCfg]):
    # inner freqs stores MHz on disk (disk Hz) -> scale=1e6; outer gains -> IDENTITY
    AXES_SPEC: ClassVar[AxesSpec[PowerResult, PowerCfg]] = AxesSpec(
        axes=(
            AxesSpec.Axis("freqs", label="Frequency", unit="Hz", scale=1e6),
            AxesSpec.Axis("gains", label="Power", unit="a.u."),
        ),
        z=AxesSpec.Z("signals", label="Signal", unit="a.u."),
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

        gain_sweep = cfg.gain
        freq_sweep = cfg.freq

        gains = sweep2array(
            gain_sweep,
            "gain",
            {"soccfg": soccfg, "gen_ch": cfg.qub_pulse.ch},
            allow_array=True,
        )
        freqs = sweep2array(
            freq_sweep,
            "freq",
            {"soccfg": soccfg, "gen_ch": cfg.qub_pulse.ch},
        )

        fig, _ = context.plots.subplots(1, 2, num='measurement')
        heat_ax, line_ax = fig.axes
        heat_ax.set(xlabel='Pulse Gain (a.u.)', ylabel='Frequency (MHz)')
        line_ax.set_xlabel('Frequency (MHz)')
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-2, 2))
        heat_ax.xaxis.set_major_formatter(formatter)
        heat_ax.yaxis.set_major_formatter(formatter)
        heatmap = live_heatmap(heat_ax, uniform=True)
        line_ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        line_ax.grid()
        lines = live_line(line_ax, num_lines=2)
        for artist in line_ax.lines[:-1]:
            artist.set(marker="None", alpha=0.3, color="red")
        line_ax.lines[-1].set(label="current line", marker=".", color="C0")
        line_ax.legend()
        
        @context.plots.on_plot_thread
        def update(data):
            real = gain_signal2real(data)
            valid = np.flatnonzero(~np.all(np.isnan(real), axis=1))
            current = int(valid[-1]) if valid.size else -1
            history = np.full((2, len(freqs)), np.nan)
            for offset in range(min(2, current + 1)):
                history[-offset - 1] = real[current - offset]
            return (*heatmap.set(gains, freqs, real), *lines.set(freqs, history))
        
        fig.show()
        signals_buffer = SignalBuffer(
            (len(gains), len(freqs)),
            on_update=update,
        )

        # Acquire a complete spectrum at each fixed drive gain.
        with Schedule(signals_buffer, scope=context.scope) as sched:
            for _, step in sched.scan("gain", gains.tolist()):
                builder = TimelineBuilder(
                    soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay
                )

                freq_sweep = cfg.freq
                freq_param = builder.add_sweep("freq", freq_sweep)

                # Reset and initialize before each probe; read out its response.

                if cfg.reset is not None:
                    builder.emit(cfg.reset.build("reset"))
                builder.emit(Pulse("init_pulse", cfg.init_pulse))
                builder.emit(
                    Pulse("qub_pulse", cfg.qub_pulse, freq=freq_param, gain=step.value)
                )
                builder.emit(Readout("readout", cfg.readout))
                prog = builder.build()
                step.run_acquire(prog, soc, rounds=cfg.rounds)

        return PowerResult(
            gains=gains,
            freqs=freqs,
            signals=signals_buffer.array,
        )
