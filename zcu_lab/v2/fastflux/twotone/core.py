from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, ClassVar

import numpy as np
from matplotlib.ticker import ScalarFormatter
from numpy.typing import NDArray
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    GainSweep,
    Pulse,
    Readout,
    Reset,
    SweepCfg,
    TimelineBuilder,
)
from zcu_tools.utils.process import rotate2real

from zcu_lab.v2.support import live_heatmap


@dataclass(frozen=True)
class TwoToneResult:
    gains: NDArray[np.float64]
    freqs: NDArray[np.float64]
    signals: NDArray[np.complex128]


def twotone_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return rotate2real(signals).real


class TwotoneCfg(ConfigBase):
    reset: Reset.Cfg | None = None
    flux_pulse: Pulse.Cfg
    qub_pulse: Pulse.Cfg
    readout: Readout.Cfg
    gain: Annotated[GainSweep, FieldMeta(label="gain", unit="a.u.")]
    freq: SweepCfg
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


class TwoToneExp(Experiment[TwoToneResult, TwotoneCfg]):
    # inner freqs stores MHz on disk (disk Hz) -> scale=1e6; outer gains -> IDENTITY
    AXES_SPEC: ClassVar[AxesSpec[TwoToneResult, TwotoneCfg]] = AxesSpec(
        axes=(
            AxesSpec.Axis("freqs", label="Frequency", unit="Hz", scale=1e6),
            AxesSpec.Axis("gains", label="Flux Pulse Gain", unit="a.u."),
        ),
        z=AxesSpec.Z("signals", label="Signal", unit="a.u."),
        result_type=TwoToneResult,
        cfg_type=TwotoneCfg,
        cfg_schema_version="3.0",
        tag="fastflux/twotone",
    )

    def run(self, config: TwotoneCfg, *, context: RunContext) -> TwoToneResult:
        """Map qubit spectroscopy versus flux pulse gain.

        After reset, start flux and frequency-swept qubit-probe branches
        together, then read out after both finish. Return flux gains in a.u.,
        probe frequencies in MHz and mean complex IQ shaped (gain, frequency).
        The map locates spectral features; it does not fit a transition or
        calibrate gain to physical flux.
        """
        # Quantize flux gain and probe frequency for their own generators.
        cfg = config
        soc, soccfg = context.soc, context.soccfg

        gains = sweep2array(
            cfg.gain,
            "gain",
            {"soccfg": soccfg, "gen_ch": cfg.flux_pulse.ch},
        )
        freqs = sweep2array(
            cfg.freq,
            "freq",
            {"soccfg": soccfg, "gen_ch": cfg.qub_pulse.ch},
        )

        # Keep one averaged spectrum per flux gain.
        fig, ax = context.plots.subplots(num="measurement")
        ax.set(xlabel="Flux Pulse Gain (a.u.)", ylabel="Frequency (MHz)")
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-2, 2))
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)
        viewer = live_heatmap(ax)

        @context.plots.on_plot_thread
        def update(data):
            return viewer.set(gains, freqs, twotone_signal2real(data))

        fig.show()
        signals_buffer = SignalBuffer(
            (len(gains), len(freqs)),
            on_update=update,
        )

        with Schedule(signals_buffer, scope=context.scope) as sched:
            # Build the pulse sequence without changing the caller's cfg.
            builder = TimelineBuilder(
                soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay
            )
            gain = builder.add_sweep("gain", cfg.gain)
            freq = builder.add_sweep("freq", cfg.freq)
            if cfg.reset is not None:
                builder.emit(Reset("reset", cfg.reset))
            builder.fork("default", into=["flux", "probe"])
            builder.emit(
                Pulse("flux_pulse", cfg.flux_pulse, gain=gain), timeline="flux"
            )
            builder.emit(Pulse("qub_pulse", cfg.qub_pulse, freq=freq), timeline="probe")
            builder.join("flux", "probe", into="default")
            builder.emit(Readout("readout", cfg.readout))
            prog = builder.build()
            sched.run_acquire(prog, soc, rounds=cfg.rounds)

        return TwoToneResult(gains, freqs, signals_buffer.array)

    def analyze(
        self,
        source: RunRecord[TwotoneCfg, TwoToneResult],
        options: None,
        *,
        plots: Plots,
    ) -> None:
        """Plot the projected IQ spectroscopy map without a resonance fit."""
        del options  # This analysis has no configurable options.
        result = source.result

        gains, freqs, signals2D = result.gains, result.freqs, result.signals

        real_signals = twotone_signal2real(signals2D)

        fig, ax = plots.subplots(num="fit")

        ax.imshow(
            real_signals.T,
            extent=(
                float(gains[0]),
                float(gains[-1]),
                float(freqs[0]),
                float(freqs[-1]),
            ),
            aspect="auto",
            origin="lower",
            interpolation="none",
            cmap="RdBu_r",
        )
        ax.set_xlabel("Flux Pulse Gain (a.u.)")
        ax.set_ylabel("Frequency (MHz)")

        fig.tight_layout()
