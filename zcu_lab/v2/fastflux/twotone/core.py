from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Annotated, ClassVar

import numpy as np
from matplotlib.artist import Artist
from numpy.typing import NDArray
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
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
    """Return the real-valued measurement view of complex IQ without changing the input."""
    return rotate2real(signals).real


class TwotoneCfg(ConfigBase):
    reset: Annotated[
        Reset.Cfg | None,
        FieldMeta(
            label="Reset",
            description="Choose the reset module before each shot, or leave it disabled.",
        ),
    ] = None
    flux_pulse: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Flux pulse",
            description="Enter the flux pulse played alongside the spectroscopy pulse; Flux pulse gain supplies its swept gain.",
        ),
    ]
    qub_pulse: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Qubit probe",
            description="Enter the spectroscopy pulse; Qubit probe frequency supplies its swept carrier.",
        ),
    ]
    readout: Annotated[
        Readout.Cfg,
        FieldMeta(
            label="Readout",
            description="Enter the calibrated resonator readout used after both drives finish.",
        ),
    ]
    gain: Annotated[
        SweepCfg,
        FieldMeta(
            label="Flux pulse gain",
            unit="a.u.",
            description="Enter the signed DAC gain scan of the flux pulse.",
        ),
    ]
    freq: Annotated[
        SweepCfg,
        FieldMeta(
            label="Qubit probe frequency",
            unit="MHz",
            description="Enter the absolute RF carrier scan that spans the qubit resonance.",
        ),
    ]
    reps: Annotated[
        int,
        FieldMeta(
            label="Reps",
            unit="count",
            description="Enter the number of shots per scan point in each round.",
        ),
    ]
    rounds: Annotated[
        int,
        FieldMeta(
            label="Rounds",
            unit="count",
            description="Enter the number of times to acquire and average the full scan.",
        ),
    ] = 1
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relax delay",
            unit="µs",
            decimals=3,
            description="Enter the time for relaxation after each shot.",
        ),
    ]


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
        # 1. Quantize flux gain and probe frequency for their own generators.
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

        # 2. Keep one averaged spectrum per flux gain.
        fig, ax = context.plots.subplots(num="measurement", layout="constrained")
        fig.suptitle("Fast-flux spectroscopy · Qubit · Resonance map")
        ax.set(xlabel="Flux pulse gain (a.u.)", ylabel="Qubit probe frequency (MHz)")
        ax.set_title(f"0/{gains.size * freqs.size} scan points with IQ")
        ax.grid(False)
        viewer = live_heatmap(ax)
        (image,) = viewer.set(gains, freqs, np.full((gains.size, freqs.size), np.nan))
        fig.colorbar(image, ax=ax, label="Mean IQ real projection (a.u.)")

        @context.plots.on_plot_thread
        def update(data: NDArray[np.complex128]) -> Iterable[Artist]:
            """Display available IQ on the scan grid without changing the buffer."""
            count = np.count_nonzero(np.isfinite(data))
            title = ax.set_title(f"{count}/{data.size} scan points with IQ")
            return (*viewer.set(gains, freqs, twotone_signal2real(data)), title)

        fig.show()
        signals_buffer = SignalBuffer(
            (len(gains), len(freqs)),
            on_update=update,
        )

        with Schedule(signals_buffer, scope=context.scope) as sched:
            # 3. Start flux and spectroscopy together; read out after both pulses finish.
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
        # 1. Project mean IQ to spectroscopy contrast without fitting a resonance.
        del options  # This analysis has no configurable options.
        result = source.result

        gains, freqs, signals2D = result.gains, result.freqs, result.signals

        real_signals = twotone_signal2real(signals2D)

        # 2. Locate spectral features in the flux-gain/frequency contrast map.
        fig, ax = plots.subplots(num="fit", layout="constrained")
        fig.suptitle("Fast-flux spectroscopy · Qubit · Resonance map")
        ax.set(xlabel="Flux pulse gain (a.u.)", ylabel="Qubit probe frequency (MHz)")
        ax.grid(False)
        (image,) = live_heatmap(ax).set(gains, freqs, real_signals)
        fig.colorbar(image, ax=ax, label="Mean IQ real projection (a.u.)")
