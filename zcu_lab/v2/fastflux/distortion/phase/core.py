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
class PhaseResult:
    lengths: NDArray[np.float64]
    phases: NDArray[np.float64]
    signals: NDArray[np.complex128]


def phase_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    """Return the real-valued measurement view of complex IQ without changing the input."""
    return rotate2real(signals).real


class PhaseCfg(ConfigBase):
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
            description="Enter the flux transient to characterize; its pre-delay places the leading edge.",
        ),
    ]
    pi2_pulse: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="π/2 pulse",
            description="Enter a calibrated qubit π/2 pulse for the delayed pair.",
        ),
    ]
    readout: Annotated[
        Readout.Cfg,
        FieldMeta(
            label="Readout",
            description="Enter the calibrated resonator readout used after both drives finish.",
        ),
    ]
    readout_t: Annotated[
        float,
        FieldMeta(
            label="Minimum readout wait",
            unit="µs",
            description="Enter the minimum time before the final readout branch can start.",
        ),
    ]
    length: Annotated[
        SweepCfg,
        FieldMeta(
            label="π/2 pair wait",
            unit="µs",
            description="Enter the waiting time before the first pulse of the π/2 pair starts.",
        ),
    ]
    phase: Annotated[
        SweepCfg,
        FieldMeta(
            label="Projection phase",
            unit="deg",
            description="Enter the carrier phase of the second π/2 pulse.",
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


class PhaseExp(Experiment[PhaseResult, PhaseCfg]):
    # inner phases raw deg (IDENTITY); outer lengths memory-us -> disk-s via US_TO_S
    AXES_SPEC: ClassVar[AxesSpec[PhaseResult, PhaseCfg]] = AxesSpec(
        axes=(
            AxesSpec.Axis("phases", label="Phase", unit="deg"),
            AxesSpec.Axis("lengths", label="Wait Time", unit="s", scale=1e-6),
        ),
        z=AxesSpec.Z("signals", label="Signal", unit="a.u."),
        result_type=PhaseResult,
        cfg_type=PhaseCfg,
        cfg_schema_version="3.0",
        tag="fastflux/distortion/phase",
    )

    def run(self, config: PhaseCfg, *, context: RunContext) -> PhaseResult:
        """Measure the phase response of a delayed π/2 pair to a flux transient.

        After reset, run flux alongside wait→π/2→phase-swept π/2 and a
        readout_t wait. Both π/2 pulses stay inside the same branch, with no
        added free-evolution delay between them. Read out after the longest
        branch; readout_t is a minimum duration in us. Return waits in us,
        phases in degrees and mean complex IQ shaped (wait, phase).
        """
        # 1. Quantize the pair start time and the second pulse projection phase.
        cfg = config
        soc, soccfg = context.soc, context.soccfg

        length_sweep = cfg.length
        phase_sweep = cfg.phase

        pi2_pulse = cfg.pi2_pulse

        lengths = sweep2array(length_sweep, "time", {"soccfg": soccfg})
        phases = sweep2array(
            phase_sweep, "phase", {"soccfg": soccfg, "gen_ch": pi2_pulse.ch}
        )

        # 2. Keep one mean IQ value for each pair-start/projection-phase setting.
        centers = lengths + float(pi2_pulse.waveform.length)
        fig, ax = context.plots.subplots(num="measurement", layout="constrained")
        fig.suptitle("Fast-flux transient · Qubit · Local phase")
        ax.set(xlabel="π/2 pair midpoint (µs)", ylabel="Projection phase (deg)")
        ax.grid(False)
        viewer = live_heatmap(ax)
        (image,) = viewer.set(
            centers, phases, np.full((centers.size, phases.size), np.nan)
        )
        fig.colorbar(image, ax=ax, label="Mean IQ real projection (a.u.)")

        @context.plots.on_plot_thread
        def update(data: NDArray[np.complex128]) -> Iterable[Artist]:
            """Display available IQ on the scan grid without changing the buffer."""
            return viewer.set(centers, phases, phase_signal2real(data))

        fig.show()
        signals_buffer = SignalBuffer(
            (len(lengths), len(phases)),
            on_update=update,
        )

        # 3. Project the delayed pi/2 pair during flux, then join before readout.
        with Schedule(signals_buffer, scope=context.scope) as sched:
            builder = TimelineBuilder(
                soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay
            )
            length = builder.add_sweep("length", cfg.length)
            phase = builder.add_sweep("phase", cfg.phase)
            if cfg.reset is not None:
                builder.emit(Reset("reset", cfg.reset))
            builder.fork("default", into=["flux", "pair", "readout_wait"])
            builder.emit(Pulse("flux_pulse", cfg.flux_pulse), timeline="flux")
            builder.timeline_inc(length, timeline="pair")
            builder.emit(Pulse("pi2_pulse1", cfg.pi2_pulse), timeline="pair")
            builder.emit(
                Pulse("pi2_pulse2", cfg.pi2_pulse, phase=phase), timeline="pair"
            )
            builder.timeline_inc(cfg.readout_t, timeline="readout_wait")
            builder.join("flux", "pair", "readout_wait", into="default")
            builder.emit(Readout("readout", cfg.readout))
            prog = builder.build()
            sched.run_acquire(prog, soc, rounds=cfg.rounds)

        return PhaseResult(lengths, phases, signals_buffer.array)

    def analyze(
        self,
        source: RunRecord[PhaseCfg, PhaseResult],
        options: None,
        *,
        plots: Plots,
    ) -> None:
        """Extract unwrapped phase in degrees versus the π/2-pair midpoint (us).

        Fit sine/cosine coefficients across projection phase at each wait.
        Require source.cfg for pair/flux timing, otherwise raise ValueError.
        The overlaid rectangular curve is a timing reference, not a fitted
        transfer function.
        """
        # 1. Require saved pulse timing to interpret the projection-phase map.
        del options  # This analysis has no configurable options.
        result = source.result

        cfg = source.cfg
        if cfg is None:
            raise ValueError("Analysis requires source.cfg")

        flux_pulse = cfg.flux_pulse
        pi2_len = float(cfg.pi2_pulse.waveform.length)

        lengths, phases, signals2D = result.lengths, result.phases, result.signals

        # 2. Recover unwrapped projection phase at the midpoint of each pi/2 pair.
        lengths = lengths + pi2_len

        real_signals = phase_signal2real(signals2D)

        thetas = np.deg2rad(phases)
        X = np.column_stack([np.ones_like(thetas), np.cos(thetas), np.sin(thetas)])
        coeffs = np.linalg.pinv(X) @ real_signals.T
        init_phases = np.rad2deg(np.unwrap(np.arctan2(coeffs[2], coeffs[1])))

        # 3. Use phase plateaus to construct a rectangular flux-timing reference.
        sort_idxs = np.argsort(np.abs(init_phases))
        mean_background = np.median(
            init_phases[sort_idxs[: int(len(init_phases) * 0.2)]]
        )
        mean_topdetune = np.median(
            init_phases[sort_idxs[int(len(init_phases) * 0.8) :]]
        )

        start_t = float(flux_pulse.pre_delay)
        end_t = start_t + float(flux_pulse.waveform.length)

        ideal_lengths = np.linspace(lengths[0], lengths[-1], 1000)
        ideal_curve = np.full_like(ideal_lengths, mean_background)
        ideal_curve[(ideal_lengths >= start_t) & (ideal_lengths <= end_t)] = (
            mean_topdetune
        )

        # 4. Compare the phase map and recovered phase with flux-edge timing windows.
        fig, ax = plots.subplots(num="fit", layout="constrained")
        fig.suptitle("Fast-flux transient · Qubit · Local phase")
        ax.set(xlabel="π/2 pair midpoint (µs)", ylabel="Projection phase (deg)")
        ax.grid(False)
        (image,) = live_heatmap(ax).set(lengths, phases, real_signals)
        fig.colorbar(image, ax=ax, label="Mean IQ real projection (a.u.)")

        fig, ax = plots.subplots(num="phase_response", layout="constrained")
        fig.suptitle("Fast-flux transient · Qubit · Phase response")
        ax.set(xlabel="π/2 pair midpoint (µs)", ylabel="Local phase (deg)")
        ax.plot(lengths, init_phases, "k.-", label="Phase from IQ fringe")
        ax.plot(
            ideal_lengths, ideal_curve, color="green", label="Rectangular reference"
        )
        edge_half_width = pi2_len
        ax.axvspan(
            start_t - edge_half_width,
            start_t + edge_half_width,
            color="gray",
            alpha=0.25,
            label="Pulse overlaps edge",
        )
        ax.axvspan(
            end_t - edge_half_width, end_t + edge_half_width, color="gray", alpha=0.25
        )
        ax.legend(loc="upper left", bbox_to_anchor=(0, -0.15), ncols=2)
