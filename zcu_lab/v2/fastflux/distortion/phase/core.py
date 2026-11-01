from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, ClassVar

import numpy as np
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
    return rotate2real(signals).real


class PhaseCfg(ConfigBase):
    reset: Reset.Cfg | None = None
    flux_pulse: Pulse.Cfg
    pi2_pulse: Pulse.Cfg
    readout: Readout.Cfg
    readout_t: float
    length: SweepCfg
    phase: SweepCfg
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


@dataclass(frozen=True)
class PhaseAnalysis:
    lengths_us: NDArray[np.float64]
    phases_deg: NDArray[np.float64]


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
        # Quantize the pair start time and the second pulse projection phase.
        cfg = config
        soc, soccfg = context.soc, context.soccfg

        length_sweep = cfg.length
        phase_sweep = cfg.phase

        pi2_pulse = cfg.pi2_pulse

        lengths = sweep2array(length_sweep, "time", {"soccfg": soccfg})
        phases = sweep2array(
            phase_sweep, "phase", {"soccfg": soccfg, "gen_ch": pi2_pulse.ch}
        )

        # Keep one mean IQ value for each pair-start/projection-phase setting.
        viewer = live_heatmap(context.plots, "measurement", "Time (us)", "Phase (deg)")
        signals_buffer = SignalBuffer(
            (len(lengths), len(phases)),
            on_update=lambda data: viewer.update(
                lengths, phases, phase_signal2real(data)
            ),
        )

        with Schedule( signals_buffer, scope=context.scope) as sched:
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
    ) -> PhaseAnalysis:
        """Extract unwrapped phase in degrees versus the π/2-pair midpoint (us).

        Fit sine/cosine coefficients across projection phase at each wait.
        Require source.cfg for pair/flux timing, otherwise raise ValueError.
        The overlaid rectangular curve is a timing reference, not a fitted
        transfer function.
        """
        del options  # This analysis has no configurable options.
        result = source.result

        cfg = source.cfg
        if cfg is None:
            raise ValueError("Analysis requires source.cfg")

        flux_pulse = cfg.flux_pulse
        pi2_len = float(cfg.pi2_pulse.waveform.length)

        lengths, phases, signals2D = result.lengths, result.phases, result.signals

        # align to middle of two pi/2 pulses
        lengths = lengths + pi2_len

        real_signals = phase_signal2real(signals2D)

        thetas = np.deg2rad(phases)
        X = np.column_stack([np.ones_like(thetas), np.cos(thetas), np.sin(thetas)])
        coeffs = np.linalg.pinv(X) @ real_signals.T
        init_phases = np.rad2deg(np.unwrap(np.arctan2(coeffs[2], coeffs[1])))

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

        fig, _ = plots.subplots(
            num="fit", nrows=2, ncols=1, figsize=(8, 4), sharex=True
        )
        ax1, ax2 = fig.axes
        ax1.imshow(
            real_signals.T,
            extent=(lengths[0], lengths[-1], phases[0], phases[-1]),
            aspect="auto",
            interpolation="none",
            cmap="RdBu_r",
        )
        ax1.set_ylabel("Phase (deg)")

        ax2.plot(ideal_lengths, ideal_curve, "g-", label="Ideal")
        ax2.plot(lengths, init_phases, ".-")
        ax2.set_ylabel("Phase (deg)")
        ax2.set_xlabel("Wait Time (us)")

        ax2.axvspan(start_t - pi2_len, start_t + pi2_len, color="gray", alpha=0.3)
        ax1.axvline(start_t, color="black", linestyle="--")
        ax2.axvspan(end_t - pi2_len, end_t + pi2_len, color="gray", alpha=0.3)
        ax1.axvline(end_t, color="black", linestyle="--")

        ax2.legend()

        fig.tight_layout()

        return PhaseAnalysis(lengths_us=lengths, phases_deg=init_phases)
