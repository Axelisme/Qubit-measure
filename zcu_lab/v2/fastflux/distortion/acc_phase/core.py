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
class AccPhaseResult:
    lengths: NDArray[np.float64]
    phases: NDArray[np.float64]
    signals: NDArray[np.complex128]


def acc_phase_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return rotate2real(signals).real


class AccPhaseCfg(ConfigBase):
    reset: Reset.Cfg | None = None
    flux_pulse: Pulse.Cfg
    pi2_pulse: Pulse.Cfg
    readout: Readout.Cfg
    length: SweepCfg
    phase: SweepCfg
    readout_t: float
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
class AccPhaseAnalysis:
    lengths_us: NDArray[np.float64]
    phases_rad: NDArray[np.float64]
    detunes_mhz: NDArray[np.float64]


class AccPhaseExp(Experiment[AccPhaseResult, AccPhaseCfg]):
    # inner phases stored as-is (deg) -> IDENTITY; outer lengths mem us, disk s -> US_TO_S
    AXES_SPEC: ClassVar[AxesSpec[AccPhaseResult, AccPhaseCfg]] = AxesSpec(
        axes=(
            AxesSpec.Axis("phases", label="Phase", unit="deg"),
            AxesSpec.Axis("lengths", label="Wait Time", unit="s", scale=1e-6),
        ),
        z=AxesSpec.Z("signals", label="Signal", unit="a.u."),
        result_type=AccPhaseResult,
        cfg_type=AccPhaseCfg,
        cfg_schema_version="3.0",
        tag="fastflux/distortion/acc_phase",
    )

    def run(self, config: AccPhaseCfg, *, context: RunContext) -> AccPhaseResult:
        """Measure flux-induced accumulated phase versus the first π/2 start time.

        After reset, run flux, wait→first π/2, and readout_t in parallel. Apply
        the phase-swept second π/2 only after all three branches finish, then
        read out. readout_t is a minimum join duration in us, not an exact read
        time. Return wait times in us, projection phases in degrees and mean
        complex IQ shaped (wait, phase); analyze derives phase and detuning.
        """
        # Quantize wait and projection-phase axes on the configured hardware.
        cfg = config
        soc, soccfg = context.soc, context.soccfg

        length_sweep = cfg.length
        phase_sweep = cfg.phase

        pi2_pulse = cfg.pi2_pulse

        lengths = sweep2array(length_sweep, "time", {"soccfg": soccfg})
        phases = sweep2array(
            phase_sweep, "phase", {"soccfg": soccfg, "gen_ch": pi2_pulse.ch}
        )

        # Keep one mean IQ value for each wait/projection-phase pair.
        viewer = live_heatmap(context.plots, "measurement", "Time (us)", "Phase (deg)")
        signals_buffer = SignalBuffer(
            (len(lengths), len(phases)),
            on_update=lambda data: viewer.update(
                lengths, phases, acc_phase_signal2real(data)
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
            builder.fork("default", into=["flux", "prepare", "readout_wait"])
            builder.emit(Pulse("flux_pulse", cfg.flux_pulse), timeline="flux")
            builder.timeline_inc(length, timeline="prepare")
            builder.emit(Pulse("pi2_pulse1", cfg.pi2_pulse), timeline="prepare")
            builder.timeline_inc(cfg.readout_t, timeline="readout_wait")
            builder.join("flux", "prepare", "readout_wait", into="default")
            builder.emit(Pulse("pi2_pulse2", cfg.pi2_pulse, phase=phase))
            builder.emit(Readout("readout", cfg.readout))
            prog = builder.build()
            sched.run_acquire(prog, soc, rounds=cfg.rounds)

        return AccPhaseResult(lengths, phases, signals_buffer.array)

    def analyze(
        self,
        source: RunRecord[AccPhaseCfg, AccPhaseResult],
        options: None,
        *,
        plots: Plots,
    ) -> AccPhaseAnalysis:
        """Extract unwrapped phase (rad) and its negative time slope (MHz).

        Fit sine/cosine coefficients across projection phase at each wait.
        Times refer to the first π/2 center; detuning is -dphase/dt/(2π).
        Require source.cfg for pulse timing, otherwise raise ValueError.
        """
        del options  # This analysis has no configurable options.
        result = source.result

        cfg = source.cfg
        if cfg is None:
            raise ValueError("Analysis requires source.cfg")

        flux_pulse = cfg.flux_pulse
        pi2_len = float(cfg.pi2_pulse.waveform.length)

        lengths, phases, signals2D = result.lengths, result.phases, result.signals

        # align to center of pi/2 pulse
        lengths = lengths + pi2_len / 2

        real_signals = acc_phase_signal2real(signals2D)

        thetas = np.deg2rad(phases)
        X = np.column_stack([np.ones_like(thetas), np.cos(thetas), np.sin(thetas)])
        X_inv = np.linalg.pinv(X)
        coeffs = X_inv @ real_signals.T
        init_phases = np.unwrap(np.arctan2(coeffs[2], coeffs[1]))
        detunes = -np.gradient(init_phases, lengths) / (2 * np.pi)

        sort_idxs = np.argsort(np.abs(detunes))
        mean_background = np.mean(detunes[sort_idxs[: int(len(detunes) * 0.1)]])
        mean_topdetune = np.mean(detunes[sort_idxs[int(len(detunes) * 0.9) :]])

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
        ax2.plot(lengths, detunes, ".-")
        ax2.set_ylabel("Detune (MHz)")
        ax2.set_xlabel("Wait Time (us)")

        ax2.axvspan(
            start_t - pi2_len / 2, start_t + pi2_len / 2, color="gray", alpha=0.3
        )
        ax2.axvspan(end_t - pi2_len / 2, end_t + pi2_len / 2, color="gray", alpha=0.3)
        ax1.axvline(start_t, color="black", linestyle="--")
        ax1.axvline(end_t, color="black", linestyle="--")

        ax2.legend()

        fig.tight_layout()

        return AccPhaseAnalysis(
            lengths_us=lengths, phases_rad=init_phases, detunes_mhz=detunes
        )
