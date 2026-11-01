from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Annotated, ClassVar

import numpy as np
from numpy.typing import NDArray
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime.schedule import Schedule
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    Pulse,
    PulseReset,
    Readout,
    Reset,
    SweepCfg,
    TimelineBuilder,
)

from zcu_lab.v2.support.reset import reset_scan_buffer


@dataclass(frozen=True)
class LengthResult:
    """Quantized lengths (us) and complex IQ shaped (before_pi, after_pi, length).

    before_pi and after_pi each store [0, 1]: calibrated pi off, then on.
    signals[a, b, i] is mean IQ at lengths[i] with those two switches.
    Unacquired values remain NaN.
    """

    lengths: NDArray[np.float64]
    signals: NDArray[np.complex128]
    before_pi: NDArray[np.int64] = field(
        default_factory=lambda: np.array([0, 1], dtype=np.int64)
    )
    after_pi: NDArray[np.int64] = field(
        default_factory=lambda: np.array([0, 1], dtype=np.int64)
    )


@dataclass(frozen=True)
class LengthAnalysis:
    """Four-branch contrasts and an unclipped residual estimate per length.

    contrast_ground is d0=S00-S01; contrast_excited is d1=S10-S11, in IQ units.
    contrast_ratio is Re(d1*conj(d0))/|d0|², NaN where undefined.
    residual_estimate=(1-r)/2 assumes p0≈0; it is not reset fidelity.
    selected_length is the shortest quantized duration (us) with residual ≤ epsilon
    and all longer measured finite points also passing, or None if none qualify.
    epsilon records the operator's analysis threshold.
    """

    contrast_ground: NDArray[np.complex128]
    contrast_excited: NDArray[np.complex128]
    contrast_ratio: NDArray[np.float64]
    residual_estimate: NDArray[np.float64]
    selected_length: float | None
    epsilon: float


@dataclass(frozen=True)
class LengthAnalyzeOptions:
    """Residual threshold for selection only; it does not change acquisition."""

    epsilon: Annotated[
        float,
        FieldMeta(
            label="Epsilon",
            description="Maximum p0≈0 residual at the candidate and all longer finite points.",
        ),
    ] = 0.05


class LengthCfg(ConfigBase):
    """Sweep a const flux reset's length between two calibrated pi switches."""

    reset: Annotated[
        Reset.Cfg | None,
        FieldMeta(label="Reset", description="Optional reset before each branch."),
    ] = None
    pi_pulse: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Pi pulse",
            description="Calibrated pi used both before and after the tested reset.",
        ),
    ]
    tested_reset: Annotated[
        PulseReset.Cfg,
        FieldMeta(
            label="Tested reset",
            description="Zero-frequency const flux pulse at fixed gain; hold length is swept.",
        ),
    ]
    readout: Annotated[Readout.Cfg, FieldMeta(label="Readout")]
    length: Annotated[SweepCfg, FieldMeta(label="Length", unit="us")]

    reps: Annotated[
        int,
        FieldMeta(label="Reps", description="Repetitions per branch/length per round."),
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


class LengthExp(Experiment[LengthResult, LengthCfg]):
    Options: ClassVar[type[LengthAnalyzeOptions]] = LengthAnalyzeOptions

    AXES_SPEC: ClassVar[AxesSpec[LengthResult, LengthCfg]] = AxesSpec(
        axes=(
            AxesSpec.Axis("lengths", label="Flux reset length", unit="s", scale=1e-6),
            AxesSpec.Axis(
                "after_pi",
                label="After reset pi (0=off, 1=on)",
                unit="None",
                dtype=np.int64,
            ),
            AxesSpec.Axis(
                "before_pi",
                label="Before reset pi (0=off, 1=on)",
                unit="None",
                dtype=np.int64,
            ),
        ),
        z=AxesSpec.Z("signals", label="Signal", unit="a.u.", dtype=np.complex128),
        result_type=LengthResult,
        cfg_type=LengthCfg,
        cfg_schema_version="3.0",
        tag="twotone/reset/flux/length",
    )

    def run(self, cfg: LengthCfg, *, context: RunContext) -> LengthResult:
        """Measure four mean-IQ branches per flux hold length at fixed gain.

        In one program apply optional reset, before-reset pi off/on, the tested
        flux pulse, after-reset pi off/on, and the same readout. Both switches
        use pi_pulse unchanged. Keep flux gain and pre/post delays fixed.
        Return rounded lengths (us) and IQ shaped (2, 2, N), switches [0, 1].
        Reject non-const or nonzero-frequency flux pulses before acquisition.
        Live curves show I only, without choosing a population direction.
        """
        cfg = deepcopy(cfg)
        pulse_cfg = cfg.tested_reset.pulse_cfg
        if pulse_cfg.waveform.style != "const" or pulse_cfg.freq != 0.0:
            raise ValueError("Flux reset requires a const pulse at zero frequency.")

        soc, soccfg = context.soc, context.soccfg
        lengths = sweep2array(
            cfg.length, "time", {"soccfg": soccfg, "gen_ch": pulse_cfg.ch}
        )
        signals_buffer = reset_scan_buffer(
            context.plots,
            lengths,
            xlabel="Flux reset length (us)",
            branch_shape=(2, 2),
            signal2real=lambda data: data.real.reshape(4, -1),
        )

        # Each hold length has the Cartesian product of the two calibrated pi switches.
        with Schedule( signals_buffer, scope=context.scope) as sched:
            builder = TimelineBuilder(
                soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay
            )
            builder.add_loop("before_pi", 2)
            builder.add_loop("after_pi", 2)
            length = builder.add_sweep("length", cfg.length)
            if cfg.reset is not None:
                builder.emit(Reset("reset", cfg.reset))
            # Keep the selected pi duration; the off case must not gain idle time.
            builder.delay_by_timeline()
            with (
                builder.branch(by="before_pi", align_timeline=False) as cases,
                cases.case(1),
            ):
                builder.emit(Pulse("pi_pulse", cfg.pi_pulse))
            builder.delay_by_timeline()
            builder.emit(PulseReset("tested_reset", cfg.tested_reset, length=length))
            builder.delay_by_timeline()
            with (
                builder.branch(by="after_pi", align_timeline=False) as cases,
                cases.case(1),
            ):
                builder.emit(Pulse("pi_pulse", cfg.pi_pulse))
            builder.delay_by_timeline()
            builder.emit(Readout("readout", cfg.readout))
            sched.run_acquire(builder.build(), soc, rounds=cfg.rounds)

        return LengthResult(lengths=lengths, signals=signals_buffer.array)

    def analyze(
        self,
        source: RunRecord[LengthCfg, LengthResult],
        options: LengthAnalyzeOptions,
        *,
        plots: Plots,
    ) -> LengthAnalysis:
        """Estimate residual excitation and select the shortest qualifying duration.

        Use d0=S00-S01, d1=S10-S11 and r=Re(d1*conj(d0))/|d0|².
        Under calibrated ideal pi, identical readout and a two-state model,
        p1=(1-r)/2+r*p0 exactly. Four IQs cannot identify p0 separately.
        Return the unclipped estimate (1-r)/2 assuming reset hardly heats g
        (p0≈0). A thermally excited idle point, such as fluxonium near half
        flux, invalidates this approximation. The fixed offline validation
        model has idle first-excited-state population about 0.358. Only after
        an initial reset makes p0 small, with the other assumptions satisfied,
        can the estimate represent absolute residual excitation.
        Offline validation checks run, save, restore, analyze, PNG output and
        selection/no-solution handling, not physical reset effectiveness.
        This cannot exclude leakage or establish reset fidelity.
        Zero d0 or nonfinite IQ/contrasts/estimates are not candidates. Select the
        shortest quantized length with residual ≤ options.epsilon only if every
        longer measured finite point also passes. Scan order is irrelevant.
        No finite or qualifying candidate gives selected_length=None.
        Publish raw I/Q, r and residual curves. source.cfg may be None.
        """
        if not np.isfinite(options.epsilon):
            raise ValueError("Epsilon must be finite")
        result = source.result
        lengths, signals = result.lengths, result.signals
        if lengths.ndim != 1 or signals.shape != (2, 2, lengths.size):
            raise ValueError("Flux reset IQ must have shape (2, 2, number of lengths)")
        if not np.array_equal(result.before_pi, [0, 1]) or not np.array_equal(
            result.after_pi, [0, 1]
        ):
            raise ValueError("Both pi switch axes must be [0, 1] (off, on)")

        # After-pi differences cancel the common readout offset, not p0.
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            d0 = signals[0, 0] - signals[0, 1]
            d1 = signals[1, 0] - signals[1, 1]
            ratio = np.real(d1 * np.conj(d0)) / np.abs(d0) ** 2
            residual = (1 - ratio) / 2
        valid = (
            np.isfinite(lengths)
            & np.all(np.isfinite(signals), axis=(0, 1))
            & np.isfinite(d0)
            & np.isfinite(d1)
            & (d0 != 0)
            & np.isfinite(ratio)
            & np.isfinite(residual)
        )
        ratio = np.where(valid, ratio, np.nan)
        residual = np.where(valid, residual, np.nan)
        candidates = np.flatnonzero(valid)
        qualifying = [
            i
            for i in candidates
            if residual[i] <= options.epsilon
            and np.all(residual[valid & (lengths > lengths[i])] <= options.epsilon)
        ]
        selected = float(np.min(lengths[qualifying])) if qualifying else None

        # Keep the measured branches visible beside the uncalibrated estimate.
        fig, _ = plots.subplots(
            "fit", nrows=4, sharex=True, figsize=(10, 10), layout="constrained"
        )
        axes = fig.axes
        for a in range(2):
            for b in range(2):
                label = f"Before pi={a}, after pi={b}"
                axes[0].plot(lengths, signals[a, b].real, ".-", label=label)
                axes[1].plot(lengths, signals[a, b].imag, ".-", label=label)
        axes[2].plot(lengths, ratio, ".-")
        axes[3].plot(lengths, residual, ".-", label="p0≈0 estimate (unclipped)")
        axes[3].axhline(
            options.epsilon,
            color="black",
            linestyle=":",
            label=f"ε = {options.epsilon:g}",
        )
        axes[0].set_ylabel("I (a.u.)")
        axes[1].set_ylabel("Q (a.u.)")
        axes[2].set_ylabel("Contrast ratio r")
        axes[3].set_ylabel("Residual estimate")
        axes[3].set_xlabel("Flux reset length (us)")
        axes[0].legend(fontsize=8)
        axes[3].legend(fontsize=8)
        axes[0].set_title(
            "No qualifying finite length; no writeback"
            if selected is None
            else f"Selected length = {selected:.6g} us | p0≈0; not reset fidelity",
            fontsize=10,
        )
        for ax in axes:
            ax.grid(True)
            if selected is not None:
                ax.axvline(selected, color="gray", linestyle="--")
        return LengthAnalysis(d0, d1, ratio, residual, selected, options.epsilon)
