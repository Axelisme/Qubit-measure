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
    GainSweep,
    Pulse,
    PulseReset,
    Readout,
    Reset,
    TimelineBuilder,
)

from zcu_lab.v2.support.reset import reset_scan_buffer


@dataclass(frozen=True)
class GainResult:
    """Quantized gains (a.u.) and complex IQ shaped (before_pi, after_pi, gain).

    before_pi and after_pi each store [0, 1]: calibrated pi off, then on.
    signals[a, b, i] is mean IQ at gains[i] with those two switches.
    Unacquired values remain NaN.
    """

    gains: NDArray[np.float64]
    signals: NDArray[np.complex128]
    before_pi: NDArray[np.int64] = field(
        default_factory=lambda: np.array([0, 1], dtype=np.int64)
    )
    after_pi: NDArray[np.int64] = field(
        default_factory=lambda: np.array([0, 1], dtype=np.int64)
    )


@dataclass(frozen=True)
class GainAnalysis:
    """Four-branch contrasts and an unclipped residual estimate per gain.

    contrast_ground is d0=S00-S01; contrast_excited is d1=S10-S11, in IQ units.
    contrast_ratio is Re(d1*conj(d0))/|d0|², NaN where undefined.
    residual_estimate=(1-r)/2 assumes p0≈0; it is not reset fidelity.
    selected_gain is the minimum finite estimate's gain (a.u.), first in scan
    order for ties, or None when no finite candidate exists.
    """

    contrast_ground: NDArray[np.complex128]
    contrast_excited: NDArray[np.complex128]
    contrast_ratio: NDArray[np.float64]
    residual_estimate: NDArray[np.float64]
    selected_gain: float | None


class GainCfg(ConfigBase):
    """Sweep a const flux reset's gain between two calibrated pi switches."""

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
            description="Zero-frequency const flux pulse at fixed length; gain is swept.",
        ),
    ]
    readout: Annotated[Readout.Cfg, FieldMeta(label="Readout")]
    gain: Annotated[GainSweep, FieldMeta(label="Gain", unit="a.u.")]

    reps: Annotated[
        int,
        FieldMeta(label="Reps", description="Repetitions per branch/gain per round."),
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


class GainExp(Experiment[GainResult, GainCfg]):
    AXES_SPEC: ClassVar[AxesSpec[GainResult, GainCfg]] = AxesSpec(
        axes=(
            AxesSpec.Axis("gains", label="Flux reset gain", unit="a.u."),
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
        result_type=GainResult,
        cfg_type=GainCfg,
        cfg_schema_version="3.0",
        tag="twotone/reset/flux/gain",
    )

    def run(self, cfg: GainCfg, *, context: RunContext) -> GainResult:
        """Measure four mean-IQ branches per flux gain at fixed duration.

        In one program apply optional reset, before-reset pi off/on, the tested
        flux pulse, after-reset pi off/on, and the same readout. Both switches
        use pi_pulse unchanged. Keep flux duration and pre/post delays fixed.
        Return rounded gains (a.u.) and IQ shaped (2, 2, N), switches [0, 1].
        Reject non-const or nonzero-frequency flux pulses before acquisition.
        Live curves show I only, without choosing a population direction.
        """
        cfg = deepcopy(cfg)
        pulse_cfg = cfg.tested_reset.pulse_cfg
        if pulse_cfg.waveform.style != "const" or pulse_cfg.freq != 0.0:
            raise ValueError("Flux reset requires a const pulse at zero frequency.")

        soc, soccfg = context.soc, context.soccfg
        gains = sweep2array(
            cfg.gain, "gain", {"soccfg": soccfg, "gen_ch": pulse_cfg.ch}
        )
        signals_buffer = reset_scan_buffer(
            context.plots,
            gains,
            xlabel="Flux reset gain (a.u.)",
            branch_shape=(2, 2),
            signal2real=lambda data: data.real.reshape(4, -1),
        )

        # Each gain has the Cartesian product of the two calibrated pi switches.
        with Schedule(None, signals_buffer, scope=context.scope) as sched:
            builder = TimelineBuilder(
                soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay
            )
            builder.add_loop("before_pi", 2)
            builder.add_loop("after_pi", 2)
            gain = builder.add_sweep("gain", cfg.gain)
            if cfg.reset is not None:
                builder.emit(Reset("reset", cfg.reset))
            # Keep the selected pi duration; the off case must not gain idle time.
            builder.sync()
            with (
                builder.branch(by="before_pi", align_timeline=False) as cases,
                cases.case(1),
            ):
                builder.emit(Pulse("pi_pulse", cfg.pi_pulse))
            builder.sync()
            builder.emit(PulseReset("tested_reset", cfg.tested_reset, gain=gain))
            builder.sync()
            with (
                builder.branch(by="after_pi", align_timeline=False) as cases,
                cases.case(1),
            ):
                builder.emit(Pulse("pi_pulse", cfg.pi_pulse))
            builder.sync()
            builder.emit(Readout("readout", cfg.readout))
            sched.run_acquire(builder.build(), soc, rounds=cfg.rounds)

        return GainResult(gains=gains, signals=signals_buffer.array)

    def analyze(
        self,
        source: RunRecord[GainCfg, GainResult],
        options: None,
        *,
        plots: Plots,
    ) -> GainAnalysis:
        """Estimate residual excitation and select its minimum finite gain.

        Use d0=S00-S01, d1=S10-S11 and r=Re(d1*conj(d0))/|d0|².
        Under calibrated ideal pi, identical readout and a two-state model,
        p1=(1-r)/2+r*p0 exactly. Four IQs cannot identify p0 separately.
        Return the unclipped estimate (1-r)/2 assuming reset hardly heats g
        (p0≈0). A thermally excited idle point, such as fluxonium near half
        flux, invalidates this approximation. The fixed offline model used for
        validation has idle first-excited-state population about 0.358. Only
        after an initial reset makes p0 small can the estimate represent absolute
        residual excitation, with the other assumptions still required.
        Offline validation checks run/save/restore/analyze/PNG and selection,
        not physical reset effectiveness. Do not use model populations to
        calibrate this estimate. It cannot exclude leakage or establish fidelity.
        Zero d0 or nonfinite IQ/contrasts/estimates are not candidates. Ties
        choose the first gain in scan order; no candidate gives selected_gain=None.
        Publish raw I/Q, r and residual curves. source.cfg may be None.
        """
        del options
        result = source.result
        gains, signals = result.gains, result.signals
        if gains.ndim != 1 or signals.shape != (2, 2, gains.size):
            raise ValueError("Flux reset IQ must have shape (2, 2, number of gains)")
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
            np.isfinite(gains)
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
        selected = (
            float(gains[candidates[np.argmin(residual[candidates])]])
            if candidates.size
            else None
        )

        # Keep the measured branches visible beside the uncalibrated estimate.
        fig, _ = plots.subplots(
            "fit", nrows=4, sharex=True, figsize=(10, 10), layout="constrained"
        )
        axes = fig.axes
        for a in range(2):
            for b in range(2):
                label = f"Before pi={a}, after pi={b}"
                axes[0].plot(gains, signals[a, b].real, ".-", label=label)
                axes[1].plot(gains, signals[a, b].imag, ".-", label=label)
        axes[2].plot(gains, ratio, ".-")
        axes[3].plot(gains, residual, ".-", label="p0≈0 estimate (unclipped)")
        axes[0].set_ylabel("I (a.u.)")
        axes[1].set_ylabel("Q (a.u.)")
        axes[2].set_ylabel("Contrast ratio r")
        axes[3].set_ylabel("Residual estimate")
        axes[3].set_xlabel("Flux reset gain (a.u.)")
        axes[0].legend(fontsize=8)
        axes[3].legend(fontsize=8)
        axes[0].set_title(
            "No finite candidate; no writeback"
            if selected is None
            else f"Selected gain = {selected:.6g} a.u. | p0≈0; not reset fidelity",
            fontsize=10,
        )
        for ax in axes:
            ax.grid(True)
            if selected is not None:
                ax.axvline(selected, color="gray", linestyle="--")
        return GainAnalysis(d0, d1, ratio, residual, selected)
