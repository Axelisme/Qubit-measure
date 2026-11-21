"""Acquisition, analysis and persistence for zipped mux amplitude Rabi.

Technical reference for lab authors and framework developers.
MuxAmpRabiExp.run(config, *, context) sweeps participant gains simultaneously.

Configuration and timing
------------------------
The full tones list matches the complete MuxReadoutCfg in count and order.
Each MuxAmpRabiToneCfg has a fixed name, a required drive and gain bounds. Participants is
a nonempty, unique list of original indices. Q3/Q1 selects original indices [2, 0]
from Q1/Q2/Q3 and retains that execution/data order. Every drive must be valid,
only for participants, on independent generators distinct from readout.
The caller supplies the complete qubit/resonator bindings; numeric cfg validation
does not resolve component identities.

Sweep ranges may differ; cfg.expts supplies one positive point count shared by
selected tones. Point p pairs each active tone's p-th value,
not a Cartesian product. One program advances all gains in a shared hardware loop.
Each point executes optional common reset, parallel emission of independent drive pulses,
then active mux readout. Branches start together; short branches wait for the
longest branch at that point. Pre/post delays stay
unchanged. There is no end alignment, end padding or sweep-wide maximum padding.
All participants share reps, rounds, initial_delay and relax_delay.
Program modules perform native conversion and averaging.

Results and cancellation
------------------------
MuxSweepResult.tone_indices is int64 (K,) in participant order. Coordinates are
float64 (K,P) on each drive's actual hardware-quantized gain grid. Signals are
complex128 (K,P), ADC-length-normalized average IQ in a.u. The measurement_i magnitude plots use
original tone indices and each tone's own coordinates, so duplicate labels retain
distinct plots. Cancellation preserves completed-round averages across the full
sweep. If no round completes, all signals remain NaN.

Amplitude scan and analysis
--------------------------
MuxAmpRabiCfg.reset is optional; MuxAmpRabiCfg.readout is the full mux readout.
Sweep bounds are DAC full-scale fractions in a.u., within [-1, 1]. prepare_amplitude_scan copies
cfg and builds independently rounded scalar gain grids. run publishes the
compiled integer-step gain grids instead.

Analysis uses the single-qubit undamped Rabi fitter and zero-gain phase policy.
Options.skip removes leading zipped points before finite-IQ filtering. Each
curve needs six remaining points. Each original index returns MuxAmpRabiFit or
MuxFitFailure with a reason, and a fit_i figure. Pi/pi2 gains and errors use
DAC full-scale fractions in a.u. within [-1, 1]. FitQuality describes the numerical fit; a finite fit
alone does not establish calibration quality.

Native persistence retains the full cfg, original indices, independent gain axes
and complex IQ. The source pulse and its scalar length remain available to a
calibration caller for pulse-copy and pi-product proposals.

Calling and ownership
---------------------
For context, plots, stop/error handling and record/persistence ownership, see
lib/zcu_tools/experiment/README.md, section "實驗介面與資料", and
lib/zcu_tools/experiment/v2/runtime/README.md, section "Stop 與 Retry".

Analysis uses the explicit historical source, not the current Form or library.
The numerical core leaves component bindings and writeback to its caller.

Example with validated cfg, RunContext and fresh analysis_plots::

    from zcu_tools.experiment.records import RunRecord
    from zcu_lab.v2.mux.amp_rabi import MuxAmpRabiExp

    core = MuxAmpRabiExp()
    result = core.run(cfg, context=context)
    source = RunRecord(cfg=cfg, result=result)
    outcomes = core.analyze(source, core.Options(), plots=analysis_plots)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import Annotated, ClassVar

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, StrictInt, model_validator
from pydantic.dataclasses import dataclass as validated_dataclass
from qick import QickConfig
from zcu_tools.analysis.failure import AnalysisFailed
from zcu_tools.analysis.fitting import FitQuality, compute_fit_quality, fit_rabi
from zcu_tools.cfg_model import ConfigBase, FieldMeta, Gain
from zcu_tools.experiment import Experiment
from zcu_tools.experiment.axes_spec import GroupedAxesSpec
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.utils.sweep import make_sweep
from zcu_tools.experiment.v2.runtime import RunOptions, Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import round_zcu_gain, sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import Reset, ResetCfg, TimelineBuilder
from zcu_tools.program.v2.modules import (
    MuxReadout,
    MuxReadoutCfg,
    Pulse,
    PulseCfg,
)
from zcu_tools.program.v2.utils import sweep2param
from zcu_tools.utils.process import rotate2real

from zcu_lab.v2.support import live_line
from zcu_lab.v2.support.mux import (
    MuxFitFailure,
    MuxSweepResult,
    averaged_mux_signal,
    validate_mux_result,
)
from zcu_lab.v2.support.mux_persistence import mux_sweep_spec
from zcu_lab.v2.support.mux_sweep_cfg import MuxDriveToneCfg, validate_mux_scan


@validated_dataclass(frozen=True)
class AmpRabiAnalyzeOptions:
    """Skip leading gain points before the undamped per-tone Rabi fit."""

    skip: Annotated[int, FieldMeta(label="Skip points"), Field(ge=0)] = 0


@dataclass(frozen=True)
class MuxAmpRabiFit:
    """Original tone index, pi/pi2 DAC gains in a.u. and one-sigma errors.

    quality describes the undamped cosine fit with the single-qubit owner's
    zero-gain phase choice. Gains are DAC full-scale fractions in a.u., within
    [-1, 1]. Errors use the same a.u. scale.
    """

    index: int
    pi_gain: float
    pi_gain_err: float
    pi2_gain: float
    pi2_gain_err: float
    quality: FitQuality


@dataclass(frozen=True)
class MuxPulseScan:
    """Prepared numeric zip data, without binding or acquisition state.

    indices is the once-lowered active original index sequence, in full tone
    order. coordinates is float64 (K,P), using each drive's own hardware grid.
    point_pulses is (P,K) fixed native PulseCfg values in matching order; these
    retain nominal requested parameters for program-owned native conversion.
    This record does not choose a scheduler or acquire. Treat values as read-only.
    """

    indices: Sequence[int]
    coordinates: NDArray[np.float64]
    point_pulses: Sequence[Sequence[PulseCfg]]


class ToneRangeCfg(ConfigBase):
    """One tone's first and last scan gains in a.u., each within [-1, 1]."""

    start: Gain = Field(description="First scan value in experiment units.")
    stop: Gain = Field(description="Last scan value in experiment units.")


class MuxAmpRabiToneCfg(MuxDriveToneCfg):
    """One named drive and its independent gain bounds in gain a.u.."""

    gain: Annotated[ToneRangeCfg | None, FieldMeta(unit="")] = None

    def require_gain(self) -> ToneRangeCfg:
        """Return active gain bounds, or raise ValueError naming this tone."""
        if self.gain is None:
            raise ValueError(f"{self.name}: active tone requires gain")
        return self.gain


class MuxAmpRabiCfg(ConfigBase):
    """Complete named tones, ordered participants and shared acquisition controls.

    reset and readout hold the common modules. tones contains one
    MuxAmpRabiToneCfg per original readout position, with unique nonempty names.
    participants is a nonempty, duplicate-free list of original tone indices; its order
    determines execution/data rows (Q3/Q1 maps to original indices 2/0).
    Selected drives are required/validated on independent
    generators distinct from readout. Each supplied gain range is in a.u. within [-1, 1];
    expts is the positive hardware point count shared by all tones.
    Explicit reps/rounds are positive; initial_delay/relax_delay are finite
    nonnegative us. No capture, bindings or writeback destinations are stored.
    """

    reset: ResetCfg | None = None
    readout: MuxReadoutCfg
    tones: Annotated[list[MuxAmpRabiToneCfg], FieldMeta(element_label="name")]
    expts: int = Field(
        strict=True,
        gt=0,
        title="Expts",
        description="Hardware scan point count shared by all tones.",
    )
    participants: list[StrictInt]

    reps: Annotated[
        int, FieldMeta(label="Reps", description="Number of repetitions per round.")
    ] = Field(gt=0)
    rounds: Annotated[
        int,
        FieldMeta(label="Rounds", description="Number of software acquisition rounds."),
    ] = Field(default=1, gt=0)
    initial_delay: Annotated[
        float,
        FieldMeta(
            label="Initial delay",
            unit="us",
            description="Delay before the first repetition.",
        ),
    ] = Field(default=1.0, ge=0, allow_inf_nan=False)
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relax delay",
            unit="us",
            decimals=3,
            description="Relaxation delay after each repetition.",
        ),
    ] = Field(ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_scan(self) -> MuxAmpRabiCfg:
        """Validate shared controls and require a gain range for each participant."""
        validate_mux_scan(self, self.tones, self.participants, self.expts)
        for index in self.participants:
            self.tones[index].require_gain()
        return self


def prepare_amplitude_scan(cfg: MuxAmpRabiCfg, *, soccfg: QickConfig) -> MuxPulseScan:
    """Prepare simultaneous gain grids; propagate board conversion errors.

    cfg must already be validated and supplies complete tones and ordered
    participants. It is deep-copied before use.
    soccfg describes the target board. Each participant gets its own normalized
    gain coordinates and one pulse per point, rounded as independent scalar
    pulses. Sweeps advance together by point index, not as a Cartesian product.
    Pulses retain nominal gains for compilation. run replaces these preliminary
    coordinates with the compiled hardware sweep's integer-step gain grid.
    Board conversion errors propagate.

    Return MuxPulseScan without acquiring data or modifying the input cfg.
    """
    cfg = cfg.model_copy(deep=True)
    indices = list(cfg.participants)
    coordinates = []
    points = [[] for _ in range(cfg.expts)]
    for index in indices:
        tone = cfg.tones[index]
        drive = tone.require_drive()
        nominal = sweep2array(
            make_sweep(tone.require_gain().start, tone.require_gain().stop, cfg.expts),
            "none",
        )
        actual = round_zcu_gain(nominal, soccfg, drive.ch)
        coordinates.append(actual)
        for point, value in enumerate(nominal):
            points[point].append(drive.with_updates(gain=float(value)))
    return MuxPulseScan(indices, np.asarray(coordinates, dtype=np.float64), points)


class MuxAmpRabiExp(Experiment[MuxSweepResult, MuxAmpRabiCfg]):
    """Measure each participant's Rabi oscillation versus drive gain.

    Reset precedes simultaneous drives and a shared mux readout at each point.
    Fits give pi and pi/2 gains. Results retain each tone's actual gain grid,
    complex IQ and original index; saved data also retains the complete cfg.
    """

    AXES_SPEC: ClassVar[GroupedAxesSpec[MuxSweepResult, MuxAmpRabiCfg]] = (
        mux_sweep_spec(
            MuxAmpRabiCfg,
            tag="mux/rabi/amp",
            coordinate_label="Pulse gain",
            unit="",
            scale=1.0,
        )
    )

    Options: ClassVar[type[AmpRabiAnalyzeOptions]] = AmpRabiAnalyzeOptions

    def run(self, config: MuxAmpRabiCfg, *, context: RunContext) -> MuxSweepResult:
        """Sweep participant gains together and measure their averaged complex IQ.

        config supplies reset, readout, drives, gain sweeps and averaging controls;
        it is copied before use. After reset, all drives start together. Readout
        waits for the longest drive at that point, without end-alignment padding.
        context supplies the board, plots and stop signal. Coordinates and IQ have
        shape (participants, points), in participant order, with actual hardware-
        quantized gains in a.u. within [-1, 1]. All gains advance in one shared hardware loop.
        Each completed round updates the full sweep's live plots. Cancellation
        keeps completed-round averages, or NaN if no round completes.
        Acquisition and cfg/build errors propagate. The caller finishes/releases
        plots and decides whether returned data can be analyzed.

        """
        # Prepare each gain grid and select readout tones in participant order.
        # 1. Pair quantized gain grids with the selected readout tones in participant order.
        cfg = config.model_copy(deep=True)
        scan = prepare_amplitude_scan(cfg, soccfg=context.soccfg)
        readout = cfg.readout.with_updates(
            tones=[cfg.readout.tones[index] for index in scan.indices]
        )

        # Keep live traces on each drive's own quantized gain axis.
        viewers = []
        for index in scan.indices:
            fig, ax = context.plots.subplots(num=f"measurement_{index}")
            ax.set(
                xlabel="Pulse gain",
                ylabel="Magnitude (a.u.)",
                title=f"{index}: {cfg.tones[index].name}",
            )
            viewers.append(live_line(ax))
            fig.show()

        @context.plots.on_plot_thread
        def update(data: NDArray[np.complex128]):
            return tuple(
                artist
                for row, viewer in enumerate(viewers)
                for artist in viewer.set(scan.coordinates[row], np.abs(data[:, row]))
            )

        # Advance each drive gain in the same tProc point loop.
        # 2. Sweep gains in one hardware loop with simultaneous drives after reset.
        signals = SignalBuffer(scan.coordinates.T.shape, on_update=update)
        with Schedule(signals, scope=context.scope) as sched:
            drives = []
            for index in scan.indices:
                tone = cfg.tones[index]
                gain = sweep2param(
                    "point",
                    make_sweep(
                        tone.require_gain().start, tone.require_gain().stop, cfg.expts
                    ),
                )
                drives.append(
                    Pulse(
                        f"drive_{index}",
                        tone.require_drive(),
                        gain=gain,
                        pulse_id=f"drive_{index}",
                    )
                )

            # Reset -> simultaneous drives -> fixed readout after the longest drive.
            builder = TimelineBuilder(
                context.soccfg,
                reps=cfg.reps,
                initial_delay=cfg.initial_delay,
                relax_delay=cfg.relax_delay,
            )
            builder.add_loop("point", cfg.expts)
            if cfg.reset is not None:
                builder.emit(Reset("reset", cfg.reset))
            timelines = [f"drive_{index}" for index in scan.indices]
            builder.fork("default", into=timelines)
            for drive, timeline in zip(drives, timelines, strict=True):
                builder.emit(drive, timeline=timeline)
            builder.join(*timelines, into="default")
            readout_module = MuxReadout("readout", readout)
            builder.emit(readout_module)

            # Integer register steps can differ from independently rounded points.
            # Keep live plots, analysis and persistence on the compiled gain grid.
            prog = builder.build()
            for row, index in enumerate(scan.indices):
                scan.coordinates[row] = prog.get_pulse_param(
                    f"drive_{index}", "gain", as_array=True
                )
            _ = sched.run_acquire(
                prog,
                context.soc,
                rounds=cfg.rounds,
                options=RunOptions(
                    converter=partial(averaged_mux_signal, prog, readout_module)
                ),
            )

        # Publish final plots and return tone-major data, including unmeasured NaNs.
        signals.trigger_update(flush=True)
        # 3. Preserve compiled gain coordinates and completed-round IQ; NaN marks no completed round.
        return MuxSweepResult(
            np.asarray(scan.indices, dtype=np.int64),
            scan.coordinates,
            signals.array.T.copy(),
        )

    def analyze(
        self,
        source: RunRecord[MuxAmpRabiCfg, MuxSweepResult],
        options: AmpRabiAnalyzeOptions,
        *,
        plots: Plots,
    ) -> tuple[MuxAmpRabiFit | MuxFitFailure, ...]:
        """Fit each tone's IQ oscillation to obtain pi and pi/2 gains.

        source contains the acquired gain grids and IQ; its cfg may be None.
        options.skip omits leading points. Return gains, errors and fit quality,
        or a failure reason per original tone index. plots receives fit_i figures.

        """
        # 1. Validate saved gain/IQ axes before calibrating the original participant tones.
        result = source.result
        validate_mux_result(result)
        outcomes: list[MuxAmpRabiFit | MuxFitFailure] = []
        # 2. Analyze each tone on its own acquired grid after the requested leading-point skip.
        for row, raw_index in enumerate(result.tone_indices):
            index = int(raw_index)
            signal = result.signals[row, options.skip :]
            finite = np.isfinite(signal)
            xs = result.coordinates[row, options.skip :][finite]
            signal = signal[finite]
            fig, ax = plots.subplots(num=f"fit_{index}", figsize=(8, 5))
            ax.set(xlabel="Pulse gain (a.u.)", ylabel="Signal (a.u.)")
            fig.suptitle(f"tone {index}")
            try:
                if len(xs) < 6:
                    raise ValueError(
                        "Amp Rabi fit requires at least six acquired points"
                    )
                # 3. Project IQ and fix the undamped Rabi phase from the acquired gain nearest zero.
                real = rotate2real(signal).real
                zero = real[np.argmin(np.abs(xs))]
                phase = 0.0 if zero > 0.5 * (np.max(real) + np.min(real)) else 180.0
                fit = fit_rabi(xs, real, decay=False, init_phase=phase)
                pi, pierr = fit.pi_x, fit.pi_x_err
                pi2, pi2err = fit.pi2_x, fit.pi2_x_err
                fitted, params, covariance = fit.curve, fit.params, fit.cov
                if not np.all(np.isfinite((pi, pierr, pi2, pi2err))):
                    raise ValueError("Amp Rabi fit returned nonfinite estimates")
                names = ("y0", "yscale", "freq", "phase")
                quality = compute_fit_quality(
                    real,
                    fitted,
                    {
                        name: float(value)
                        for name, value in zip(names, params, strict=True)
                    },
                    covariance,
                )
                outcome = MuxAmpRabiFit(index, pi, pierr, pi2, pi2err, quality)
            except (
                AnalysisFailed,
                ValueError,
                RuntimeError,
                ArithmeticError,
                np.linalg.LinAlgError,
            ) as exc:
                # 4. Isolate each tone's fit failure and show its acquired magnitude for diagnosis.
                ax.plot(xs, np.abs(signal), "o", label="acquired magnitude")
                ax.text(
                    0.02, 0.98, str(exc), transform=ax.transAxes, va="top", wrap=True
                )
                outcomes.append(MuxFitFailure(index, str(exc)))
            else:
                # 5. Mark pi/pi2 gains and uncertainty bands on the projected Rabi fit.
                ax.plot(xs, real, "o", label="measured")
                ax.plot(xs, fitted, label="fit")
                for value, error, label in ((pi, pierr, "pi"), (pi2, pi2err, "pi/2")):
                    ax.axvline(
                        value, ls="--", label=f"{label}={value:.4g} ± {error:.2g}"
                    )
                    ax.axvspan(value - error, value + error, alpha=0.2, color="red")
                outcomes.append(outcome)
            ax.legend()
            ax.grid(True)
            fig.tight_layout()
        # 6. Keep pi/pi2 gains or a failure reason under each original tone identity.
        return tuple(outcomes)
