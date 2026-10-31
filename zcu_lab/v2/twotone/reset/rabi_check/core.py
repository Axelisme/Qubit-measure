from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Annotated, ClassVar

import numpy as np
from matplotlib.artist import Artist
from numpy.typing import NDArray
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment, config
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
    TimelineBuilder,
)
from zcu_tools.utils.process import find_rotate_angle

from zcu_lab.v2.support import live_line
from zcu_lab.v2.twotone.reset.rabi_check.fit import RabiCheckFit, fit_reset_rabi


@dataclass(frozen=True)
class RabiCheckResult:
    gains: NDArray[np.float64]
    signals: NDArray[np.complex128]
    reset_states: NDArray[np.int64] = field(
        default_factory=lambda: np.array([0, 1, 2], dtype=np.int64)
    )


def reset_rabi_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    # Anchor all branches to the before-reset IQ axis. Branch offsets must not
    # rotate the projection or independently rescale the measured contrasts.
    try:
        angle = find_rotate_angle(signals[0])
    except ValueError:
        # Live buffers can be empty before the first acquisition completes.
        angle = 0.0
    return (signals * np.exp(-1j * angle)).real


class RabiCheckCfg(ConfigBase):
    reset: Annotated[Reset.Cfg | None, FieldMeta(label="Reset")] = None
    rabi_pulse: Annotated[Pulse.Cfg, FieldMeta(label="Rabi pulse")]
    tested_reset: Annotated[Reset.Cfg, FieldMeta(label="Tested reset")]
    readout: Annotated[Readout.Cfg, FieldMeta(label="Readout")]
    gain: Annotated[GainSweep, FieldMeta(label="Gain", unit="a.u.")]

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


class RabiCheckExp(Experiment[RabiCheckResult, RabiCheckCfg]):
    AXES_SPEC: ClassVar[AxesSpec[RabiCheckResult, RabiCheckCfg]] = AxesSpec(
        axes=(
            AxesSpec.Axis("gains", label="Amplitude", unit="a.u.", dtype=np.float64),
            AxesSpec.Axis("reset_states", label="Reset", unit="None", dtype=np.int64),
        ),
        z=AxesSpec.Z("signals", label="Signal", unit="a.u.", dtype=np.complex128),
        result_type=RabiCheckResult,
        cfg_type=RabiCheckCfg,
        cfg_schema_version="3.0",
        tag="twotone/reset/rabi_check",
    )

    def run(self, cfg: RabiCheckCfg, *, context: RunContext) -> RabiCheckResult:
        """Compare averaged IQ before reset, after reset, and after reset + Rabi.

        Prepare with the optional reset, then sweep the Rabi pulse gain. Select
        no tested reset, tested reset alone, or tested reset followed by the same
        swept pulse, and read out. Return quantized gains and mean complex IQ
        with shape (3, number of gains), in that branch order. These curves
        describe relative contrast, not reset fidelity; unacquired data stay NaN.
        """
        # 1. Prepare the quantized gain axis without changing the caller's cfg.
        cfg = deepcopy(cfg)
        soc, soccfg = context.soc, context.soccfg

        gains = sweep2array(
            cfg.gain,
            "gain",
            {"soccfg": soccfg, "gen_ch": cfg.rabi_pulse.ch},
        )

        # 2. Project all three mean-IQ curves onto the before-reset IQ axis.
        plots = context.plots
        fig, ax = plots.subplots(num="measurement")
        ax.set(xlabel="Pulse gain", ylabel="Amplitude")
        viewer = live_line(ax, num_lines=3)

        @plots.on_plot_thread
        def update(data: NDArray[np.complex128]) -> tuple[Artist, ...]:
            return viewer.set(gains, reset_rabi_signal2real(data))

        fig.show()
        signals_buffer = SignalBuffer((3, len(gains)), on_update=update)

        with Schedule(None, signals_buffer, scope=context.scope) as sched:
            builder = TimelineBuilder(
                soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay
            )
            builder.add_loop("reset_sel", 3)
            gain = builder.add_sweep("gain", cfg.gain)
            builder.emit(Reset("reset", cfg.reset))
            builder.emit(Pulse("rabi_pulse", cfg.rabi_pulse, gain=gain))
            # Preserve each selected case's duration instead of padding to the maximum.
            builder.sync()
            with builder.branch(by="reset_sel", align_timeline=False) as cases:
                with cases.case(1):
                    builder.emit(Reset("tested_reset", cfg.tested_reset))
                with cases.case(2):
                    builder.emit(Reset("tested_reset", cfg.tested_reset))
                    builder.emit(Pulse("rabi_pulse", cfg.rabi_pulse, gain=gain))
            builder.sync()
            builder.emit(Readout("readout", cfg.readout))
            sched.run_acquire(builder.build(), soc, rounds=cfg.rounds)

        return RabiCheckResult(gains, signals_buffer.array)

    def analyze(
        self,
        source: RunRecord[RabiCheckCfg, RabiCheckResult],
        options: None,
        *,
        plots: Plots,
    ) -> RabiCheckFit:
        """Return descriptive contrast fits and publish fit, without reset fidelity.

        Frequency is fitted only to the before-reset branch. All branches share
        its IQ projection and frequency; after-reset also includes a 2f term.
        """
        del options
        result = source.result
        gains, signals = result.gains, result.signals
        if gains.ndim != 1 or signals.shape != (3, gains.size):
            raise ValueError("Reset-check signals must have shape (3, number of gains)")
        if not np.array_equal(result.reset_states, [0, 1, 2]):
            raise ValueError("Reset-check branch order must be [0, 1, 2]")
        if np.any(np.isinf(signals)):
            raise ValueError("Reset-check signals must not contain infinities")
        real_signals = reset_rabi_signal2real(signals)
        fit = fit_reset_rabi(gains, real_signals)
        fig, _ = plots.subplots(
            "fit",
            nrows=2,
            ncols=1,
            sharex=True,
            figsize=(max(config.figsize[0], 9), max(config.figsize[1], 6)),
            gridspec_kw={"height_ratios": [3, 1]},
            layout="constrained",
        )
        ax, residual_ax = fig.axes
        dense_gains = np.linspace(np.min(gains), np.max(gains), 600)
        branches = (fit.before, fit.reset, fit.after)
        labels = ("Before reset", "Reset only", "Reset + Rabi")
        for index, (branch, label) in enumerate(zip(branches, labels, strict=True)):
            color = f"C{index}"
            ax.plot(gains, real_signals[index], ".", color=color, label=label)
            ax.plot(
                dense_gains, branch.evaluate(dense_gains, fit.frequency), color=color
            )
            residual_ax.plot(
                gains,
                real_signals[index] - branch.evaluate(gains, fit.frequency),
                ".",
                color=color,
                label=f"{label}: RMS={branch.residual_rms:.3g}",
            )
        ax.set_title(
            f"Reset Rabi check | f = {fit.frequency:.5g} cycles/gain\n"
            f"A before = {fit.before.amplitude:.4g}, A after = {fit.after.amplitude:.4g}, "
            f"relative contrast = {fit.contrast_ratio:.4g}\n"
            f"Reset residual amplitude = {fit.reset.amplitude:.3g}, "
            f"after 2f amplitude = {fit.after.second_harmonic_amplitude:.3g}",
            fontsize=11,
        )
        ax.set_ylabel("Projected IQ (a.u.)")
        residual_ax.set_xlabel("Pulse gain (a.u.)")
        residual_ax.set_ylabel("Residual (a.u.)")
        residual_ax.axhline(0, color="gray", linewidth=0.8)
        for axes in (ax, residual_ax):
            axes.legend(fontsize=9)
            axes.grid(True)
        return fit
