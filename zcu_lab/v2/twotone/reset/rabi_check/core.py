from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar

import numpy as np
from matplotlib.artist import Artist
from numpy.typing import NDArray
from zcu_tools.analysis.shot_uncertainty import iq_shot_sem
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.datafile import DataVariable
from zcu_tools.experiment import (
    Experiment,
    GroupedAxesSpec,
    GroupedLoadData,
    VariableAxisSpec,
    VariableSpec,
    VariableZSpec,
)
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    IQ,
    Pulse,
    Readout,
    Reset,
    SweepCfg,
    TimelineBuilder,
    shot_sums,
    split_sums,
)
from zcu_tools.utils.process import find_rotate_angle

from zcu_lab.v2.support import live_errorbar
from zcu_lab.v2.twotone.reset.rabi_check.fit import RabiCheckFit, fit_reset_rabi

_TITLE = "Reset Rabi check · qubit · IQ projection"
_XLABEL = "Rabi pulse gain (a.u.)"
_YLABEL = "Mean IQ on before-reset axis (a.u.)"
_BRANCH_LABELS = ("Rabi only (before reset)", "Rabi + reset", "Rabi + reset + Rabi")


@dataclass(frozen=True)
class RabiCheckResult:
    gains: NDArray[np.float64]
    signals: NDArray[np.complex128]
    # [N, ΣI, ΣQ, ΣI², ΣIQ, ΣQ²] per branch and gain; absent in mean-only files.
    iq_moments: NDArray[np.float64] | None = None


def _rotation(signals: NDArray[np.complex128]) -> complex:
    """Align the before-reset branch to I for all branches, or keep raw I."""
    try:
        angle = find_rotate_angle(signals[0])
    except ValueError:
        return 1.0 + 0.0j
    return complex(np.exp(-1j * angle))


def reset_rabi_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    """Return all three projected branch means on the before-reset IQ axis."""
    return (signals * _rotation(signals)).real


class RabiCheckCfg(ConfigBase):
    reset: Annotated[
        Reset.Cfg | None,
        FieldMeta(
            label="Reset",
            description="Select a preparation reset before each shot, or leave blank to skip it.",
        ),
    ] = None
    rabi_pulse: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Rabi pulse",
            description="Fill in the fixed-duration qubit pulse used for both gain-swept rotations.",
        ),
    ]
    tested_reset: Annotated[
        Reset.Cfg,
        FieldMeta(
            label="Tested reset",
            description="Fill in the reset applied after the first Rabi rotation in the two reset branches.",
        ),
    ]
    readout: Annotated[
        Readout.Cfg,
        FieldMeta(
            label="Readout",
            description="Fill in the calibrated resonator readout for the end of each shot.",
        ),
    ]
    gain: Annotated[
        SweepCfg,
        FieldMeta(
            label="Rabi pulse gain",
            unit="a.u.",
            description="Enter the common Rabi gain range for the first and second rotations (a.u.).",
        ),
    ]

    reps: Annotated[
        int,
        FieldMeta(
            label="Reps",
            unit="count",
            description="Enter the shots per sweep point and sequence branch in each acquisition round (count).",
        ),
    ]
    rounds: Annotated[
        int,
        FieldMeta(
            label="Rounds",
            unit="count",
            description="Enter the number of acquisition rounds to average (count).",
        ),
    ] = 1
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relax delay",
            unit="µs",
            decimals=3,
            description="Enter the delay after each shot to let the qubit relax (µs).",
        ),
    ]


_MEAN_AXES = (
    VariableAxisSpec(
        field_name="gains", label="Amplitude", unit="a.u.", dtype=np.float64
    ),
    VariableAxisSpec(values=(0, 1, 2), label="Reset", unit="None", dtype=np.int64),
)
_MOMENT_COLUMNS = (0, 1, 2, 3, 4, 5)


def _restore(data: GroupedLoadData[RabiCheckCfg]) -> RabiCheckResult:
    """Load old branch means, plus shot moments when saved on the same grid."""
    mean = data.variable("data")
    moments = data.variables.get(DataVariable("iq_moments"))
    if moments is None:
        return RabiCheckResult(mean.axes[0], mean.z)
    if not np.array_equal(moments.axes[0], _MOMENT_COLUMNS) or not all(
        np.array_equal(old, new)
        for old, new in zip(mean.axes, moments.axes[1:], strict=True)
    ):
        raise ValueError("iq_moments axes do not match the IQ gain and branch grid")
    return RabiCheckResult(mean.axes[0], mean.z, moments.z)


class RabiCheckExp(Experiment[RabiCheckResult, RabiCheckCfg]):
    # ADR-0069: retain the old data variable and Labber export.
    AXES_SPEC: ClassVar[GroupedAxesSpec[RabiCheckResult, RabiCheckCfg]] = (
        GroupedAxesSpec(
            variables=(
                VariableSpec(
                    "data",
                    _MEAN_AXES,
                    VariableZSpec(
                        "signals", label="Signal", unit="a.u.", dtype=np.complex128
                    ),
                ),
                VariableSpec(
                    "iq_moments",
                    (
                        VariableAxisSpec(
                            label="Moment", unit="", values=_MOMENT_COLUMNS
                        ),
                        *_MEAN_AXES,
                    ),
                    VariableZSpec("iq_moments", label="IQ moments", unit=""),
                    required=False,
                ),
            ),
            result_type=RabiCheckResult,
            cfg_type=RabiCheckCfg,
            cfg_schema_version="3.0",
            tag="twotone/reset/rabi_check",
            result_builder=_restore,
            labber_variables=("data",),
        )
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
        ax.set(xlabel=_XLABEL, ylabel=_YLABEL, title=_TITLE)
        viewer = live_errorbar(ax, num_lines=3)
        for line, label in zip(ax.lines, _BRANCH_LABELS, strict=True):
            line.set_label(label)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncols=2)
        fig.set_layout_engine("constrained")

        @plots.on_plot_thread
        def update(moments: NDArray[np.float64]) -> tuple[Artist, ...]:
            """Update three branches with the same projection for their means and SEM."""
            signals = split_sums(moments)[1]
            rotation = _rotation(signals)
            contrast = (signals * rotation).real
            sem = iq_shot_sem(moments, direction=(rotation.real, -rotation.imag))
            shots = int(np.nan_to_num(moments[..., 0]).max())
            title = ax.set_title(
                f"{_TITLE}\nRetained shots per point and branch: {shots} count"
            )
            return (*viewer.set(gains, contrast, sem), title)

        fig.show()
        buffer = SignalBuffer(
            (3, len(gains), len(_MOMENT_COLUMNS)), dtype=np.float64, on_update=update
        )

        # 3. Compare Rabi response before reset, after reset, and after reset plus a second rotation.
        with Schedule(buffer, scope=context.scope) as sched:
            builder = TimelineBuilder(
                soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay
            )
            builder.add_loop("reset_sel", 3)
            gain = builder.add_sweep("gain", cfg.gain)
            if cfg.reset is not None:
                builder.emit(Reset("reset", cfg.reset))
            builder.emit(Pulse("rabi_pulse", cfg.rabi_pulse, gain=gain))
            # Preserve each selected case's duration instead of padding to the maximum.
            builder.delay_by_timeline()
            with builder.branch(by="reset_sel", align_timeline=False) as cases:
                with cases.case(1):
                    builder.emit(Reset("tested_reset", cfg.tested_reset))
                with cases.case(2):
                    builder.emit(Reset("tested_reset", cfg.tested_reset))
                    builder.emit(Pulse("rabi_pulse", cfg.rabi_pulse, gain=gain))
            builder.delay_by_timeline()
            builder.emit(Readout("readout", cfg.readout))
            program = builder.build()
            reducer = shot_sums(program, (IQ("readout", offset_corrected=True),))
            sched.run_acquire_summed(program, soc, reducer=reducer, rounds=cfg.rounds)

        moments = buffer.array
        return RabiCheckResult(gains, split_sums(moments)[1], moments)

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
        # 1. Estimate reset contrast from three gain-response branches with a shared frequency.
        del options
        result = source.result
        gains, signals = result.gains, result.signals
        if gains.ndim != 1 or signals.shape != (3, gains.size):
            raise ValueError("Reset-check signals must have shape (3, number of gains)")
        if np.any(np.isinf(signals)):
            raise ValueError("Reset-check signals must not contain infinities")
        real_signals = reset_rabi_signal2real(signals)
        fit = fit_reset_rabi(gains, real_signals)
        sem = np.full(real_signals.shape, np.nan)
        if result.iq_moments is not None:
            rotation = _rotation(signals)
            sem = iq_shot_sem(
                result.iq_moments, direction=(rotation.real, -rotation.imag)
            )
        self._plot_fit(plots, gains, real_signals, sem, fit)
        return fit

    def _plot_fit(
        self,
        plots: Plots,
        gains: NDArray[np.float64],
        real_signals: NDArray[np.float64],
        sem: NDArray[np.float64],
        fit: RabiCheckFit,
    ) -> None:
        """Draw one fit per reset sequence and compare the residuals."""
        fig, _ = plots.subplots(
            num="fit",
            nrows=2,
            ncols=2,
            sharex=True,
            figsize=(10, 7),
            layout="constrained",
        )
        before_ax, reset_ax, after_ax, residual_ax = fig.axes
        fig.suptitle(
            f"{_TITLE}\nf = {fit.frequency:.4g} cycles/a.u. · After/before contrast = {fit.contrast_ratio:.3g} a.u."
        )
        dense_gains = np.linspace(np.min(gains), np.max(gains), 600)
        branches = (fit.before, fit.reset, fit.after)
        for index, (ax, branch, label) in enumerate(
            zip((before_ax, reset_ax, after_ax), branches, _BRANCH_LABELS, strict=True)
        ):
            color = f"C{index}"
            ax.errorbar(
                gains,
                real_signals[index],
                yerr=sem[index],
                fmt=".",
                color=color,
                markersize=5,
                label="Measured",
            )
            ax.plot(
                dense_gains,
                branch.evaluate(dense_gains, fit.frequency),
                color=color,
                linestyle="-",
                label="Fit: f + 2f" if index == 2 else "Fit: f",
            )
            ax.set_title(f"{label}\nAmplitude at f = {branch.amplitude:.3g} a.u.")
            ax.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.25 if index == 2 else -0.02),
                ncols=2,
            )
            residual_ax.errorbar(
                gains,
                real_signals[index] - branch.evaluate(gains, fit.frequency),
                yerr=sem[index],
                fmt=".-",
                color=color,
                label=f"{label}: {branch.residual_rms:.3g} a.u.",
            )
        before_ax.set_ylabel("Projected IQ (a.u.)")
        after_ax.set(xlabel=_XLABEL, ylabel="Projected IQ (a.u.)")
        after_ax.set_title(
            f"{_BRANCH_LABELS[2]}\nf amplitude = {fit.after.amplitude:.3g} a.u.; 2f = {fit.after.second_harmonic_amplitude:.3g} a.u."
        )
        residual_ax.set(
            xlabel=_XLABEL, ylabel="Measured − fitted IQ (a.u.)", title="Fit residuals"
        )
        residual_ax.axhline(0, color="gray", linewidth=0.8)
        residual_ax.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.25), title="Residual RMS"
        )
