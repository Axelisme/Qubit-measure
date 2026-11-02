"""Sweep readout gain using paired IQ moments to compare separation and QND."""

from dataclasses import dataclass
from typing import Annotated, Any, ClassVar

import numpy as np
from numpy.typing import NDArray
from pydantic import Field
from pydantic.dataclasses import dataclass as pydantic_dataclass
from zcu_tools.analysis.readout_metrics import readout_metrics
from zcu_tools.cfg_model import ConfigBase, FieldMeta
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
from zcu_tools.experiment.v2.runtime import RunOptions
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    IQ,
    Pulse,
    PulseReadout,
    Reset,
    SweepCfg,
    TimelineBuilder,
    shot_sums,
)

from zcu_lab.v2.support import live_line


@dataclass(frozen=True)
class PowerResult:
    """gains: quantized readout gain (Ngain,) in normalized a.u. within [-1, 1].

    sums: float64 (Ngain, 8, 6, 6), without round normalization.
        Branch = prepare_pi*4 + r1_on*2 + middle_pi. Feature axes are
        IQ(r1), IQ(r2), each ordered (1, I, Q, I², IQ, Q²).
        sums[..., 0, 0] is the shot count; unmeasured sums are NaN.
    """

    gains: NDArray[np.float64]
    sums: NDArray[np.float64]


class PowerCfg(ConfigBase):
    """Paired-readout modules, gain sweep and wait in us; counts add over rounds."""

    reset: Annotated[Reset.Cfg | None, FieldMeta(label="Reset")] = None
    qub_pulse: Annotated[Pulse.Cfg, FieldMeta(label="Qubit pi pulse")]
    readout: Annotated[PulseReadout.Cfg, FieldMeta(label="Readout")]
    gain: Annotated[SweepCfg, FieldMeta(label="Gain", unit="a.u.")]
    wait: Annotated[float, FieldMeta(label="Wait", unit="us", decimals=3)] = Field(
        default=0.0, ge=0.0
    )
    reps: Annotated[int, FieldMeta(label="Reps")] = 1000
    rounds: Annotated[int, FieldMeta(label="Rounds")] = 10
    relax_delay: Annotated[
        float, FieldMeta(label="Relax delay", unit="us", decimals=3)
    ] = 1.0


@pydantic_dataclass(frozen=True)
class PowerAnalyzeOptions:
    """gain: optional human-selected gain in a.u.; None proposes no writeback.

    Construction rejects nonfinite gains and values outside [-1, 1].
    """

    gain: Annotated[float, Field(ge=-1, le=1, allow_inf_nan=False)] | None = None


@dataclass(frozen=True)
class PowerQndAnalysis:
    """Human-selected gain in a.u., or None for no writeback.

    Gain-wise metrics and the Pareto frontier stay in the analysis figures.
    No automatic gain selection is performed.
    """

    gain: float | None


def restore_result(data: GroupedLoadData[PowerCfg]) -> PowerResult:
    """Restore paired IQ sums and the gain axis from the QND native record."""
    variable = data.variable("sums")
    return PowerResult(
        gains=variable.axes[3].astype(np.float64),
        sums=variable.z.astype(np.float64),
    )


class PowerExp(Experiment[PowerResult, PowerCfg]):
    """Measure eight preparation/R1/middle-pi branches at each readout gain."""

    Options: ClassVar[type[PowerAnalyzeOptions]] = PowerAnalyzeOptions
    AXES_SPEC: ClassVar[GroupedAxesSpec[PowerResult, PowerCfg]] = GroupedAxesSpec(
        variables=(
            VariableSpec(
                variable="sums",
                axes=(
                    VariableAxisSpec(label="R2 features", unit="", values=tuple(range(6))),
                    VariableAxisSpec(label="R1 features", unit="", values=tuple(range(6))),
                    VariableAxisSpec(label="Branch", unit="", values=tuple(range(8))),
                    VariableAxisSpec(label="Gain", unit="a.u.", field_name="gains"),
                ),
                z=VariableZSpec(
                    field_name="sums", label="Shot sums", unit="", dtype=np.float64
                ),
            ),
        ),
        result_type=PowerResult,
        cfg_type=PowerCfg,
        cfg_schema_version="1.0",
        tag="twotone/ro_optimize/power_qnd",
        result_builder=restore_result,
    )

    def run(
        self,
        cfg: PowerCfg,
        *,
        context: RunContext,
        acquire_kwargs: dict[str, Any] | None = None,
    ) -> PowerResult:
        """Acquire paired-readout sums (Ngain, 8, 6, 6) versus gain.

        Each branch resets optionally, prepares I/pi, emits R1 on/zero-gain,
        waits cfg.wait us, applies middle I/pi, then emits R2. R1 off retains
        the generator slot and ADC trigger. R2 always uses the swept gain.
        Wait begins at the R1 waveform end plus its configured post-delay,
        not the ADC window end. Readout timing must keep ADC windows disjoint.
        Identity plays the same qubit pulse at zero gain, preserving its native
        slot. All eight branches share reset, preparation, R1, wait, middle and
        R2 positions, including the same R1-to-R2 trigger interval.
        Live shows the R1-on, middle-I preparation mean-IQ difference magnitude.
        cfg is not mutated. acquire_kwargs is forwarded to summed acquisition.
        """
        gains = sweep2array(
            cfg.gain, "gain", {"soccfg": context.soccfg, "gen_ch": cfg.readout.gen_ch}
        )
        fig, ax = context.plots.subplots(num="measurement")
        ax.set(xlabel="Gain (a.u.)", ylabel="R1 preparation contrast (a.u.)")
        viewer = live_line(ax)

        @context.plots.on_plot_thread
        def update(sums):
            counts = sums[:, :, 0, 0]
            counts = np.where(counts == 0, np.nan, counts)
            means = sums[:, :, 1:3, 0] / counts[..., None]
            contrast = np.linalg.norm(means[:, 6] - means[:, 2], axis=-1)
            return viewer.set(gains, contrast)

        fig.show()
        buffer = SignalBuffer(
            (len(gains), 8, 6, 6), dtype=np.float64, on_update=update
        )
        with Schedule(buffer, scope=context.scope) as sched:
            builder = TimelineBuilder(
                context.soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay
            )
            gain = builder.add_sweep("gain", cfg.gain)
            builder.add_loop("branch", 8)
            r1_on = PulseReadout("r1", cfg.readout, gain=gain)
            r1_off = PulseReadout("r1", cfg.readout, gain=0.0)
            r2 = PulseReadout("r2", cfg.readout, gain=gain)
            qub_pulse = Pulse("qub_pulse", cfg.qub_pulse)
            identity = Pulse("qub_pulse", cfg.qub_pulse, gain=0.0)
            reset = Reset("reset", cfg.reset) if cfg.reset is not None else None
            with builder.branch(by="branch", align_timeline=True) as cases:
                for index in range(8):
                    with cases.case(index):
                        if reset is not None:
                            builder.emit(reset)
                        builder.emit(qub_pulse if index & 4 else identity)
                        builder.emit(r1_on if index & 2 else r1_off)
                        builder.timeline_inc(cfg.wait)
                        builder.emit(qub_pulse if index & 1 else identity)
                        builder.emit(r2)
            prog = builder.build()
            prog.set_reads_per_shot(2)
            processor = shot_sums(prog, [IQ("r1"), IQ("r2")])
            sched.run_acquire_summed(
                prog,
                context.soc,
                rounds=cfg.rounds,
                processor=processor,
                options=RunOptions(converter=np.copy),
                **(acquire_kwargs or {}),
            )
        return PowerResult(gains, buffer.array)

    def analyze(
        self,
        source: RunRecord[PowerCfg, PowerResult],
        options: PowerAnalyzeOptions,
        *,
        plots: Plots,
    ) -> PowerQndAnalysis:
        """Compute gain-wise QND metrics and plot curves plus the Pareto frontier.

        source.cfg may be None. Each gain uses readout_metrics without smoothing
        or clipping. Only finite f_sep/qnd_on pairs enter the maximizing frontier.
        options.gain records a human choice, never an inferred optimum.
        PowerAnalyzeOptions validates finite gains within [-1, 1] at construction.
        """
        result = source.result
        metrics = [readout_metrics(sums) for sums in result.sums]
        arrays = {
            name: np.array(
                [getattr(point, name) for point in metrics], dtype=np.float64
            )
            for name in (
                "snr",
                "f_sep",
                "thermal_population",
                "retention_on",
                "retention_off",
                "qnd_on",
                "qnd_off",
                "flip_up_on",
                "flip_up_off",
                "flip_down_on",
                "flip_down_off",
                "pi_miss_on",
                "pi_miss_off",
                "retention_from_covariance",
            )
        }
        f_sep, qnd = arrays["f_sep"], arrays["qnd_on"]
        valid = np.isfinite(f_sep) & np.isfinite(qnd)
        frontier = []
        for index in np.flatnonzero(valid):
            dominates = valid & (f_sep >= f_sep[index]) & (qnd >= qnd[index])
            dominates &= (f_sep > f_sep[index]) | (qnd > qnd[index])
            if not np.any(dominates):
                frontier.append(index)
        indices = np.array(frontier, dtype=np.int64)
        indices = indices[np.argsort(f_sep[indices], kind="stable")]

        fig, _ = plots.subplots(3, 3, num="metrics", figsize=(14, 11))
        for ax, name, label, symbol in zip(
            fig.axes,
            (
                "snr",
                "f_sep",
                "thermal_population",
                "retention",
                "qnd",
                "flip_up",
                "flip_down",
                "pi_miss",
                "retention_from_covariance",
            ),
            (
                "SNR",
                "Gaussian separation fidelity",
                "Thermal population p",
                "Retention S",
                "QND Q",
                "Upward flip probability",
                "Downward flip probability",
                "Middle pi miss fraction f",
                "Covariance retention (R1 on)",
            ),
            ("SNR", "F_sep", "p", "S", "Q", "epsilon up", "epsilon down", "f", "S_cov"),
            strict=True,
        ):
            if name + "_on" in arrays:
                ax.plot(result.gains, arrays[name + "_on"], "o-", label="R1 on")
                ax.plot(result.gains, arrays[name + "_off"], "s--", label="R1 off")
            else:
                ax.plot(result.gains, arrays[name], "o-", label="R1 on")
            ax.set_title(label, fontsize=11)
            ax.set(xlabel="Gain (a.u.)", ylabel=symbol + " (dimensionless)")
            ax.legend()
            ax.grid(True)
        fig.tight_layout()

        _, ax = plots.subplots(num="pareto", figsize=(7, 5))
        points = ax.scatter(
            f_sep[valid], qnd[valid], c=result.gains[valid], label="Gains"
        )
        ax.figure.colorbar(points, ax=ax, label="Gain (a.u.)")
        ax.plot(f_sep[indices], qnd[indices], "r.-", label="Pareto frontier")
        for index in np.flatnonzero(valid):
            ax.annotate(f"{result.gains[index]:.4g}", (f_sep[index], qnd[index]))
        ax.set(xlabel="Gaussian separation fidelity F_sep", ylabel="QND Q (R1 on)")
        ax.legend()
        ax.grid(True)
        return PowerQndAnalysis(gain=options.gain)
