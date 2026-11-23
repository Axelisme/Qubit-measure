from __future__ import annotations

from collections.abc import Callable, Iterable
from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar, Literal

import numpy as np
from matplotlib import colormaps
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D
from numpy.typing import NDArray
from pydantic import Field
from zcu_tools.cfg_model import ConfigBase, DeviceName, FieldMeta
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
from zcu_tools.experiment.v2.runtime import RunOptions, Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.snr import snr_as_signal
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    IQ,
    Pulse,
    Readout,
    Reset,
    SweepCfg,
    TimelineBuilder,
    shot_sums,
)

from zcu_lab.v2.jpa.auto_optimize.optimizer import JPAOptimizer
from zcu_lab.v2.support import live_line, live_scatter
from zcu_lab.v2.support.device import set_flux, set_rf_frequency, set_rf_power


@dataclass(frozen=True)
class JPAOptimizeResult:
    """Sampled [flux device value, MHz, dBm] rows, optimizer phases, and GE SNR.

    All arrays follow iteration order; unmeasured params/signals remain NaN.
    Phase labels record the search stage, not a physical pulse phase.
    """

    params: NDArray[np.float64]
    phases: NDArray[np.int32]
    signals: NDArray[np.float64]


@dataclass(frozen=True)
class JPAOptimizeAnalysis:
    best_flux: float
    best_freq: float
    best_power: float


def _validate_jpa_auto_arrays(
    params: NDArray[np.float64],
    phases: NDArray[Any],
    signals: NDArray[np.float64],
) -> None:
    if params.ndim != 2 or params.shape[1] != 3:
        raise ValueError(
            f"JPA auto-optimize params must have shape (N, 3), got {params.shape}"
        )
    if phases.ndim != 1:
        raise ValueError(
            f"JPA auto-optimize phases must be 1-D, got shape {phases.shape}"
        )
    if signals.ndim != 1:
        raise ValueError(
            f"JPA auto-optimize signals must be 1-D, got shape {signals.shape}"
        )
    if phases.shape[0] != params.shape[0]:
        raise ValueError(
            "JPA auto-optimize phases length must match params rows "
            f"(got phases={phases.shape}, params={params.shape})"
        )
    if signals.shape[0] != params.shape[0]:
        raise ValueError(
            "JPA auto-optimize signals length must match params rows "
            f"(got signals={signals.shape}, params={params.shape})"
        )


JPA_AUTO_FLUX_VARIABLE = "jpa_flux"
JPA_AUTO_FREQ_VARIABLE = "jpa_freq"
JPA_AUTO_POWER_VARIABLE = "jpa_power"
JPA_AUTO_PHASE_VARIABLE = "jpa_phase"
JPA_AUTO_SNR_VARIABLE = "snr"
JPA_AUTO_GROUPED_VARIABLES = (
    JPA_AUTO_FLUX_VARIABLE,
    JPA_AUTO_FREQ_VARIABLE,
    JPA_AUTO_POWER_VARIABLE,
    JPA_AUTO_PHASE_VARIABLE,
    JPA_AUTO_SNR_VARIABLE,
)


def _validate_jpa_auto_result(result: JPAOptimizeResult) -> None:
    _validate_jpa_auto_arrays(
        np.asarray(result.params, dtype=np.float64),
        np.asarray(result.phases),
        np.asarray(result.signals, dtype=np.float64),
    )


def _build_jpa_auto_result(
    data: GroupedLoadData[JPAOptCfg],
) -> JPAOptimizeResult:
    """Build the JPA result from grouped saved variables without acquiring or plotting."""
    params = np.column_stack(
        [
            data.variable(JPA_AUTO_FLUX_VARIABLE).z,
            data.variable(JPA_AUTO_FREQ_VARIABLE).z,
            data.variable(JPA_AUTO_POWER_VARIABLE).z,
        ]
    ).astype(np.float64)
    phases = data.variable(JPA_AUTO_PHASE_VARIABLE).z.astype(np.int32)
    signals = data.variable(JPA_AUTO_SNR_VARIABLE).z.astype(np.float64)
    _validate_jpa_auto_arrays(params, phases, signals)
    return JPAOptimizeResult(
        params=params,
        phases=phases,
        signals=signals,
    )


class JPAOptCfg(ConfigBase):
    reset: Annotated[
        Reset.Cfg | None,
        FieldMeta(
            label="Reset",
            description="Select a qubit reset before readout; leave empty to skip it.",
        ),
    ] = None
    pi_pulse: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Qubit π pulse",
            description="Select the calibrated g→e pulse for the excited preparation only.",
        ),
    ]
    readout: Annotated[
        Readout.Cfg,
        FieldMeta(
            label="Readout",
            description="Select the calibrated resonator pulse and IQ integration for both preparations.",
        ),
    ]
    jpa_flux_dev: Annotated[
        DeviceName,
        FieldMeta(
            label="JPA flux-bias source",
            description="Select the DC source that sets the JPA bias.",
        ),
    ]
    jpa_rf_dev: Annotated[
        DeviceName,
        FieldMeta(
            label="JPA pump RF source",
            description="Select the RF source whose JPA pump frequency and power are varied.",
        ),
    ] = ""
    flux_unit: Annotated[
        Literal["A", "V", "native"],
        FieldMeta(
            label="JPA bias unit",
            description="Match the source unit: amps, volts, or its native coordinate; the sweep values are not converted.",
        ),
    ]
    jpa_flux: Annotated[
        SweepCfg,
        FieldMeta(
            label="JPA flux-bias value",
            description="Enter bias search bounds in JPA bias unit and a relative sampling resolution.",
        ),
    ]
    jpa_freq: Annotated[
        SweepCfg,
        FieldMeta(
            label="JPA pump RF frequency",
            unit="MHz",
            description="Enter the absolute pump RF search bounds (MHz) and relative sampling resolution.",
        ),
    ]
    jpa_power: Annotated[
        SweepCfg,
        FieldMeta(
            label="JPA pump source power",
            unit="dBm",
            description="Enter the pump source power bounds (dBm) and relative sampling resolution.",
        ),
    ]
    num_points: Annotated[
        int,
        FieldMeta(
            label="Candidate budget",
            unit="count",
            description="Enter the total number of JPA settings to measure (at least four).",
        ),
    ] = Field(ge=4, strict=True)
    skew_penalty: Annotated[
        float,
        FieldMeta(
            label="IQ shape penalty",
            unit="a.u.",
            decimals=3,
            description="Enter a nonnegative weight for unequal IQ widths and skew; 0 uses separation SNR alone.",
        ),
    ] = Field(default=0.0, ge=0.0)
    seed: Annotated[
        int | None,
        FieldMeta(
            label="Optimizer seed",
            description=(
                "Enter a nonnegative integer to repeat candidate choices for the same score history; "
                "leave blank for a new search."
            ),
        ),
    ] = None
    reps: Annotated[
        int,
        FieldMeta(
            label="Reps",
            unit="count",
            description="Enter the shots per preparation at each setting in one round.",
        ),
    ]
    rounds: Annotated[
        int,
        FieldMeta(
            label="Rounds",
            unit="count",
            description="Enter the number of acquisition rounds per setting.",
        ),
    ] = 1
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relaxation wait",
            unit="µs",
            decimals=3,
            description="Enter the wait after readout for relaxation before the next shot (µs).",
        ),
    ]


_JPA_AUTO_ITERATION_AXIS = (
    VariableAxisSpec.generated_arange(label="Iteration", unit="a.u.", dtype=np.int64),
)
JPA_AUTO_GROUPED_AXES_SPEC = GroupedAxesSpec(
    variables=(
        VariableSpec(
            variable=JPA_AUTO_FLUX_VARIABLE,
            axes=_JPA_AUTO_ITERATION_AXIS,
            z=VariableZSpec(
                field_name="params",
                label="JPA Flux",
                # Canonical flux unit is the neutral device-native value: the
                # generic set_flux knob carries no physical-unit guarantee, so
                # a.u. is the only honest cross-device contract (identity
                # scale; legacy 'A' grouped files migrate, never load).
                unit="a.u.",
                dtype=np.float64,
                index=0,
                index_axis=1,
            ),
        ),
        VariableSpec(
            variable=JPA_AUTO_FREQ_VARIABLE,
            axes=_JPA_AUTO_ITERATION_AXIS,
            z=VariableZSpec(
                field_name="params",
                label="JPA Frequency",
                unit="Hz",
                scale=1e6,
                dtype=np.float64,
                index=1,
                index_axis=1,
            ),
        ),
        VariableSpec(
            variable=JPA_AUTO_POWER_VARIABLE,
            axes=_JPA_AUTO_ITERATION_AXIS,
            z=VariableZSpec(
                field_name="params",
                label="JPA Power",
                unit="dBm",
                dtype=np.float64,
                index=2,
                index_axis=1,
            ),
        ),
        VariableSpec(
            variable=JPA_AUTO_PHASE_VARIABLE,
            axes=_JPA_AUTO_ITERATION_AXIS,
            z=VariableZSpec(
                field_name="phases",
                label="JPA Phase",
                unit="index",
                dtype=np.int32,
            ),
        ),
        VariableSpec(
            variable=JPA_AUTO_SNR_VARIABLE,
            axes=_JPA_AUTO_ITERATION_AXIS,
            z=VariableZSpec(
                field_name="signals",
                label="SNR",
                unit="a.u.",
                dtype=np.float64,
            ),
        ),
    ),
    result_type=JPAOptimizeResult,
    cfg_type=JPAOptCfg,
    tag="jpa/auto_optimize",
    cfg_schema_version="3.0",
    result_builder=_build_jpa_auto_result,
    result_validator=_validate_jpa_auto_result,
)


class AutoOptimizeExp(Experiment[JPAOptimizeResult, JPAOptCfg]):
    AXES_SPEC: ClassVar[GroupedAxesSpec[JPAOptimizeResult, JPAOptCfg]] = (
        JPA_AUTO_GROUPED_AXES_SPEC
    )

    def run(self, cfg: JPAOptCfg, *, context: RunContext) -> JPAOptimizeResult:
        """Search JPA flux, pump frequency, and pump power for high readout SNR.

        For each candidate, set flux/frequency/power, reset, select no pulse
        or a pi pulse, and read out both preparation branches. Their IQ
        moments give SNR with the configured skew penalty, not assignment
        fidelity. Feed the previous SNR to the optimizer for the next point.
        Return sampled [flux device value, MHz, dBm], search phases, and SNR.

        Sweep bounds and relative resolution hints guide the search; they
        do not define a Cartesian scan. ``num_points`` is an integer budget
        of at least four evaluations. Both device names must exist in
        ``context.devices``; ``flux_unit`` is used without conversion.
        Optimizer exhaustion before the budget is consumed is an error.
        """
        # 1. Allocate the search budget and iteration-aligned result arrays.
        cfg = deepcopy(cfg)
        soc, soccfg = context.soc, context.soccfg
        num_points = cfg.num_points
        flux_sweep = cfg.jpa_flux
        freq_sweep = cfg.jpa_freq
        gain_sweep = cfg.jpa_power

        optimizer = JPAOptimizer(
            flux_sweep, freq_sweep, gain_sweep, num_points, cfg.seed
        )
        params = np.full((num_points, 3), np.nan, dtype=np.float64)
        phases = np.zeros(num_points, dtype=np.int32)
        unit = "a.u." if cfg.flux_unit == "native" else cfg.flux_unit
        update = self._show_search_plot(context, unit)

        def plot_fn(data: NDArray[np.float64]) -> None:
            """Update the live JPA artists from the latest acquired signals."""
            update(JPAOptimizeResult(params=params, phases=phases, signals=data))

        signals_buffer = SignalBuffer(
            (num_points,), dtype=np.float64, on_update=plot_fn
        )

        def score(sums: NDArray[np.float64]) -> NDArray[np.float64]:
            """Return the g/e separation score for one pump and bias setting.

            Convert the two states' readout IQ sums through order three using
            the configured skew penalty. Return a scalar float64 array, NaN
            if either state has no shots.
            """
            return snr_as_signal(sums, skew_penalty=cfg.skew_penalty)

        # 2. Use the previous SNR to choose and apply the next pump/bias setting.
        with Schedule(signals_buffer, scope=context.scope) as sched:
            for idx, step in sched.scan("Iteration", range(num_points)):
                last_snr = None
                if idx > 0:
                    last_snr = np.abs(signals_buffer.array[idx - 1])
                cur_params = optimizer.next_params(idx, last_snr)
                if cur_params is None:
                    raise RuntimeError(
                        "JPA optimizer exhausted before consuming its budget: "
                        f"iteration={idx}, num_points={num_points}, "
                        f"phase={optimizer.phase}"
                    )

                params[idx, :] = cur_params
                phases[idx] = optimizer.phase
                set_flux(
                    context.devices[cfg.jpa_flux_dev],
                    params[idx, 0],
                    unit=cfg.flux_unit,
                    stop=context.scope.stop,
                    progress=False,
                )
                set_rf_frequency(context.devices[cfg.jpa_rf_dev], 1e6 * params[idx, 1])
                set_rf_power(context.devices[cfg.jpa_rf_dev], params[idx, 2])

                # 3. Compare reset-only and pi-prepared readout for this candidate.
                builder = TimelineBuilder(
                    soccfg,
                    reps=cfg.reps,
                    relax_delay=cfg.relax_delay,
                )
                builder.add_loop("ge", 2)
                if cfg.reset is not None:
                    builder.emit(Reset("reset", cfg.reset))
                builder.delay_by_timeline()
                # Empty g case has no pi-length padding; preserve each case's readout time.
                with (
                    builder.branch(by="ge", align_timeline=False) as cases,
                    cases.case(1),
                ):
                    builder.emit(Pulse("pi_pulse", cfg.pi_pulse))
                builder.delay_by_timeline()
                builder.emit(Readout("readout", cfg.readout))
                prog = builder.build()

                step.run_acquire_summed(
                    prog,
                    soc,
                    reducer=shot_sums(prog, [IQ("readout", order=3)]),
                    options=RunOptions(converter=score),
                    rounds=cfg.rounds,
                )

        # 4. Keep candidates, search phases, and measured SNR paired by iteration.
        return JPAOptimizeResult(
            params=params, phases=phases, signals=signals_buffer.array
        )

    def _search_panels(
        self, plots: Plots, name: str, unit: str
    ) -> tuple[Figure, tuple[Axes, ...]]:
        """Lay out search progression and three views of the same measured candidates."""
        figure, _ = plots.subplots(2, 2, num=name, figsize=(9, 6), layout="constrained")
        panels = tuple(figure.axes)
        figure.suptitle(
            "JPA joint search · pump and bias settings · g/e separation score"
        )
        for ax, label in zip(
            panels,
            (
                "Candidate number (count)",
                f"JPA flux-bias value ({unit})",
                "JPA pump RF frequency (MHz)",
                "JPA pump source power (dBm)",
            ),
            strict=True,
        ):
            ax.set(xlabel=label, ylabel="g/e separation score (a.u.)")
            ax.ticklabel_format(axis="x", style="plain", useOffset=False)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=ax is panels[0]))
        return figure, panels

    def _show_search_plot(
        self, context: RunContext, unit: str
    ) -> Callable[[JPAOptimizeResult], None]:
        """Show measured candidate scores and their search stages."""
        figure, axes = self._search_panels(context.plots, "measurement", unit)
        progression = live_line(axes[0])
        axes[0].lines[0].set(marker="None", color="0.5", zorder=1)
        viewers = [live_scatter(ax) for ax in axes]
        colorbar = figure.colorbar(
            axes[0].collections[0], ax=axes, label="Search stage (count)"
        )

        @context.plots.on_plot_thread
        def update(result: JPAOptimizeResult) -> Iterable[Artist]:
            """Update the score trace and stage-colored scatter plots from the copied result."""
            measured = np.isfinite(result.signals)
            if not measured.any():
                return ()

            stages = result.phases[measured].astype(np.float64)
            colorbar.set_ticks(np.unique(stages).tolist())
            candidates = np.arange(1, len(measured) + 1, dtype=np.float64)
            scores = np.abs(result.signals)
            artists: list[Artist] = list(progression.set(candidates, scores))
            for viewer, xs in zip(
                viewers,
                (candidates[measured], *result.params[measured].T),
                strict=True,
            ):
                (scatter,) = viewer.set(xs, scores[measured], colors=stages)
                scatter.set_sizes(np.array([32.0]))
                artists.append(scatter)
            return artists

        figure.show()
        return update

    def analyze(
        self,
        source: RunRecord[JPAOptCfg, JPAOptimizeResult],
        # tolerated: Experiment requires options even when this analysis has no options.
        options: None,  # noqa: ARG002
        *,
        plots: Plots,
    ) -> JPAOptimizeAnalysis:
        """Return the measured candidate with peak SNR and plot four projections.

        Use ``source`` data and None options, with no interpolation. The
        returned flux is a device value, frequency is MHz, and power is dBm.
        All-NaN SNR fails rather than proposing an unmeasured candidate.
        """
        # 1. Choose the highest measured SNR without interpolating a new setting.
        result = source.result

        params = result.params
        phases = result.phases
        signals = result.signals
        snrs = np.abs(signals)

        max_id = np.nanargmax(snrs)
        max_snr = float(snrs[max_id])
        best_params = params[max_id, :]

        # 2. Overlay the selected tuple on the same four views as the live figure.
        unit = source.cfg.flux_unit if source.cfg is not None else "native units"
        unit = "a.u." if unit == "native" else unit
        fig, axes = self._search_panels(plots, "fit", unit)
        measured = np.isfinite(snrs)
        candidates = np.arange(1, len(snrs) + 1)
        axes[0].plot(candidates, snrs, "-", color="0.5", zorder=1)
        stages = phases[measured]
        norm = Normalize(vmin=float(stages.min()), vmax=float(stages.max()))
        best_labels = (
            f"Candidate {max_id + 1}: score = {max_snr:.3g} a.u.",
            f"Selected bias = {best_params[0]:.6g} {unit}",
            f"Selected pump RF = {best_params[1]:.3f} MHz",
            f"Selected pump power = {best_params[2]:.2f} dBm",
        )
        for ax, xs, best_value, label in zip(
            axes,
            (candidates, *params.T),
            (max_id + 1, *best_params),
            best_labels,
            strict=True,
        ):
            ax.scatter(
                xs[measured],
                snrs[measured],
                c=stages,
                cmap="viridis",
                norm=norm,
                marker=".",
                s=32,
            )
            ax.plot(best_value, max_snr, "*", color="C3", markersize=9)
            ax.set_title(label, fontsize=10, loc="left")
        fig.colorbar(
            axes[0].collections[0],
            ax=axes,
            label="Search stage (count)",
            ticks=np.unique(stages),
        )

        return JPAOptimizeAnalysis(
            float(best_params[0]), float(best_params[1]), float(best_params[2])
        )

    def plot_sample_params(
        self, source: RunRecord[JPAOptCfg, JPAOptimizeResult], *, plots: Plots
    ) -> None:
        """Plot ``source`` candidates in flux/MHz/dBm space for search diagnosis.

        Color marks the optimizer phase; opacity scales with measured SNR.
        This view does not select or apply a pump setting.
        """
        result = source.result

        params = result.params
        phases = result.phases
        signals = result.signals
        snrs = np.abs(signals)

        max_snr = np.nanmax(snrs)
        alphas = snrs / max(max_snr, 1e-12)

        _, ax = plots.subplots(num="sample_params", subplot_kw={"projection": "3d"})
        assert isinstance(ax, Axes3D)

        cmap = colormaps["viridis"]
        norm = Normalize(vmin=float(np.nanmin(phases)), vmax=float(np.nanmax(phases)))
        colors = cmap(norm(phases))
        colors[:, 3] = alphas

        # Convert NumPy coordinates/size at the Matplotlib boundary; its scalar defaults are typed as int.
        ax.scatter(
            params[:, 0],
            params[:, 1],
            params[:, 2].tolist(),
            c=colors,
            s=np.asarray(0.1).item(),
        )

        ax.set_xlabel("Flux value")
        ax.set_ylabel("Freq (MHz)")
        ax.set_zlabel("Power (dBm)")
