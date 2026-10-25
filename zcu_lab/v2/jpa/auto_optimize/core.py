from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar, Literal

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
from numpy.typing import NDArray
from pydantic import Field
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
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.snr import snr_as_signal
from zcu_tools.experiment.v2.utils.tracker.moment import MomentTracker
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import Branch, Pulse, Readout, Reset, SweepCfg

from zcu_lab.v2.jpa.auto_optimize.optimizer import JPAOptimizer
from zcu_lab.v2.support import live_scatter
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


class JPAOptCfg(ConfigBase):
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
    flux_unit: Annotated[
        Literal["A", "V", "native"],
        FieldMeta(
            label="Flux unit",
            description="Working unit for the selected flux driver; no conversion.",
        ),
    ]
    reset: Annotated[Reset.Cfg | None, FieldMeta(label="Reset")] = None
    pi_pulse: Annotated[Pulse.Cfg, FieldMeta(label="Pi pulse")]
    readout: Annotated[Readout.Cfg, FieldMeta(label="Readout")]
    jpa_flux_dev: Annotated[
        str,
        FieldMeta(
            label="JPA flux device",
            description="Connected flux device driven at each optimizer point.",
        ),
    ]
    jpa_rf_dev: Annotated[
        str,
        FieldMeta(
            label="JPA RF device",
            description="Connected pump RF device driven at each optimizer point.",
        ),
    ] = ""
    jpa_flux: Annotated[
        SweepCfg,
        FieldMeta(
            label="JPA flux device value",
            description=(
                "Flux bounds in flux_unit coordinates. Start/stop are search bounds. "
                "Point counts are relative resolution hints for phase-1 allocation, "
                "not Cartesian sample counts."
            ),
        ),
    ]
    jpa_freq: Annotated[SweepCfg, FieldMeta(label="JPA pump freq", unit="MHz")]
    jpa_power: Annotated[SweepCfg, FieldMeta(label="JPA pump power", unit="dBm")]
    num_points: Annotated[
        int,
        FieldMeta(
            label="Global optimizer points",
            description="Total evaluation budget, at least four points.",
        ),
    ] = Field(ge=4, strict=True)
    skew_penalty: Annotated[
        float,
        FieldMeta(
            label="Skew penalty",
            decimals=3,
            description="Nonnegative SNR skew penalty.",
        ),
    ] = Field(default=0.0, ge=0.0)


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


def _validate_jpa_auto_result(result: JPAOptimizeResult) -> None:
    _validate_jpa_auto_arrays(
        np.asarray(result.params, dtype=np.float64),
        np.asarray(result.phases),
        np.asarray(result.signals, dtype=np.float64),
    )


def _build_jpa_auto_result(
    data: GroupedLoadData[JPAOptCfg],
) -> JPAOptimizeResult:
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


_JPA_AUTO_ITERATION_AXIS = (
    VariableAxisSpec.generated_arange("Iteration", "a.u.", dtype=np.int64),
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

        optimizer = JPAOptimizer(flux_sweep, freq_sweep, gain_sweep, num_points)
        params = np.full((num_points, 3), np.nan, dtype=np.float64)
        phases = np.zeros(num_points, dtype=np.int32)
        point_indices = np.arange(num_points, dtype=np.float64)
        # Show SNR versus iteration and each candidate parameter, colored by phase.
        figure, initial_ax = context.plots.subplots("measurement", figsize=(8, 5))
        initial_ax.remove()
        figure.suptitle("JPA Auto Optimization")
        grid = figure.add_gridspec(3, 2, width_ratios=[1.5, 1])
        viewers = [
            live_scatter(
                context.plots,
                "measurement",
                label,
                "SNR (a.u.)",
                axes=figure.add_subplot(cell),
            )
            for label, cell in (
                ("Iteration", grid[:, 0]),
                ("JPA Flux value (a.u.)", grid[0, 1]),
                ("JPA Frequency (MHz)", grid[1, 1]),
                ("JPA Power (dBm)", grid[2, 1]),
            )
        ]
        current_index = 0

        def plot_fn(data: NDArray[np.float64]) -> None:
            idx = current_index
            snrs = np.abs(data)
            cur_flux, cur_freq, cur_gain = params[idx, :]
            title = (
                f"Iteration {idx}, Phase {phases[idx]}, Flux: {cur_flux:.2g} (a.u.), "
                f"Freq: {1e-3 * cur_freq:.4g} (GHz), Power: {cur_gain:.2g} (dBm)"
            )
            colors = phases.astype(np.float64)
            for viewer, xs, viewer_title in zip(
                viewers,
                (point_indices, params[:, 0], params[:, 1], params[:, 2]),
                (title, None, None, None),
                strict=True,
            ):
                viewer.update(
                    xs, snrs, colors=colors, title=viewer_title, refresh=False
                )
            context.plots.refresh("measurement")

        signals_buffer = SignalBuffer(
            (num_points,), dtype=np.float64, on_update=plot_fn
        )

        # 2. Use the previous SNR to choose and apply the next pump/bias setting.
        with Schedule(cfg, signals_buffer, scope=context.scope) as sched:
            for idx, step in sched.scan("Iteration", range(num_points)):
                current_index = idx
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
                    context.devices[step.cfg.jpa_flux_dev],
                    params[idx, 0],
                    unit=step.cfg.flux_unit,
                    stop=context.scope.stop,
                    progress=False,
                )
                set_rf_frequency(
                    context.devices[step.cfg.jpa_rf_dev], 1e6 * params[idx, 1]
                )
                set_rf_power(context.devices[step.cfg.jpa_rf_dev], params[idx, 2])

                # 3. Compare reset-only and pi-prepared readout for this candidate.
                tracker = MomentTracker()
                _ = (
                    step.prog_builder(
                        soc,
                        soccfg,
                        reps=step.cfg.reps,
                        relax_delay=step.cfg.relax_delay,
                    )
                    .add(
                        Reset("reset", step.cfg.reset),
                        Branch("ge", [], Pulse("pi_pulse", step.cfg.pi_pulse)),
                        Readout("readout", step.cfg.readout),
                    )
                    .declare_sweep("ge", 2)
                    .build_and_acquire(
                        raw2signal_fn=lambda raw, tracker=tracker: snr_as_signal(
                            [tracker],
                            ge_axis=1,
                            skew_penalty=sched.cfg.skew_penalty,
                        ),
                        trackers=[tracker],
                        rounds=cfg.rounds,
                    )
                )

        # 4. Keep candidates, search phases, and measured SNR paired by iteration.
        return JPAOptimizeResult(
            params=params, phases=phases, signals=signals_buffer.array
        )

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
        result = source.result

        params = result.params
        phases = result.phases
        signals = result.signals
        snrs = np.abs(signals)

        max_id = np.nanargmax(snrs)
        max_snr = float(snrs[max_id])
        best_params = params[max_id, :]

        colors = phases

        figsize = (8, 5)
        fig, initial_ax = plots.subplots("fit", figsize=figsize)
        initial_ax.remove()
        gs = fig.add_gridspec(3, 2, width_ratios=[1.5, 1])

        fig.suptitle("JPA Auto Optimization")

        ax_iter = fig.add_subplot(gs[:, 0])
        ax_flux = fig.add_subplot(gs[0, 1])
        ax_freq = fig.add_subplot(gs[1, 1])
        ax_power = fig.add_subplot(gs[2, 1])

        ax_iter.scatter(np.arange(len(snrs)), snrs, c=colors, s=1)
        ax_iter.axhline(max_snr, color="r", ls="--", label=f"best = {max_snr:.2g}")
        ax_iter.scatter([max_id], [max_snr], color="r", marker="*")
        ax_iter.set_xlabel("Iteration")
        ax_iter.set_ylabel("SNR")
        ax_iter.legend()
        ax_iter.grid(True)

        def plot_ax(ax, param_idx, label_name) -> None:
            ax.scatter(params[:, param_idx], snrs, c=colors, s=1)
            best_value = best_params[param_idx]
            ax.axvline(best_value, color="r", ls="--", label=f"best = {best_value:.2g}")
            ax.scatter([best_value], [max_snr], color="r", marker="*")
            ax.set_xlabel(label_name)
            ax.set_ylabel("SNR")
            ax.legend()
            ax.grid(True)

        plot_ax(ax_flux, 0, "JPA Flux value (a.u.)")
        plot_ax(ax_freq, 1, "JPA Frequency (MHz)")
        plot_ax(ax_power, 2, "JPA Power (dBm)")

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

        _, ax = plots.subplots("sample_params", subplot_kw={"projection": "3d"})
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
