from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from numpy.typing import NDArray
from pydantic import Field

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.device import DeviceInfo
from zcu_tools.experiment import (
    MHZ_TO_HZ,
    AbsExperiment,
    GroupedAxesSpec,
    GroupedLoadData,
    RoleAxisSpec,
    RoleSpec,
    RoleZSpec,
)
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import (
    make_comment,
    set_flux_in_dev_cfg,
    set_freq_in_dev_cfg,
    set_power_in_dev_cfg,
    setup_devices,
)
from zcu_tools.experiment.v2.runner import Task, TaskState, run_task
from zcu_tools.experiment.v2.utils import snr_as_signal
from zcu_tools.experiment.v2.utils.tracker import MomentTracker
from zcu_tools.liveplot import LivePlotScatter, MultiLivePlot
from zcu_tools.liveplot.backend.jupyter import instant_plot
from zcu_tools.program.v2 import (
    Branch,
    ModularProgramV2,
    ProgramV2Cfg,
    Pulse,
    PulseCfg,
    Readout,
    ReadoutCfg,
    Reset,
    ResetCfg,
    SweepCfg,
)
from zcu_tools.utils.datasaver import LabberPayload

from .jpa_optimizer import JPAOptimizer


@dataclass(frozen=True)
class JPAOptimizeResult:
    params: NDArray[np.float64]
    phases: NDArray[np.int32]
    signals: NDArray[np.float64]
    cfg_snapshot: JPAOptCfg | None = None


class JPAOptModuleCfg(ConfigBase):
    reset: ResetCfg | None = None
    pi_pulse: PulseCfg
    readout: ReadoutCfg


class JPAOptSweepCfg(ConfigBase):
    jpa_flux: SweepCfg
    jpa_freq: SweepCfg
    jpa_power: SweepCfg


class JPAOptCfg(ProgramV2Cfg, ExpCfgModel):
    modules: JPAOptModuleCfg
    # Field(...) makes dev required in this subclass, overriding the Optional
    # default from ExpCfgModel — intentional Pydantic pattern (type: ignore[override]).
    dev: Mapping[str, DeviceInfo] = Field(...)  # type: ignore[override]
    sweep: JPAOptSweepCfg
    skew_penalty: float = Field(default=0.0, ge=0.0)


JPA_AUTO_FLUX_ROLE = "jpa_flux"
JPA_AUTO_FREQ_ROLE = "jpa_freq"
JPA_AUTO_POWER_ROLE = "jpa_power"
JPA_AUTO_PHASE_ROLE = "jpa_phase"
JPA_AUTO_SNR_ROLE = "snr"
JPA_AUTO_GROUPED_ROLES = (
    JPA_AUTO_FLUX_ROLE,
    JPA_AUTO_FREQ_ROLE,
    JPA_AUTO_POWER_ROLE,
    JPA_AUTO_PHASE_ROLE,
    JPA_AUTO_SNR_ROLE,
)


def jpa_auto_result_to_grouped_payloads(
    result: JPAOptimizeResult,
) -> dict[str, LabberPayload]:
    return JPA_AUTO_GROUPED_AXES_SPEC.payloads_from_result(result)


def save_jpa_auto_grouped_result(
    filepath: str,
    result: JPAOptimizeResult,
    *,
    comment: str = "",
    tag: str = "jpa/auto_optimize",
) -> str:
    return JPA_AUTO_GROUPED_AXES_SPEC.save_grouped_result(
        filepath,
        result,
        comment=comment,
        tag=tag,
    )


def load_jpa_auto_grouped_result(filepath: str) -> JPAOptimizeResult:
    return JPA_AUTO_GROUPED_AXES_SPEC.load_result(filepath)


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
            data.role(JPA_AUTO_FLUX_ROLE).z,
            data.role(JPA_AUTO_FREQ_ROLE).z,
            data.role(JPA_AUTO_POWER_ROLE).z,
        ]
    ).astype(np.float64)
    phases = data.role(JPA_AUTO_PHASE_ROLE).z.astype(np.int32)
    signals = data.role(JPA_AUTO_SNR_ROLE).z.astype(np.float64)
    _validate_jpa_auto_arrays(params, phases, signals)
    return JPAOptimizeResult(
        params=params,
        phases=phases,
        signals=signals,
        cfg_snapshot=data.cfg_snapshot,
    )


_JPA_AUTO_ITERATION_AXIS = (
    RoleAxisSpec.generated_arange("Iteration", "a.u.", dtype=np.int64),
)
JPA_AUTO_GROUPED_AXES_SPEC = GroupedAxesSpec(
    roles=(
        RoleSpec(
            role=JPA_AUTO_FLUX_ROLE,
            axes=_JPA_AUTO_ITERATION_AXIS,
            z=RoleZSpec(
                field_name="params",
                label="JPA Flux",
                unit="A",
                dtype=np.float64,
                index=0,
                index_axis=1,
            ),
        ),
        RoleSpec(
            role=JPA_AUTO_FREQ_ROLE,
            axes=_JPA_AUTO_ITERATION_AXIS,
            z=RoleZSpec(
                field_name="params",
                label="JPA Frequency",
                unit="Hz",
                scale=MHZ_TO_HZ,
                dtype=np.float64,
                index=1,
                index_axis=1,
            ),
        ),
        RoleSpec(
            role=JPA_AUTO_POWER_ROLE,
            axes=_JPA_AUTO_ITERATION_AXIS,
            z=RoleZSpec(
                field_name="params",
                label="JPA Power",
                unit="dBm",
                dtype=np.float64,
                index=2,
                index_axis=1,
            ),
        ),
        RoleSpec(
            role=JPA_AUTO_PHASE_ROLE,
            axes=_JPA_AUTO_ITERATION_AXIS,
            z=RoleZSpec(
                field_name="phases",
                label="JPA Phase",
                unit="index",
                dtype=np.int32,
            ),
        ),
        RoleSpec(
            role=JPA_AUTO_SNR_ROLE,
            axes=_JPA_AUTO_ITERATION_AXIS,
            z=RoleZSpec(
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
    result_builder=_build_jpa_auto_result,
    result_validator=_validate_jpa_auto_result,
)


class AutoOptimizeExp(AbsExperiment[JPAOptimizeResult, JPAOptCfg]):
    def run(self, soc, soccfg, cfg: JPAOptCfg, num_points: int) -> JPAOptimizeResult:
        cfg = deepcopy(cfg)
        flux_sweep = cfg.sweep.jpa_flux
        freq_sweep = cfg.sweep.jpa_freq
        gain_sweep = cfg.sweep.jpa_power

        optimizer = JPAOptimizer(flux_sweep, freq_sweep, gain_sweep, num_points)

        # (num_points, [flux, freq, power])
        params = np.full((num_points, 3), np.nan, dtype=np.float64)
        phases = np.zeros(num_points, dtype=np.int32)

        def measure_fn(
            ctx: TaskState[NDArray[np.float64], Any, JPAOptCfg], update_hook: Callable
        ) -> list[MomentTracker]:
            cfg = ctx.cfg
            setup_devices(cfg, progress=False)
            modules = cfg.modules

            prog = ModularProgramV2(
                soccfg,
                cfg,
                modules=[
                    Reset("reset", modules.reset),
                    Branch("ge", [], Pulse("pi_pulse", modules.pi_pulse)),
                    Readout("readout", modules.readout),
                ],
                sweep=[("ge", 2)],
            )

            tracker = MomentTracker()
            prog.acquire(
                soc,
                progress=False,
                round_hook=lambda i, avg_d: update_hook(i, [tracker]),
                trackers=[tracker],
                stop_checkers=[ctx.is_stop],
            )
            return [tracker]

        def update_fn(i, ctx: TaskState[Any, Any, JPAOptCfg], _) -> None:
            ctx.env["index"] = i

            last_snr = None
            if i > 0:
                last_snr = np.abs(ctx.root_data[i - 1])
            cur_params = optimizer.next_params(i, last_snr)

            if cur_params is None:
                # Exhausting the optimizer stops the task through the existing interrupt path.
                raise KeyboardInterrupt("No more parameters to optimize.")

            params[i, :] = cur_params
            phases[i] = optimizer.phase

            dev = ctx.cfg.dev
            assert dev is not None, "JPA auto optimize requires cfg.dev"
            set_flux_in_dev_cfg(dev, params[i, 0], label="jpa_flux_dev")
            set_freq_in_dev_cfg(dev, 1e6 * params[i, 1], label="jpa_rf_dev")
            set_power_in_dev_cfg(dev, params[i, 2], label="jpa_rf_dev")

        # initialize figure and axes
        figsize = (8, 5)
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 2, width_ratios=[1.5, 1])

        fig.suptitle("JPA Auto Optimization")

        ax_iter = fig.add_subplot(gs[:, 0])
        ax_flux = fig.add_subplot(gs[0, 1])
        ax_freq = fig.add_subplot(gs[1, 1])
        ax_power = fig.add_subplot(gs[2, 1])

        instant_plot(fig)  # show the figure immediately

        with MultiLivePlot(
            fig,
            plotters=dict(
                iter_scatter=LivePlotScatter(
                    "Iteration", "SNR (a.u.)", existed_axes=[[ax_iter]]
                ),
                flux_scatter=LivePlotScatter(
                    "JPA Flux (mA)", "SNR (a.u.)", existed_axes=[[ax_flux]]
                ),
                freq_scatter=LivePlotScatter(
                    "JPA Frequency (MHz)", "SNR (a.u.)", existed_axes=[[ax_freq]]
                ),
                power_scatter=LivePlotScatter(
                    "JPA Power (dBm)", "SNR (a.u.)", existed_axes=[[ax_power]]
                ),
            ),
        ) as viewer:

            def plot_fn(ctx: TaskState) -> None:
                idx: int = ctx.env["index"]
                snrs = np.abs(ctx.root_data)  # (num_points, )

                cur_flux, cur_freq, cur_gain = params[idx, :]

                fig.suptitle(
                    f"Iteration {idx}, Phase {phases[idx]}, Flux: {1e3 * cur_flux:.2g} (mA), Freq: {1e-3 * cur_freq:.4g} (GHz), Power: {cur_gain:.2g} (dBm)"
                )

                colors = phases

                viewer.get_plotter("iter_scatter").update(
                    np.arange(num_points), snrs, colors=colors, refresh=False
                )
                viewer.get_plotter("flux_scatter").update(
                    1e3 * params[:, 0], snrs, colors=colors, refresh=False
                )
                viewer.get_plotter("freq_scatter").update(
                    params[:, 1], snrs, colors=colors, refresh=False
                )
                viewer.get_plotter("power_scatter").update(
                    params[:, 2], snrs, colors=colors, refresh=False
                )
                viewer.refresh()

            results = run_task(
                task=Task(
                    measure_fn=measure_fn,
                    raw2signal_fn=lambda raw: snr_as_signal(
                        raw, ge_axis=0, skew_penalty=cfg.skew_penalty
                    ),
                    dtype=np.float64,
                    pbar_n=cfg.rounds,
                ).scan(
                    "Iteration",
                    list(range(num_points)),
                    before_each=update_fn,
                ),
                init_cfg=cfg,
                on_update=plot_fn,
            )
            signals = np.asarray(results)

        plt.close(fig)

        self.last_result = JPAOptimizeResult(
            params=params, phases=phases, signals=signals, cfg_snapshot=cfg
        )

        return self.last_result

    def analyze(
        self, result: JPAOptimizeResult | None = None
    ) -> tuple[float, float, float, Figure]:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        params = result.params
        phases = result.phases
        signals = result.signals
        snrs = np.abs(signals)

        max_id = np.nanargmax(snrs)
        max_snr = float(snrs[max_id])
        best_params = params[max_id, :]

        colors = phases

        figsize = (8, 5)
        fig = plt.figure(figsize=figsize)
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

        return float(best_params[0]), float(best_params[1]), float(best_params[2]), fig

    def plot_sample_params(self, result: JPAOptimizeResult | None = None) -> Figure:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        params = result.params
        phases = result.phases
        signals = result.signals
        snrs = np.abs(signals)

        max_snr = np.nanmax(snrs)
        alphas = snrs / max(max_snr, 1e-12)

        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        assert isinstance(ax, Axes3D)

        cmap = plt.get_cmap("viridis")
        norm = Normalize(vmin=float(np.nanmin(phases)), vmax=float(np.nanmax(phases)))
        colors = cmap(norm(phases))
        colors[:, 3] = alphas

        ax.scatter(params[:, 0], params[:, 1], params[:, 2], c=colors, s=0.1)  # type: ignore

        ax.set_xlabel("Flux value")
        ax.set_ylabel("Freq (MHz)")
        ax.set_zlabel("Power (dBm)")

        return fig

    def save(
        self,
        filepath: str,
        result: JPAOptimizeResult | None = None,
        comment: str | None = None,
        tag: str = "jpa/auto_optimize",
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        if result.cfg_snapshot is None:
            raise ValueError("cfg_snapshot is None")
        JPA_AUTO_GROUPED_AXES_SPEC.save_experiment_result(
            filepath,
            result,
            comment=comment,
            tag=tag,
            make_comment_fn=make_comment,
        )

    def load(self, filepath: str) -> JPAOptimizeResult:
        self.last_result = load_jpa_auto_grouped_result(filepath)
        return self.last_result
