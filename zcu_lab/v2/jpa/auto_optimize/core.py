from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
from numpy.typing import NDArray
from pydantic import Field
from zcu_tools.cfg_model import ConfigBase
from zcu_tools.datafile import LabberPayload
from zcu_tools.device import DeviceInfo
from zcu_tools.experiment import (
    MHZ_TO_HZ,
    GroupedAxesSpec,
    GroupedLoadData,
    RoleAxisSpec,
    RoleSpec,
    RoleZSpec,
)
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.utils import (
    set_flux_in_dev_cfg,
    set_freq_in_dev_cfg,
    set_power_in_dev_cfg,
    setup_devices,
)
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.snr import snr_as_signal
from zcu_tools.experiment.v2.utils.tracker.moment import MomentTracker
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    Branch,
    ProgramV2Cfg,
    Pulse,
    PulseCfg,
    Readout,
    ReadoutCfg,
    Reset,
    ResetCfg,
    SweepCfg,
)

from zcu_lab.v2.jpa.auto_optimize.optimizer import JPAOptimizer


@dataclass(frozen=True)
class JPAOptimizeResult:
    params: NDArray[np.float64]
    phases: NDArray[np.int32]
    signals: NDArray[np.float64]


@dataclass(frozen=True)
class JPAOptimizeAnalysis:
    best_flux: float
    best_freq: float
    best_power: float


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
    num_points: int = Field(ge=4, strict=True)
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


def load_jpa_auto_grouped_result(
    source: Path,
) -> RunRecord[JPAOptCfg, JPAOptimizeResult]:
    return JPA_AUTO_GROUPED_AXES_SPEC.load(source)


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


class AutoOptimizeExp:
    def run(self, cfg: JPAOptCfg, *, context: RunContext) -> JPAOptimizeResult:
        cfg = deepcopy(cfg)
        soc, soccfg = context.soc, context.soccfg
        num_points = cfg.num_points
        flux_sweep = cfg.sweep.jpa_flux
        freq_sweep = cfg.sweep.jpa_freq
        gain_sweep = cfg.sweep.jpa_power

        optimizer = JPAOptimizer(flux_sweep, freq_sweep, gain_sweep, num_points)
        params = np.full((num_points, 3), np.nan, dtype=np.float64)
        phases = np.zeros(num_points, dtype=np.int32)
        point_indices = np.arange(num_points, dtype=np.float64)
        viewers = [
            context.plots.liveplot_scatter(f"measurement.{name}", label, "SNR (a.u.)")
            for name, label in (
                ("iteration", "Iteration"),
                ("flux", "JPA Flux value (a.u.)"),
                ("freq", "JPA Frequency (MHz)"),
                ("power", "JPA Power (dBm)"),
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
            for viewer, xs in zip(
                viewers,
                (point_indices, params[:, 0], params[:, 1], params[:, 2]),
                strict=True,
            ):
                viewer.update(xs, snrs, colors=colors, title=title)

        signals_buffer = SignalBuffer(
            (num_points,), dtype=np.float64, on_update=plot_fn
        )
        with Schedule(cfg, signals_buffer, stop=context.cancel_signal) as sched:
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
                dev = step.cfg.dev
                set_flux_in_dev_cfg(dev, params[idx, 0], label="jpa_flux_dev")
                set_freq_in_dev_cfg(dev, 1e6 * params[idx, 1], label="jpa_rf_dev")
                set_power_in_dev_cfg(dev, params[idx, 2], label="jpa_rf_dev")
                setup_devices(
                    step.cfg,
                    context.devices,
                    cancel_signal=context.cancel_signal,
                    progress=False,
                )
                modules = step.cfg.modules
                tracker = MomentTracker()
                _ = (
                    step.prog_builder(soc, soccfg)
                    .add(
                        Reset("reset", modules.reset),
                        Branch("ge", [], Pulse("pi_pulse", modules.pi_pulse)),
                        Readout("readout", modules.readout),
                    )
                    .declare_sweep("ge", 2)
                    .build_and_acquire(
                        raw2signal_fn=lambda raw, tracker=tracker: snr_as_signal(
                            [tracker],
                            ge_axis=1,
                            skew_penalty=sched.cfg.skew_penalty,
                        ),
                        trackers=[tracker],
                    )
                )
        return JPAOptimizeResult(
            params=params, phases=phases, signals=signals_buffer.array
        )

    def analyze(
        self,
        source: RunRecord[JPAOptCfg, JPAOptimizeResult],
        options: None,  # noqa: ARG002 - uniform synchronous analysis interface
        *,
        plots: Plots,
    ) -> JPAOptimizeAnalysis:
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

        # Matplotlib's 3D stub narrows zs/s to int, unlike the runtime API.
        ax.scatter(params[:, 0], params[:, 1], params[:, 2], c=colors, s=0.1)  # pyright: ignore[reportArgumentType]

        ax.set_xlabel("Flux value")
        ax.set_ylabel("Freq (MHz)")
        ax.set_zlabel("Power (dBm)")

    def save(
        self,
        source: RunRecord[JPAOptCfg, JPAOptimizeResult],
        destination: Path,
        *,
        comment: str | None = None,
        tag: str = "jpa/auto_optimize",
    ) -> None:
        JPA_AUTO_GROUPED_AXES_SPEC.save(source, destination, comment=comment, tag=tag)

    def load(self, source: Path) -> RunRecord[JPAOptCfg, JPAOptimizeResult]:
        return load_jpa_auto_grouped_result(source)
