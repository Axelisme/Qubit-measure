from copy import deepcopy
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from matplotlib.axes import Axes
from numpy.typing import NDArray
from typing_extensions import NotRequired, Optional, Tuple, cast

from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.v2.runner import (
    HardTask,
    RepeatOverTime,
    ReTryIfFail,
    TaskConfig,
    run_task,
    TaskContextView,
)
from zcu_tools.liveplot import (
    LivePlotter1D,
    LivePlotterScatter,
    MultiLivePlotter,
    make_plot_frame,
)
from zcu_tools.program.v2 import (
    ModularProgramCfg,
    ModularProgramV2,
    Pulse,
    PulseCfg,
    Readout,
    ReadoutCfg,
    Reset,
    ResetCfg,
)
from zcu_tools.utils.datasaver import load_data, save_data

CheckOvernightResultType = Tuple[
    NDArray[np.int64], NDArray[np.int64], NDArray[np.complex128]
]


def raw2signal_fn(shot_iq: NDArray[np.float64]) -> NDArray[np.complex128]:
    i0, q0 = shot_iq[..., 0, 0], shot_iq[..., 0, 1]  # (reps, )
    return np.array(i0 + 1j * q0)  # (reps, )


def calc_population_mask(
    signals: NDArray[np.complex128], g_center: complex, e_center: complex, radius: float
) -> NDArray[np.float64]:
    masks = np.full((*signals.shape, 3), np.nan, dtype=np.float64)

    valid_mask = np.isfinite(signals)
    if not np.any(valid_mask):
        return masks

    dists_g = np.abs(signals[valid_mask] - g_center)
    dists_e = np.abs(signals[valid_mask] - e_center)
    mask_g = dists_g < radius
    mask_e = dists_e < radius
    mask_o = ~(mask_g | mask_e)

    masks[valid_mask, 0] = mask_g
    masks[valid_mask, 1] = mask_e
    masks[valid_mask, 2] = mask_o

    return masks


def _plot_circle(
    ax: Axes, center: complex, radius: float, label: str, color: str
) -> None:
    ax.plot(
        center.real,
        center.imag,
        markerfacecolor=color,
        label=label,
        linestyle=":",
        color="k",
        marker="o",
        markersize=5,
    )
    ax.add_patch(
        Circle(
            (center.real, center.imag),
            radius,
            color=color,
            fill=False,
            linestyle="--",
        )
    )


def _calc_mask(
    signals: NDArray[np.complex128], g_center: complex, e_center: complex, radius: float
) -> NDArray[np.bool_]:
    masks = np.full((*signals.shape, 3), np.nan, dtype=np.bool_)

    valid_mask = np.isfinite(signals)
    if not np.any(valid_mask):
        return masks

    dists_g = np.abs(signals[valid_mask] - g_center)
    dists_e = np.abs(signals[valid_mask] - e_center)
    mask_g = dists_g < radius
    mask_e = dists_e < radius
    mask_o = ~(mask_g | mask_e)

    masks[valid_mask, 0] = mask_g
    masks[valid_mask, 1] = mask_e
    masks[valid_mask, 2] = mask_o

    return masks


def _calc_populations(masks: NDArray[np.bool_]) -> NDArray[np.float64]:
    float_masks = masks.astype(np.float64)
    return np.mean(float_masks, axis=-1, dtype=np.float64)


def _calc_color(
    signals: NDArray[np.complex128], masks: NDArray[np.bool_]
) -> NDArray[np.object_]:
    colors = np.full_like(signals, "gray", dtype=object)
    colors[masks[:, 0]] = "blue"
    colors[masks[:, 1]] = "red"
    colors[masks[:, 2]] = "green"

    return colors


class CheckOvernightTaskConfig(TaskConfig, ModularProgramCfg):
    reset: NotRequired[ResetCfg]
    init_pulse: NotRequired[PulseCfg]
    probe_pulse: PulseCfg
    readout: ReadoutCfg

    shots: int
    interval: float


class CheckOvernightExperiment(
    AbsExperiment[CheckOvernightResultType, CheckOvernightTaskConfig]
):
    def run(
        self,
        soc,
        soccfg,
        cfg: CheckOvernightTaskConfig,
        g_center: complex,
        e_center: complex,
        radius: float,
        *,
        num_times: int = 50,
        fail_retry: int = 3,
    ) -> CheckOvernightResultType:
        cfg = deepcopy(cfg)  # avoid in-place modification

        # Validate and setup configuration
        if cfg.setdefault("rounds", 1) != 1:
            warnings.warn("rounds will be overwritten to 1 for singleshot measurement")
            cfg["rounds"] = 1

        if "reps" in cfg:
            warnings.warn("reps will be overwritten by singleshot measurement shots")
        cfg["reps"] = cfg["shots"]

        def _get_shot_iq(prog: ModularProgramV2):
            acc_buf = prog.get_raw()
            assert acc_buf is not None

            length = cast(int, list(prog.ro_chs.values())[0]["length"])

            return acc_buf[0] / length  # (reps, 1, 2)

        def measure_fn(ctx: TaskContextView, _):
            prog = ModularProgramV2(
                soccfg,
                ctx.cfg,
                modules=[
                    Reset("reset", ctx.cfg.get("reset", {"type": "none"})),
                    Pulse("init_pulse", ctx.cfg.get("init_pulse")),
                    Pulse("probe_pulse", ctx.cfg["probe_pulse"]),
                    Readout("readout", ctx.cfg["readout"]),
                ],
            )
            prog.acquire(soc, progress=False)

            return _get_shot_iq(prog)  # (reps, 1, 2)

        iters = np.arange(num_times, dtype=np.int64)
        shots = np.arange(cfg["shots"], dtype=np.int64)

        fig, axs = make_plot_frame(1, 2, figsize=(12, 5))
        ax_scatter = axs[0][0]
        ax_1d = axs[0][1]

        ax_scatter.set_aspect("equal")
        _plot_circle(ax_scatter, g_center, radius, "Ground State", "blue")
        _plot_circle(ax_scatter, e_center, radius, "Excited State", "red")

        with MultiLivePlotter(
            fig,
            dict(
                plot_scatter=LivePlotterScatter(
                    "I value (a.u.)", "Q value (a.u.)", existed_axes=[[ax_scatter]]
                ),
                plot_1d=LivePlotter1D(
                    "Iteration",
                    "Population",
                    existed_axes=[[ax_1d]],
                    segment_kwargs=dict(
                        num_lines=3,
                        line_kwargs=[
                            dict(label="Ground"),
                            dict(label="Excited"),
                            dict(label="Other"),
                        ],
                    ),
                ),
            ),
        ) as viewer:
            downsample_num = min(10000, cfg["shots"])
            ds_idxs = np.arange(0, cfg["shots"], max(1, cfg["shots"] // downsample_num))

            def plot_fn(ctx) -> None:
                i = ctx.env_dict["repeat_idx"]

                signals = np.asarray(ctx.data)  # (iters, shots)
                masks = _calc_mask(
                    signals, g_center, e_center, radius
                )  # (iters, shots, 3)

                populations = _calc_populations(masks)  # (iters, 3)

                signals_i = signals[i]  # (shots, )
                colors_i = _calc_color(signals_i, masks[i])  # (shots, )

                viewer.get_plotter("plot_scatter").update(
                    signals_i[ds_idxs].real,
                    signals_i[ds_idxs].imag,
                    colors=colors_i[ds_idxs].tolist(),
                    refresh=False,
                )
                viewer.get_plotter("plot_1d").update(
                    iters, populations.T, refresh=False
                )

                viewer.refresh()

            signals = run_task(
                task=RepeatOverTime(
                    name="repeat_over_time",
                    num_times=num_times,
                    interval=cfg["interval"],
                    task=ReTryIfFail(
                        max_retries=fail_retry,
                        task=HardTask(
                            measure_fn=measure_fn,
                            raw2signal_fn=raw2signal_fn,
                            result_shape=(cfg["shots"],),
                        ),
                    ),
                ),
                init_cfg=cfg,
                update_hook=plot_fn,
            )
            signals = np.asarray(signals)
        plt.close(fig)

        # record the last result
        self.last_cfg = cfg
        self.last_result: CheckOvernightResultType = (iters, shots, signals)

        return iters, shots, signals

    def analyze(
        self,
        g_center: complex,
        e_center: complex,
        radius: float,
        result: Optional[CheckOvernightResultType] = None,
        *,
        confusion_matrix: Optional[NDArray[np.float64]] = None,
    ) -> Figure:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        iters, shots, signals = result

        masks = _calc_mask(signals, g_center, e_center, radius)  # (iters, shots, 3)

        populations = _calc_populations(masks)  # (iters, 3)

        if confusion_matrix is not None:  # readout correction
            populations = populations @ np.linalg.inv(confusion_matrix)
            populations = np.clip(populations, 0.0, 1.0)

        med_populations = np.median(populations, axis=0, keepdims=True)  # (1, 3)
        delta_populations = np.sum(np.abs(populations - med_populations), axis=-1)

        max_idx = np.argmax(delta_populations).item()
        min_idx = np.argmin(delta_populations).item()
        max_iter = iters[max_idx]
        min_iter = iters[min_idx]

        downdample_num = min(10000, shots.shape[0])
        ds_idxs = np.arange(0, shots.shape[0], max(1, shots.shape[0] // downdample_num))

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 8))

        ax1.set_title("Populations over time")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Population")
        ax1.plot(iters, populations[:, 0], label="Ground", color="blue")
        ax1.plot(iters, populations[:, 1], label="Excited", color="red")
        ax1.plot(iters, populations[:, 2], label="Other", color="green")
        ax1.axvline(max_iter, color="black", linestyle="--", label="Slice 1")
        ax1.axvline(min_iter, color="gray", linestyle="--", label="Slice 2")
        ax1.legend()
        ax1.grid(True)

        def _plot_slice(ax: Axes, iter_idx: int, name: str) -> None:
            signals_i = signals[iter_idx]  # (shots, )
            masks_i = masks[iter_idx]  # (shots, 3)
            population_i = populations[iter_idx]  # (3, )

            ax.set_title(
                f"{name}: Iteration {iters[iter_idx]}, "
                f"Population: ({population_i[0]:.2g}/{population_i[1]:.2g}/{population_i[2]:.2g})"
            )
            _plot_circle(ax, g_center, radius, "Ground State", "blue")
            _plot_circle(ax, e_center, radius, "Excited State", "red")
            colors_i = _calc_color(signals_i, masks_i)  # (shots, )
            ax.scatter(
                signals_i[ds_idxs].real,
                signals_i[ds_idxs].imag,
                c=colors_i[ds_idxs].tolist(),
                s=5,
                alpha=0.7,
            )
            ax.set_xlabel("I value (a.u.)")
            ax.set_ylabel("Q value (a.u.)")
            ax.set_aspect("equal")
            ax.legend()
            ax.grid(True)

        _plot_slice(ax2, max_idx, "Slice 1")
        _plot_slice(ax3, min_idx, "Slice 2")

        return fig

    def save(
        self,
        filepath: str,
        result: Optional[CheckOvernightResultType] = None,
        comment: Optional[str] = None,
        tag: str = "singleshot/check_overnight",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        iters, shots, signals = result

        _filepath = Path(filepath)

        save_data(
            filepath=str(_filepath.with_name(_filepath.name + "_g")),
            x_info={"name": "Shot Index", "unit": "None", "values": shots},
            y_info={"name": "Iteration", "unit": "None", "values": iters},
            z_info={"name": "Ground Population", "unit": "a.u.", "values": signals},
            comment=comment,
            tag=tag,
            **kwargs,
        )

    def load(self, filepath: str, **kwargs) -> CheckOvernightResultType:
        signals, shots, iters = load_data(filepath, **kwargs)
        assert iters is not None
        assert signals.shape == (len(iters), len(shots))

        signals = signals.T

        signals = signals.astype(np.complex128)
        iters = iters.astype(np.int64)
        shots = shots.astype(np.int64)

        self.last_cfg = None
        self.last_result = (iters, shots, signals)

        return iters, shots, signals
