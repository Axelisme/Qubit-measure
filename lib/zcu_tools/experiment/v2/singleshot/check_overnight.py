from copy import deepcopy
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from numpy.typing import NDArray
from typing_extensions import NotRequired, Optional, Tuple, cast, Callable

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


def calc_population_mask(
    signals: NDArray[np.float64], g_center: complex, e_center: complex, radius: float
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


class CheckOvernightTaskConfig(TaskConfig, ModularProgramCfg):
    reset: NotRequired[ResetCfg]
    init_pulse: NotRequired[PulseCfg]
    probe_pulse: PulseCfg
    readout: ReadoutCfg

    shots: int
    interval: float


class CheckOvernightExperiment(AbsExperiment):
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

        def measure_fn(ctx: TaskContextView, update_hook: Callable):
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

            prog.acquire(
                soc,
                progress=False,
                update_hook=lambda i, *_: update_hook(i, _get_shot_iq(prog)),
            )

            return _get_shot_iq(prog)  # (reps, 1, 2)

        def raw2signal_fn(shot_iq: NDArray[np.float64]) -> NDArray[np.complex128]:
            i0, q0 = shot_iq[..., 0, 0], shot_iq[..., 0, 1]  # (reps, )
            return np.array(i0 + 1j * q0)  # (reps, )

        iters = np.arange(num_times, dtype=np.int64)
        shots = np.arange(cfg["shots"], dtype=np.int64)

        fig, axs = make_plot_frame(1, 2, figsize=(12, 5))
        ax_scatter = axs[0][0]
        ax_1d = axs[0][1]

        ax_scatter.set_aspect("equal")

        # plot centers with circle
        plt_params = dict(linestyle=":", color="k", marker="o", markersize=5)
        ax_scatter.plot(
            g_center.real,
            g_center.imag,
            markerfacecolor="b",
            label="Ground",
            **plt_params,  # type: ignore
        )
        ax_scatter.plot(
            e_center.real,
            e_center.imag,
            markerfacecolor="r",
            label="Excited",
            **plt_params,  # type: ignore
        )
        ax_scatter.add_patch(
            Circle(
                (g_center.real, g_center.imag),
                radius,
                color="b",
                fill=False,
                linestyle="--",
            )
        )
        ax_scatter.add_patch(
            Circle(
                (e_center.real, e_center.imag),
                radius,
                color="r",
                fill=False,
                linestyle="--",
            )
        )

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
            downsample_mask = np.arange(
                0, cfg["shots"], max(1, cfg["shots"] // downsample_num)
            )

            def plot_fn(ctx) -> None:
                i = ctx.env_dict["repeat_idx"]

                signals = np.asarray(ctx.data)  # (iters, shots)
                masks = calc_population_mask(
                    signals, g_center, e_center, radius
                )  # (iters, shots, 3)

                signals_i = signals[i]  # (shots, )
                populations = np.mean(masks, axis=1, dtype=np.float64)  # (iters, 3)

                masks_i = masks[i].astype(bool)  # (shots, 3)
                colors_i = np.full_like(signals_i, "gray", dtype=object)
                colors_i[masks_i[:, 0]] = "blue"
                colors_i[masks_i[:, 1]] = "red"
                colors_i[masks_i[:, 2]] = "green"

                viewer.get_plotter("plot_scatter").update(
                    signals_i[downsample_mask].real,
                    signals_i[downsample_mask].imag,
                    colors=colors_i[downsample_mask].tolist(),
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
        result: Optional[CheckOvernightResultType] = None,
        *,
        confusion_matrix: Optional[NDArray[np.float64]] = None,
    ) -> Figure:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        raise NotImplementedError("analyze method is not implemented yet")

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
