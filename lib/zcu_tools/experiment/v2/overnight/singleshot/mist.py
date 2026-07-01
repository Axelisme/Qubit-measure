from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import NonUniformImage
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter
from typing_extensions import (
    TypedDict,  # closed/extra_items (PEP 728) not in stdlib 3.13
)

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import make_comment, parse_comment, setup_devices
from zcu_tools.experiment.v2.runner import Schedule
from zcu_tools.experiment.v2.runner.multi_executor import (
    MeasurementContext,
    context_signal_buffer,
)
from zcu_tools.experiment.v2.singleshot.util import correct_populations
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.liveplot import LivePlot1D, LivePlot2D
from zcu_tools.program.v2 import (
    ProgramV2Cfg,
    Pulse,
    PulseCfg,
    Readout,
    ReadoutCfg,
    Reset,
    ResetCfg,
    SweepCfg,
    sweep2param,
)
from zcu_tools.utils.datasaver import (
    load_labber_data,
    reserve_labber_filepath,
    save_labber_data,
)
from zcu_tools.utils.func_tools import MinIntervalFunc

from ..executor import MeasurementTask, OvernightCfg, T_RootResult
from .util import calc_populations


class MistResult(TypedDict, closed=True):
    gains: NDArray[np.float64]  # (N,)
    populations: NDArray[np.float64]  # (N, 2)


class MistPlotDict(TypedDict, closed=True):
    populations_g: LivePlot2D
    populations_e: LivePlot2D
    populations_o: LivePlot2D
    current: LivePlot1D


class MistModuleCfg(ConfigBase):
    reset: ResetCfg | None = None
    init_pulse: PulseCfg | None = None
    probe_pulse: PulseCfg
    readout: ReadoutCfg


class MistSweepCfg(ConfigBase):
    gain: SweepCfg


class MistCfg(ProgramV2Cfg, ExpCfgModel):
    modules: MistModuleCfg
    sweep: MistSweepCfg


class MistOvernightAnalyzer:
    def __init__(self) -> None:
        self.cfg: MistCfg | None = None
        self.result: MistResult | None = None

    def analyze(
        self,
        fig: Figure,
        result: MistResult | None = None,
        ac_coeff: float | None = None,
        confusion_matrix: NDArray[np.float64] | None = None,
        drop_extreme_num: int = 0,
        cutoff: float | None = None,
    ) -> None:
        if result is None:
            result = self.result
        assert result is not None, "no result found"

        gains = result["gains"][0]  # (Ts, )
        populations = result["populations"]  # (iters, Ts, 2)

        valid_mask = np.all(np.isfinite(populations), axis=(1, 2))
        populations = populations[valid_mask]

        if cutoff is not None:
            cutoff_idx = np.argmin(np.abs(gains - cutoff))
            gains = gains[:cutoff_idx]
            populations = populations[:, :cutoff_idx]

        iter = populations.shape[0]

        populations = np.real(populations).astype(np.float64)

        populations = gaussian_filter(populations, sigma=0.5, axes=(0, 1))

        populations = calc_populations(populations)  # (iters, Ts, 3)
        populations = correct_populations(populations, confusion_matrix)

        sort_populations = np.sort(populations, axis=0)
        max_populations = sort_populations[iter - drop_extreme_num - 1]
        min_populations = sort_populations[drop_extreme_num]
        med_populations = np.mean(
            sort_populations[drop_extreme_num : iter - drop_extreme_num], axis=0
        )
        std_populations = np.std(
            sort_populations[drop_extreme_num : iter - drop_extreme_num], axis=0
        )

        if ac_coeff is None:
            xs = gains
            xlabel = "probe gain (a.u.)"
        else:
            xs = ac_coeff * gains**2
            xlabel = r"$\bar n$"

        ax = fig.add_subplot(1, 1, 1)

        avg_kwargs = dict(marker=".", linestyle="-", markersize=4)
        side_kwargs = dict(linestyle="--", alpha=0.3)
        ax.fill_between(
            xs, min_populations[:, 0], max_populations[:, 0], color="b", alpha=0.1
        )
        ax.plot(xs, max_populations[:, 0], color="b", **side_kwargs)  # type: ignore
        ax.errorbar(
            xs,
            med_populations[:, 0],
            yerr=std_populations[:, 0],
            color="b",
            **avg_kwargs,  # type: ignore
        )
        ax.plot(xs, min_populations[:, 0], color="b", **side_kwargs)  # type: ignore

        ax.fill_between(
            xs, min_populations[:, 1], max_populations[:, 1], color="r", alpha=0.1
        )
        ax.plot(xs, max_populations[:, 1], color="r", **side_kwargs)  # type: ignore
        ax.errorbar(
            xs,
            med_populations[:, 1],
            yerr=std_populations[:, 1],
            color="r",
            **avg_kwargs,  # type: ignore
        )
        ax.plot(xs, min_populations[:, 1], color="r", **side_kwargs)  # type: ignore

        ax.fill_between(
            xs, min_populations[:, 2], max_populations[:, 2], color="g", alpha=0.1
        )
        ax.plot(xs, max_populations[:, 2], color="g", **side_kwargs)  # type: ignore
        ax.errorbar(
            xs,
            med_populations[:, 2],
            yerr=std_populations[:, 2],
            color="g",
            **avg_kwargs,  # type: ignore
        )
        ax.plot(xs, min_populations[:, 2], color="g", **side_kwargs)  # type: ignore

        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel("Population", fontsize=14)
        ax.set_xlim(xs[0], xs[-1])
        ax.set_ylim(0, 1)
        ax.grid(True)

    def plot(
        self,
        fig: Figure,
        result: MistResult | None = None,
        ac_coeff: float | None = None,
        confusion_matrix: NDArray[np.float64] | None = None,
        cutoff: float | None = None,
    ) -> None:
        if result is None:
            result = self.result
        assert result is not None, "no result found"

        gains = result["gains"][0]  # (Ts, )
        populations = result["populations"]  # (iters, Ts, 2)

        valid_mask = np.all(np.isfinite(populations), axis=(1, 2))
        populations = populations[valid_mask]

        if cutoff is not None:
            cutoff_idx = np.argmin(np.abs(gains - cutoff))
            gains = gains[:cutoff_idx]
            populations = populations[:, :cutoff_idx]

        iterations = np.arange(populations.shape[0])

        populations = np.real(populations).astype(np.float64)

        populations = gaussian_filter(populations, sigma=0.5, axes=(0, 1))

        populations = calc_populations(populations)  # (iters, Ts, 3)
        populations = correct_populations(populations, confusion_matrix)

        if ac_coeff is None:
            xs = gains
            xlabel = "probe gain (a.u.)"
        else:
            xs = ac_coeff * gains**2
            xlabel = r"$\bar n$"

        ax_g, ax_e, ax_o = fig.subplots(3, 1, sharex=True)  # type: ignore

        im_g = NonUniformImage(ax_g, cmap="RdBu_r")
        im_g.set_data(xs, iterations, populations[..., 0])
        im_g.set_extent((xs[0], xs[-1], iterations[0], iterations[-1]))
        ax_g.add_image(im_g)
        ax_g.scatter([], [], color="b", marker=".", s=1, label="Ground")
        ax_g.set_ylabel("Iteration", fontsize=14)
        ax_g.set_aspect("auto")
        ax_g.legend(fontsize=8)

        im_e = NonUniformImage(ax_e, cmap="RdBu_r")
        im_e.set_data(xs, iterations, populations[..., 1])
        im_e.set_extent((xs[0], xs[-1], iterations[0], iterations[-1]))
        ax_e.add_image(im_e)
        ax_e.scatter([], [], color="r", marker=".", s=1, label="Excited")
        ax_e.set_ylabel("Iteration", fontsize=14)
        ax_e.set_aspect("auto")
        ax_e.legend(fontsize=8)

        im_o = NonUniformImage(ax_o, cmap="RdBu_r")
        im_o.set_data(xs, iterations, populations[..., 2])
        im_o.set_extent((xs[0], xs[-1], iterations[0], iterations[-1]))
        ax_o.add_image(im_o)
        ax_o.scatter([], [], color="g", marker=".", s=1, label="Other")
        ax_o.set_xlabel(xlabel, fontsize=14)
        ax_o.set_ylabel("Iteration", fontsize=14)
        ax_o.set_aspect("auto")
        ax_o.legend(fontsize=8)

    @classmethod
    def save(
        cls,
        filepath: str,
        iters,
        result,
        cfg: MistCfg,
        comment: str | None,
        prefix_tag: str,
    ) -> None:
        _filepath = Path(filepath)

        comment = make_comment(cfg, comment)

        gains = result["gains"][0]
        populations = result["populations"]

        # native z is (Ny=iters, Nx=gains), inner axis (x) = gains, outer (y) = iters
        axes = [
            ("Readout gain", "a.u.", gains),  # inner (x)
            ("Iteration", "a.u.", iters),  # outer (y)
        ]

        # g_populations
        save_labber_data(
            reserve_labber_filepath(
                str(_filepath.with_name(_filepath.name + "_g_populations"))
            ),
            z=("Populations", "a.u.", populations[..., 0]),
            axes=axes,
            comment=comment,
            tags=prefix_tag + "/g_populations",
        )

        # e_populations
        save_labber_data(
            reserve_labber_filepath(
                str(_filepath.with_name(_filepath.name + "_e_populations"))
            ),
            z=("Populations", "a.u.", populations[..., 1]),
            axes=axes,
            comment=comment,
            tags=prefix_tag + "/e_populations",
        )

    def load(self, filepath: list[str]) -> MistResult:
        g_filepath, e_filepath = filepath

        g_ld = load_labber_data(g_filepath)
        g_pops = np.real(np.asarray(g_ld.z)).astype(np.float64)
        comment = g_ld.comment
        gains = np.asarray(g_ld.axes[0].values).astype(np.float64)  # inner (x)
        iters = np.asarray(g_ld.axes[1].values)  # outer (y)
        assert g_pops.shape == (len(iters), len(gains))

        e_ld = load_labber_data(e_filepath)
        e_pops = np.real(np.asarray(e_ld.z)).astype(np.float64)
        assert e_pops.shape == (len(iters), len(gains))

        populations = np.stack([g_pops, e_pops], axis=-1)  # (iters, gains, 2)
        gains = np.tile(gains, reps=(len(iters), 1))

        if comment:
            cfg, _, _ = parse_comment(comment)
            if cfg is not None:
                self.cfg = MistCfg.validate_or_warn(cfg, source=g_filepath)
        self.result = MistResult(gains=gains, populations=populations)

        return self.result


class MistTask(MeasurementTask[MistResult, T_RootResult, MistPlotDict]):
    def __init__(
        self, cfg: MistCfg, g_center: complex, e_center: complex, radius: float
    ) -> None:
        self.cfg = cfg
        self.last_cfg = cfg.model_copy(deep=True)

        setup_devices(self.cfg, progress=True)

        # initial values, may be rounded later
        self.gains = sweep2array(self.cfg.sweep.gain)
        self.acquire_kwargs = {
            "g_center": g_center,
            "e_center": e_center,
            "ge_radius": radius,
        }

    def init(self, dynamic_pbar: bool = False) -> None:
        pass

    def run(
        self,
        state: MeasurementContext[MistResult, T_RootResult, OvernightCfg],
    ) -> None:
        self.gains = sweep2array(
            self.gains,
            "gain",
            {"soccfg": state.env["soccfg"], "gen_ch": self.cfg.modules.probe_pulse.ch},
        )
        pop_ctx = state.child_with_cfg(
            "populations", self.cfg, child_type=NDArray[np.float64]
        )
        populations_buffer = context_signal_buffer(
            pop_ctx, (len(self.gains), 2), dtype=np.float64
        )
        with Schedule(
            self.cfg, populations_buffer, env_dict=state.env, stop=state.stop
        ) as sched:
            cfg = sched.cfg
            modules = cfg.modules

            gain_sweep = cfg.sweep.gain
            modules.probe_pulse.set_param("gain", sweep2param("gain", gain_sweep))

            _ = (
                sched.prog_builder(state.env["soc"], state.env["soccfg"])
                .add(
                    Reset("reset", modules.reset),
                    Pulse("init_pulse", cfg=modules.init_pulse),
                    Pulse("probe_pulse", cfg=modules.probe_pulse),
                    Readout("readout", modules.readout),
                )
                .declare_sweep("gain", gain_sweep)
                .build_and_acquire(
                    raw2signal_fn=lambda raw: raw[0][0],
                    **self.acquire_kwargs,
                )
            )

        with MinIntervalFunc.force_execute():
            state.set_value(
                MistResult(
                    gains=self.gains,
                    populations=state.value["populations"],
                )
            )

    def get_default_result(self) -> MistResult:
        return MistResult(
            gains=self.gains,
            populations=np.full((len(self.gains), 2), np.nan, dtype=np.float64),
        )

    def cleanup(self) -> None:
        pass

    def num_axes(self) -> dict[str, int]:
        return dict(populations_g=1, populations_e=1, populations_o=1, current=1)

    def make_plotter(self, name: str, axs: dict[str, list[Axes]]) -> MistPlotDict:
        return MistPlotDict(
            populations_g=LivePlot2D(
                "Iteration",
                "Time (us)",
                uniform=False,
                existed_axes=[axs["populations_g"]],
                segment_kwargs=dict(
                    title=f"{name} Ground",
                ),
            ),
            populations_e=LivePlot2D(
                "Iteration",
                "Time (us)",
                uniform=False,
                existed_axes=[axs["populations_e"]],
                segment_kwargs=dict(
                    title=f"{name} Excited",
                ),
            ),
            populations_o=LivePlot2D(
                "Iteration",
                "Time (us)",
                uniform=False,
                existed_axes=[axs["populations_o"]],
                segment_kwargs=dict(
                    title=f"{name} Other",
                ),
            ),
            current=LivePlot1D(
                "Time (us)",
                "Population",
                existed_axes=[axs["current"]],
                segment_kwargs=dict(
                    title=f"{name} Current",
                    num_lines=3,
                    line_kwargs=[
                        dict(label="Ground"),
                        dict(label="Excited"),
                        dict(label="Other"),
                    ],
                ),
            ),
        )

    def update_plotter(
        self,
        plotters,
        ctx: MeasurementContext[NDArray[np.float64], T_RootResult, OvernightCfg],
        results: MistResult,
    ) -> None:
        iters = ctx.env["iters"]
        i = ctx.env["repeat_idx"]

        gains = results["gains"][0]
        populations = calc_populations(results["populations"])  # (iters, times, 3)

        plotters["populations_g"].update(
            iters, gains, populations[..., 0], refresh=False
        )
        plotters["populations_e"].update(
            iters, gains, populations[..., 1], refresh=False
        )
        plotters["populations_o"].update(
            iters, gains, populations[..., 2], refresh=False
        )
        plotters["current"].update(gains, populations[i].T, refresh=False)

    def analyze(
        self, name: str, iters: NDArray[np.int64], result: MistResult, **kwargs
    ) -> None:
        MistOvernightAnalyzer().analyze(result=result, **kwargs)

    def save(self, filepath, iters, result, comment, prefix_tag) -> None:
        cfg = self.last_cfg
        assert cfg is not None
        MistOvernightAnalyzer.save(filepath, iters, result, cfg, comment, prefix_tag)
