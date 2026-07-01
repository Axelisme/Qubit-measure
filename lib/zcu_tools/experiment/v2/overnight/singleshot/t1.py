from __future__ import annotations

from pathlib import Path
from typing import Generic, TypeVar, cast

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray
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
    Branch,
    Delay,
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
from zcu_tools.progress_bar import make_pbar
from zcu_tools.utils.datasaver import (
    format_ext,
    load_labber_data,
    reserve_labber_filepath,
    save_labber_data,
)
from zcu_tools.utils.fitting.multi_decay import fit_dual_transition_rates
from zcu_tools.utils.func_tools import MinIntervalFunc

from ..executor import MeasurementTask, OvernightCfg, T_RootResult
from .util import calc_populations


class T1Result(TypedDict, closed=True):
    lengths: NDArray[np.float64]
    populations: NDArray[np.float64]


class T1PlotDict(TypedDict, closed=True):
    populations_go: LivePlot2D
    populations_eo: LivePlot2D
    current_g: LivePlot1D
    current_e: LivePlot1D


T_Cfg = TypeVar("T_Cfg", bound=ExpCfgModel)


class T1PlotAndSaveMixin(Generic[T_Cfg]):
    def __init__(self, cfg: T_Cfg, cfg_model: type[T_Cfg]) -> None:
        self.cfg: T_Cfg = cfg.model_copy(deep=True)
        self.cfg_model = cfg_model

    def num_axes(self) -> dict[str, int]:
        return dict(populations_go=1, populations_eo=1, current_g=1, current_e=1)

    def make_plotter(self, name, axs) -> T1PlotDict:
        def make_2d_plotter(ax, title):
            return LivePlot2D(
                "Iteration",
                "Time (us)",
                uniform=False,
                existed_axes=[ax],
                segment_kwargs=dict(
                    title=title,
                ),
            )

        def make_1d_plotter(ax, title):
            return LivePlot1D(
                "Time (us)",
                "Population",
                existed_axes=[ax],
                segment_kwargs=dict(
                    title=title,
                    num_lines=3,
                    line_kwargs=[
                        dict(label="Ground"),
                        dict(label="Excited"),
                        dict(label="Other"),
                    ],
                ),
            )

        return T1PlotDict(
            populations_go=make_2d_plotter(axs["populations_go"], f"{name} Ground"),
            populations_eo=make_2d_plotter(axs["populations_eo"], f"{name} Other"),
            current_g=make_1d_plotter(axs["current_g"], f"{name} Init Ground"),
            current_e=make_1d_plotter(axs["current_e"], f"{name} Init Excited"),
        )

    def update_plotter(self, plotters, ctx, results) -> None:
        iters = ctx.env["iters"]
        i = ctx.env["repeat_idx"]

        lengths = results["lengths"][0]
        populations = calc_populations(results["populations"])  # (iters, 2, times, 3)

        plotters["populations_go"].update(
            iters, lengths, populations[:, 0, :, 2], refresh=False
        )
        plotters["populations_eo"].update(
            iters, lengths, populations[:, 1, :, 2], refresh=False
        )
        plotters["current_g"].update(lengths, populations[i, 0, :, :].T, refresh=False)
        plotters["current_e"].update(lengths, populations[i, 1, :, :].T, refresh=False)

    def save(self, filepath, iters, result, comment, prefix_tag) -> None:
        filepath = Path(filepath)

        lengths = result["lengths"][0]
        populations = result["populations"]  # (iters, 2, times, 2)

        comment = make_comment(self.cfg, comment)

        axes = [
            ("Iteration", "a.u.", iters),
            ("Time", "s", 1e-6 * lengths),
        ]

        # Each (suffix, sub-tag, z-slice) writes one Labber file. z on disk is
        # native (Ny, Nx) = (times, iters), so the inner Iteration axis is last;
        # the slices below are (iters, times) and need a .T to reach (times, iters).
        for suffix, sub_tag, zslice in (
            ("_gg_pop", "gg_populations", populations[:, 0, :, 0]),
            ("_ge_populations", "ge_populations", populations[:, 0, :, 1]),
            ("_eg_pop", "eg_populations", populations[:, 1, :, 0]),
            ("_ee_populations", "ee_populations", populations[:, 1, :, 1]),
        ):
            save_labber_data(
                reserve_labber_filepath(
                    str(filepath.with_name(filepath.name + suffix))
                ),
                z=("Populations", "a.u.", zslice.T),
                axes=axes,
                comment=comment,
                tags=prefix_tag + f"/{sub_tag}",
            )

    def load(self, filepath: str) -> T1Result:
        filepath = str(filepath)

        lengths: NDArray[np.float64] | None = None
        comment: str = ""
        slices: dict[tuple[int, int], NDArray[np.float64]] = {}

        # Reassemble populations (iters, 2, times, 2) from the four per-state files.
        for suffix, (j, k) in (
            ("_gg_pop", (0, 0)),
            ("_ge_populations", (0, 1)),
            ("_eg_pop", (1, 0)),
            ("_ee_populations", (1, 1)),
        ):
            ld = load_labber_data(format_ext(filepath + suffix))
            # native z is (Ny, Nx) = (times, iters); transpose to (iters, times).
            slices[(j, k)] = np.real(np.asarray(ld.z)).astype(np.float64).T
            if lengths is None:
                lengths = (np.asarray(ld.axes[1].values) * 1e6).astype(np.float64)
                comment = ld.comment

        assert lengths is not None
        n_iters, n_times = slices[(0, 0)].shape
        populations = np.zeros((n_iters, 2, n_times, 2), dtype=np.float64)
        for (j, k), zslice in slices.items():
            populations[:, j, :, k] = zslice

        if comment:
            cfg, _, _ = parse_comment(comment)
            if cfg is not None:
                self.cfg = cast(
                    T_Cfg, self.cfg_model.validate_or_warn(cfg, source=filepath)
                )
        self.result = T1Result(lengths=lengths, populations=populations)
        return self.result

    @classmethod
    def analyze(
        cls,
        name,
        iters,
        result,
        fig: Figure,
        confusion_matrix: NDArray[np.float64] | None = None,
    ) -> None:
        Ts = result["lengths"][0]  # (Ts, )
        populations = result["populations"]  # (iters, 2, Ts, 2)

        populations = calc_populations(populations)  # (iters, 2, Ts, 3)

        populations = correct_populations(populations, confusion_matrix)

        rates = np.zeros((len(iters), 6), dtype=np.float64)
        rate_errs = np.zeros((len(iters), 6), dtype=np.float64)
        pbar = make_pbar(total=len(populations), desc=name, leave=False)
        try:
            for i, pop in enumerate(populations):
                rate, rate_err, *_ = fit_dual_transition_rates(Ts, pop[0], pop[1])
                rates[i] = rate
                rate_errs[i] = rate_err
                pbar.update()
        finally:
            pbar.close()

        ax = fig.subplots(1, 1)
        assert isinstance(ax, Axes)

        show_idxs = [0, 1, 2, 4]
        rate_names = ["T_ge", "T_eg", "T_eo", "T_oe", "T_go", "T_og"]
        for i, name in enumerate(rate_names):
            if i not in show_idxs:
                continue
            ax.errorbar(iters, rates[:, i], rate_errs[:, i], capsize=1, label=name)
        ax.legend()
        ax.grid(True)


class T1ModuleCfg(ConfigBase):
    reset: ResetCfg | None = None
    pi_pulse: PulseCfg
    readout: ReadoutCfg


class T1SweepCfg(ConfigBase):
    length: SweepCfg


class T1Cfg(ProgramV2Cfg, ExpCfgModel):
    modules: T1ModuleCfg
    sweep: T1SweepCfg


class T1Task(
    T1PlotAndSaveMixin[T1Cfg], MeasurementTask[T1Result, T_RootResult, T1PlotDict]
):
    def __init__(
        self, cfg: T1Cfg, g_center: complex, e_center: complex, radius: float
    ) -> None:
        super().__init__(cfg, T1Cfg)

        setup_devices(cfg, progress=True)

        # initial values, may be rounded later
        self.lengths = sweep2array(self.cfg.sweep.length)
        self.acquire_kwargs = {
            "g_center": g_center,
            "e_center": e_center,
            "ge_radius": radius,
        }

    def init(self, dynamic_pbar: bool = False) -> None:
        pass

    def run(
        self,
        state: MeasurementContext[T1Result, T_RootResult, OvernightCfg],
    ) -> None:
        self.lengths = sweep2array(
            self.cfg.sweep.length, "time", {"soccfg": state.env["soccfg"]}
        )
        pop_ctx = state.child_with_cfg(
            "populations", self.cfg, child_type=NDArray[np.float64]
        )
        populations_buffer = context_signal_buffer(
            pop_ctx, (2, len(self.lengths), 2), dtype=np.float64
        )
        with Schedule(
            self.cfg, populations_buffer, env_dict=state.env, stop=state.stop
        ) as sched:
            cfg = sched.cfg
            modules = cfg.modules
            length_sweep = cfg.sweep.length
            len_param = sweep2param("length", length_sweep)

            _ = (
                sched.prog_builder(state.env["soc"], state.env["soccfg"])
                .add(
                    Reset("reset", modules.reset),
                    Branch(
                        "ge",
                        [],
                        [
                            Pulse("pi_pulse", modules.pi_pulse),
                            Delay("t1_delay", delay=len_param),
                        ],
                    ),
                    Readout("readout", modules.readout),
                )
                .declare_sweep("ge", 2)
                .declare_sweep("length", length_sweep)
                .build_and_acquire(
                    raw2signal_fn=lambda raw: raw[0][0],
                    **self.acquire_kwargs,
                )
            )

        with MinIntervalFunc.force_execute():
            state.set_value(
                T1Result(
                    lengths=self.lengths,
                    populations=state.value["populations"],
                )
            )

    def get_default_result(self) -> T1Result:
        return T1Result(
            lengths=self.lengths,
            populations=np.full((2, len(self.lengths), 2), np.nan, dtype=np.float64),
        )

    def cleanup(self) -> None:
        pass


class T1WithToneModuleCfg(ConfigBase):
    reset: ResetCfg | None = None
    pi_pulse: PulseCfg
    probe_pulse: PulseCfg
    readout: ReadoutCfg


class T1WithToneSweepCfg(ConfigBase):
    length: SweepCfg


class T1WithToneCfg(ProgramV2Cfg, ExpCfgModel):
    modules: T1WithToneModuleCfg
    sweep: T1WithToneSweepCfg


class T1WithToneTask(
    T1PlotAndSaveMixin[T1WithToneCfg],
    MeasurementTask[T1Result, T_RootResult, T1PlotDict],
):
    def __init__(
        self, cfg: T1WithToneCfg, g_center: complex, e_center: complex, radius: float
    ) -> None:
        super().__init__(cfg, T1WithToneCfg)

        # initial values, may be rounded later
        self.lengths = sweep2array(self.cfg.sweep.length)
        self.acquire_kwargs = {
            "g_center": g_center,
            "e_center": e_center,
            "ge_radius": radius,
        }

    def init(self, dynamic_pbar: bool = False) -> None:
        pass

    def run(
        self,
        state: MeasurementContext[T1Result, T_RootResult, OvernightCfg],
    ) -> None:
        self.lengths = sweep2array(
            self.cfg.sweep.length, "time", {"soccfg": state.env["soccfg"]}
        )
        pop_ctx = state.child_with_cfg(
            "populations", self.cfg, child_type=NDArray[np.float64]
        )
        populations_buffer = context_signal_buffer(
            pop_ctx, (2, len(self.lengths), 2), dtype=np.float64
        )
        with Schedule(
            self.cfg, populations_buffer, env_dict=state.env, stop=state.stop
        ) as sched:
            cfg = sched.cfg
            modules = cfg.modules
            length_sweep = cfg.sweep.length
            length_param = sweep2param("length", length_sweep)
            modules.probe_pulse.set_param("length", length_param)

            _ = (
                sched.prog_builder(state.env["soc"], state.env["soccfg"])
                .add(
                    Reset("reset", modules.reset),
                    Branch(
                        "ge",
                        Pulse("probe_pulse_g", modules.probe_pulse),
                        [
                            Pulse("pi_pulse", modules.pi_pulse),
                            Pulse("probe_pulse", modules.probe_pulse),
                        ],
                    ),
                    Readout("readout", modules.readout),
                )
                .declare_sweep("ge", 2)
                .declare_sweep("length", length_sweep)
                .build_and_acquire(
                    raw2signal_fn=lambda raw: raw[0][0],
                    **self.acquire_kwargs,
                )
            )

        with MinIntervalFunc.force_execute():
            state.set_value(
                T1Result(
                    lengths=self.lengths,
                    populations=state.value["populations"],
                )
            )

    def get_default_result(self) -> T1Result:
        return T1Result(
            lengths=self.lengths,
            populations=np.full((2, len(self.lengths), 2), np.nan, dtype=np.float64),
        )

    def cleanup(self) -> None:
        pass
