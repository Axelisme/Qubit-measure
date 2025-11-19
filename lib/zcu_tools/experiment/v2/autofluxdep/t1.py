from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional, TypedDict

import numpy as np
from typing_extensions import NotRequired

from zcu_tools.experiment.utils import sweep2array
from zcu_tools.experiment.v2.runner import HardTask, TaskContext
from zcu_tools.experiment.v2.utils import wrap_earlystop_check
from zcu_tools.library import ModuleLibrary
from zcu_tools.liveplot import LivePlotter1D, LivePlotter2DwithLine
from zcu_tools.program.v2 import (
    Delay,
    ModularProgramV2,
    Pulse,
    PulseCfg,
    Readout,
    ReadoutCfg,
    Reset,
    ResetCfg,
    sweep2param,
)
from zcu_tools.utils import deepupdate
from zcu_tools.utils.datasaver import save_data
from zcu_tools.utils.fitting import fit_decay
from zcu_tools.utils.process import rotate2real

from .executor import MeasurementTask


def t1_signal2real(signals: np.ndarray) -> np.ndarray:
    return rotate2real(signals).real


def t1_fluxdep_signal2real(signals: np.ndarray) -> np.ndarray:
    return np.array(list(map(t1_signal2real, signals)), dtype=np.float64)


class T1Cfg(TypedDict, total=False):
    reset: NotRequired[ResetCfg]
    pi_pulse: PulseCfg
    readout: ReadoutCfg
    relax_delay: float
    reps: int
    rounds: int


class T1Result(TypedDict):
    raw_signals: np.ndarray
    t1: float
    t1_err: float
    success: bool


class PlotterDictType(TypedDict):
    t1: LivePlotter1D
    t1_curve: LivePlotter2DwithLine


class T1MeasurementTask(MeasurementTask[T1Result, PlotterDictType]):
    def __init__(
        self,
        length_sweep: dict,
        cfg_maker: Callable[[TaskContext, ModuleLibrary], T1Cfg],
        earlystop_snr: Optional[float] = None,
    ) -> None:
        self.length_sweep = length_sweep
        self.cfg_maker = cfg_maker
        self.earlystop_snr = earlystop_snr

        def measure_t1_fn(ctx: TaskContext, update_hook: Callable):
            import time

            from zcu_tools.utils.fitting.base import expfunc

            lengths = sweep2array(self.length_sweep)
            for i in range(ctx.cfg["rounds"]):
                raw_signals = [
                    [
                        np.stack(
                            [
                                expfunc(
                                    lengths,
                                    0,
                                    1,
                                    2 * (ctx.env_dict["flx_value"] ** 2 + 1.0),
                                )
                                + 0.01
                                * (ctx.cfg["rounds"] - i)
                                * np.random.randn(len(lengths)),
                                np.zeros_like(lengths),
                            ],
                            axis=1,
                        )
                    ]
                ]
                update_hook(i, raw_signals)
                time.sleep(0.01)

            return raw_signals

        self.task = HardTask(
            measure_fn=measure_t1_fn,
            # measure_fn=lambda ctx, update_hook: (
            #     prog := ModularProgramV2(
            #         ctx.env_dict["soccfg"],
            #         ctx.cfg,
            #         modules=[
            #             Reset("reset", ctx.cfg.get("reset", {"type": "none"})),
            #             Pulse("pi_pulse", ctx.cfg["pi_pulse"]),
            #             Delay(
            #                 name="t1_delay",
            #                 delay=sweep2param("length", self.length_sweep),
            #             ),
            #             Readout("readout", ctx.cfg["readout"]),
            #         ],
            #     )
            # ).acquire(
            #     ctx.env_dict["soc"],
            #     progress=False,
            #     callback=wrap_earlystop_check(
            #         prog,
            #         update_hook,
            #         self.earlystop_snr,
            #         signal2real_fn=t1_signal2real,
            #     ),
            # ),
            result_shape=(self.length_sweep["expts"],),
        )

    def num_axes(self) -> Dict[str, int]:
        return dict(t1=1, t1_curve=2)

    def make_plotter(self, name, axs) -> PlotterDictType:
        return PlotterDictType(
            t1=LivePlotter1D(
                "Flux device value",
                "T1 (us)",
                existed_axes=[axs["t1"]],
                segment_kwargs=dict(title=name + "(t1)"),
            ),
            t1_curve=LivePlotter2DwithLine(
                "Flux device value",
                "Signal",
                line_axis=1,
                num_lines=5,
                title=name + "(t1_curve)",
                existed_axes=[axs["t1_curve"]],
            ),
        )

    def update_plotter(self, plotters, ctx, signals) -> None:
        flx_values = ctx.env_dict["flx_values"]

        plotters["t1"].update(flx_values, signals["t1"], refresh=False)
        plotters["t1_curve"].update(
            flx_values,
            sweep2array(self.length_sweep),
            t1_fluxdep_signal2real(signals["raw_signals"]),
            refresh=False,
        )

    def save(self, filepath, flx_values, result, comment, prefix_tag) -> None:
        filepath = Path(filepath)

        lengths = sweep2array(self.length_sweep)
        np.savez_compressed(filepath, flx_values=flx_values, lengths=lengths, **result)

        x_info = {"name": "Flux value", "unit": "a.u.", "values": flx_values}

        # signals
        save_data(
            filepath=filepath.with_name(filepath.name + "_signals"),
            x_info=x_info,
            y_info={"name": "Length", "unit": "s", "values": lengths * 1e-6},
            z_info={
                "name": "Signal",
                "unit": "a.u.",
                "values": result["raw_signals"].T,
            },
            comment=comment,
            tag=prefix_tag + "/signals",
        )

        # t1
        save_data(
            filepath=filepath.with_name(filepath.name + "_t1"),
            x_info=x_info,
            z_info={"name": "T1", "unit": "s", "values": result["t1"] * 1e-6},
            comment=comment,
            tag=prefix_tag + "/t1",
        )

        # success
        save_data(
            filepath=filepath.with_name(filepath.name + "_success"),
            x_info=x_info,
            z_info={"name": "Success", "unit": "bool", "values": result["success"]},
            comment=comment,
            tag=prefix_tag + "/success",
        )

    def init(self, ctx, dynamic_pbar=False) -> None:
        self.task.init(ctx, dynamic_pbar=dynamic_pbar)

        ctx.env_dict["t1"] = np.nan

    def run(self, ctx) -> None:
        ml: ModuleLibrary = ctx.env_dict["ml"]

        cfg = self.cfg_maker(ctx, ml)
        deepupdate(
            cfg,
            {
                "dev": ctx.cfg["dev"],
                "sweep": {"length": self.length_sweep},
            },
        )
        cfg = ml.make_cfg(cfg)

        self.task.run(ctx(addr="raw_signals", new_cfg=cfg))

        raw_signals = ctx.get_current_data(append_addr=["raw_signals"])
        real_signals = t1_signal2real(raw_signals)

        t1, t1err, fit_signals, _ = fit_decay(
            sweep2array(self.length_sweep), real_signals
        )

        result = T1Result(
            raw_signals=raw_signals,
            t1=t1,
            t1_err=t1err,
            success=True,
        )

        if np.mean(np.abs(real_signals - fit_signals)) > 0.1 * np.ptp(real_signals):
            result["success"] = False

        if result["success"]:
            ctx.env_dict["t1"] = t1

        ctx.set_current_data(result)

    def get_default_result(self) -> T1Result:
        return T1Result(
            raw_signals=self.task.get_default_result(),
            t1=np.array(np.nan),
            t1_err=np.array(np.nan),
            success=np.array(False),
        )

    def cleanup(self) -> None:
        self.task.cleanup()
