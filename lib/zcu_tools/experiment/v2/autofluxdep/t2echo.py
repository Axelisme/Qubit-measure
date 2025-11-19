from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
from numpy.typing import NDArray
from typing_extensions import NotRequired, TypedDict

from zcu_tools.experiment.utils import sweep2array
from zcu_tools.experiment.v2.runner import HardTask, TaskConfig, TaskContext
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
from zcu_tools.utils.fitting import fit_decay, fit_decay_fringe
from zcu_tools.utils.process import rotate2real

from .executor import MeasurementTask


def t2echo_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return rotate2real(signals).real


def t2echo_fluxdep_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return np.array(list(map(t2echo_signal2real, signals)), dtype=np.float64)


class T2EchoCfg(TaskConfig):
    reset: NotRequired[ResetCfg]
    pi_pulse: PulseCfg
    pi2_pulse: PulseCfg
    readout: ReadoutCfg


class T2EchoResult(TypedDict, closed=True):
    raw_signals: NDArray[np.complex128]
    t2e: NDArray[np.float64]
    t2e_err: NDArray[np.float64]
    success: NDArray[np.bool_]


class PlotterDictType(TypedDict, closed=True):
    t2e: LivePlotter1D
    t2e_curve: LivePlotter2DwithLine


class T2EchoMeasurementTask(MeasurementTask[T2EchoResult, PlotterDictType]):
    def __init__(
        self,
        length_sweep: dict,
        activate_detune: float,
        cfg_maker: Callable[[TaskContext, ModuleLibrary], T2EchoCfg],
        earlystop_snr: Optional[float] = None,
    ) -> None:
        self.length_sweep = length_sweep
        self.activate_detune = activate_detune
        self.cfg_maker = cfg_maker
        self.earlystop_snr = earlystop_snr

        def measure_t2echo_fn(ctx: TaskContext, update_hook: Callable):
            import time

            from zcu_tools.utils.fitting.base import decaycos

            lengths = sweep2array(self.length_sweep)
            for i in range(ctx.cfg["rounds"]):
                raw_signals = [
                    np.array(
                        [
                            np.stack(
                                [
                                    decaycos(
                                        lengths,
                                        0,
                                        1,
                                        self.activate_detune
                                        + 0.1 * ctx.env_dict["flx_value"],
                                        0,
                                        4 * (ctx.env_dict["flx_value"] ** 2 + 1.0),
                                    )
                                    + 0.01
                                    * (ctx.cfg["rounds"] - i)
                                    * np.random.randn(len(lengths)),
                                    np.zeros_like(lengths),
                                ],
                                axis=1,
                            )
                        ],
                        dtype=np.complex128,
                    )
                ]
                update_hook(i, raw_signals)
                time.sleep(0.01)

            return raw_signals

        self.task = HardTask(
            measure_fn=measure_t2echo_fn,
            # measure_fn=lambda ctx, update_hook: (
            #     t2e_params := sweep2param("length", ctx.cfg["sweep"]["length"])
            # )
            # and (
            #     prog := ModularProgramV2(
            #         ctx.env_dict["soccfg"],
            #         ctx.cfg,
            #         modules=[
            #             Reset("reset", ctx.cfg.get("reset", {"type": "none"})),
            #             Pulse("pi2_pulse1", ctx.cfg["pi2_pulse"]),
            #             Delay("t2e_delay1", delay=0.5 * t2e_params),
            #             Pulse("pi_pulse", ctx.cfg["pi_pulse"]),
            #             Delay("t2e_delay2", delay=0.5 * t2e_params),
            #             Pulse(
            #                 name="pi2_pulse2",
            #                 cfg={
            #                     **ctx.cfg["pi2_pulse"],
            #                     "phase": ctx.cfg["pi2_pulse"]["phase"]
            #                     + 360 * self.activate_detune * t2e_params,
            #                 },
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
            #         signal2real_fn=t2echo_signal2real,
            #     ),
            # ),
            result_shape=(self.length_sweep["expts"],),
        )

    def num_axes(self) -> Dict[str, int]:
        return dict(t2e=1, t2e_curve=2)

    def make_plotter(self, name, axs) -> PlotterDictType:
        return PlotterDictType(
            t2e=LivePlotter1D(
                "Flux device value",
                "T2 Echo (us)",
                existed_axes=[axs["t2e"]],
                segment_kwargs=dict(title=name + "(t2e)"),
            ),
            t2e_curve=LivePlotter2DwithLine(
                "Flux device value",
                "Signal",
                line_axis=1,
                num_lines=5,
                title=name + "(t2e_curve)",
                existed_axes=[axs["t2e_curve"]],
            ),
        )

    def update_plotter(self, plotters, ctx, signals) -> None:
        flx_values = ctx.env_dict["flx_values"]

        plotters["t2e"].update(flx_values, signals["t2e"], refresh=False)
        plotters["t2e_curve"].update(
            flx_values,
            sweep2array(self.length_sweep),
            t2echo_fluxdep_signal2real(signals["raw_signals"]),
            refresh=False,
        )

    def save(self, filepath, flx_values, result, comment, prefix_tag) -> None:
        filepath = Path(filepath)

        lengths = sweep2array(self.length_sweep)
        np.savez_compressed(filepath, flx_values=flx_values, lengths=lengths, **result)

        x_info = {"name": "Flux value", "unit": "a.u.", "values": flx_values}

        # signals
        save_data(
            filepath=str(filepath.with_name(filepath.name + "_signals")),
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

        # t2e
        save_data(
            filepath=str(filepath.with_name(filepath.name + "_t2e")),
            x_info=x_info,
            z_info={"name": "T2 Echo", "unit": "s", "values": result["t2e"] * 1e-6},
            comment=comment,
            tag=prefix_tag + "/t2e",
        )

        # success
        save_data(
            filepath=str(filepath.with_name(filepath.name + "_success")),
            x_info=x_info,
            z_info={"name": "Success", "unit": "bool", "values": result["success"]},
            comment=comment,
            tag=prefix_tag + "/success",
        )

    def init(self, ctx, dynamic_pbar=False) -> None:
        self.task.init(ctx, dynamic_pbar=dynamic_pbar)

        ctx.env_dict["t2e"] = np.nan

    def run(self, ctx) -> None:
        ml: ModuleLibrary = ctx.env_dict["ml"]

        cfg = self.cfg_maker(ctx, ml)
        deepupdate(
            cfg,  # type: ignore
            {
                "dev": ctx.cfg["dev"],
                "sweep": {"length": self.length_sweep},
            },
        )
        cfg = ml.make_cfg(dict(cfg))

        self.task.run(ctx(addr="raw_signals", new_cfg=cfg))

        raw_signals = ctx.get_current_data(append_addr=["raw_signals"])
        assert isinstance(raw_signals, np.ndarray)

        real_signals = t2echo_signal2real(raw_signals)

        lengths = sweep2array(self.length_sweep)
        if self.activate_detune == 0.0:
            t2e, t2e_err, fit_signals, _ = fit_decay(lengths, real_signals)
        else:
            t2e, t2e_err, _, _, fit_signals, _ = fit_decay_fringe(lengths, real_signals)

        result = T2EchoResult(
            raw_signals=raw_signals,
            t2e=np.array(t2e),
            t2e_err=np.array(t2e_err),
            success=np.array(True),
        )

        if np.mean(np.abs(real_signals - fit_signals)) > 0.1 * np.ptp(real_signals):
            result["success"] = np.array(False)

        if result["success"]:
            ctx.env_dict["t2e"] = t2e

        ctx.set_current_data(result)

    def get_default_result(self) -> T2EchoResult:
        return T2EchoResult(
            raw_signals=self.task.get_default_result(),
            t2e=np.array(np.nan),
            t2e_err=np.array(np.nan),
            success=np.array(False),
        )

    def cleanup(self) -> None:
        self.task.cleanup()
