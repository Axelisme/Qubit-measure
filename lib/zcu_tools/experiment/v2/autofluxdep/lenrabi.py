from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, cast

import numpy as np
from numpy.typing import NDArray
from typing_extensions import NotRequired, TypedDict

from zcu_tools.experiment.utils import sweep2array
from zcu_tools.experiment.v2.runner import HardTask, TaskConfig, TaskContext
from zcu_tools.experiment.v2.utils import wrap_earlystop_check
from zcu_tools.library import ModuleLibrary
from zcu_tools.liveplot import LivePlotter2DwithLine
from zcu_tools.program import SweepCfg
from zcu_tools.program.v2 import (
    ModularProgramCfg,
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
from zcu_tools.utils.fitting import fit_rabi
from zcu_tools.utils.process import rotate2real

from .executor import MeasurementTask


def lenrabi_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return rotate2real(signals).real


def lenrabi_fluxdep_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return np.array(list(map(lenrabi_signal2real, signals)), dtype=np.float64)


class LenRabiCfgTemplate(TypedDict):
    reset: NotRequired[ResetCfg]
    rabi_pulse: PulseCfg
    readout: ReadoutCfg


class LenRabiCfg(TaskConfig, ModularProgramCfg):
    reset: NotRequired[ResetCfg]
    rabi_pulse: PulseCfg
    readout: ReadoutCfg


class LenRabiResult(TypedDict, closed=True):
    raw_signals: NDArray[np.complex128]
    pi_length: NDArray[np.float64]
    pi2_length: NDArray[np.float64]
    rabi_freq: NDArray[np.float64]
    success: NDArray[np.bool_]


class PlotterDictType(TypedDict, closed=True):
    rabi_curve: LivePlotter2DwithLine


class LenRabiMeasurementTask(
    MeasurementTask[LenRabiResult, LenRabiCfg, PlotterDictType]
):
    def __init__(
        self,
        length_sweep: SweepCfg,
        ref_pi_product: float,
        cfg_maker: Callable[[TaskContext, ModuleLibrary], LenRabiCfgTemplate],
        earlystop_snr: Optional[float] = None,
    ) -> None:
        self.length_sweep = length_sweep
        self.ref_pi_product = ref_pi_product
        self.cfg_maker = cfg_maker
        self.earlystop_snr = earlystop_snr

        def measure_lenrabi_fn(ctx: TaskContext, update_hook: Callable):
            import time

            from zcu_tools.utils.fitting.base import cosfunc

            lengths = sweep2array(self.length_sweep)
            for i in range(ctx.cfg["rounds"]):
                raw_signals = [
                    np.array(
                        [
                            np.stack(
                                [
                                    cosfunc(
                                        lengths,
                                        0,
                                        1,
                                        0.5
                                        * (ctx.env_dict["flx_value"] ** 2 + 3.0)
                                        * ctx.cfg["rabi_pulse"]["gain"],
                                        0,
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

            return cast(Sequence[NDArray[np.float64]], raw_signals)

        self.task = HardTask[Sequence[NDArray[np.float64]], LenRabiCfg](
            measure_fn=measure_lenrabi_fn,
            # measure_fn=lambda ctx, update_hook: (
            #     prog := ModularProgramV2(
            #         ctx.env_dict["soccfg"],
            #         ctx.cfg,
            #         modules=[
            #             Reset("reset", ctx.cfg.get("reset", {"type": "none"})),
            #             Pulse(
            #                 "rabi_pulse",
            #                 Pulse.set_param(
            #                     ctx.cfg["rabi_pulse"],
            #                     "length",
            #                     sweep2param("length", self.length_sweep),
            #                 ),
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
            #         signal2real_fn=lenrabi_signal2real,
            #     ),
            # ),
            result_shape=(self.length_sweep["expts"],),
        )

    def num_axes(self) -> Dict[str, int]:
        return dict(rabi_curve=2)

    def make_plotter(self, name, axs) -> PlotterDictType:
        self.pi_line = axs["rabi_curve"][1].axvline(np.nan, color="red", linestyle="--")
        return PlotterDictType(
            rabi_curve=LivePlotter2DwithLine(
                "Flux device value",
                "Signal",
                line_axis=1,
                num_lines=5,
                title=name + "(rabi_curve)",
                existed_axes=[axs["rabi_curve"]],
            ),
        )

    def update_plotter(self, plotters, ctx, signals) -> None:
        flx_values = ctx.env_dict["flx_values"]

        self.pi_line.set_xdata([ctx.env_dict["pi_length"]])
        plotters["rabi_curve"].update(
            flx_values,
            sweep2array(self.length_sweep),
            lenrabi_fluxdep_signal2real(signals["raw_signals"]),
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

        # pi length
        save_data(
            filepath=str(filepath.with_name(filepath.name + "_pi_length")),
            x_info=x_info,
            z_info={
                "name": "Pi length",
                "unit": "s",
                "values": result["pi_length"] * 1e-6,
            },
            comment=comment,
            tag=prefix_tag + "/pi_length",
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

        ctx.env_dict["pi_length"] = np.nan
        ctx.env_dict["pi2_length"] = np.nan
        ctx.env_dict["gain_factor"] = 1.0

    def run(self, ctx) -> None:
        ml: ModuleLibrary = ctx.env_dict["ml"]

        cfg_temp = self.cfg_maker(ctx, ml)
        deepupdate(
            cfg_temp,  # type: ignore
            {
                "dev": ctx.cfg["dev"],
                "sweep": {"length": self.length_sweep},
            },
        )
        cfg_temp = ml.make_cfg(dict(cfg_temp))

        cfg = cast(LenRabiCfg, cfg_temp)
        self.task.run(ctx(addr="raw_signals", new_cfg=cfg))

        cur_gain = cfg["rabi_pulse"]["gain"]

        raw_signals = ctx.get_current_data(append_addr=["raw_signals"])
        assert isinstance(raw_signals, np.ndarray)

        real_signals = lenrabi_signal2real(raw_signals)

        pi_len, pi2_len, rabi_freq, fit_signals, _ = fit_rabi(
            sweep2array(self.length_sweep), real_signals, decay=True
        )

        result = LenRabiResult(
            raw_signals=raw_signals,
            pi_length=np.array(pi_len),
            pi2_length=np.array(pi2_len),
            rabi_freq=np.array(rabi_freq),
            success=np.array(True),
        )

        if np.mean(np.abs(real_signals - fit_signals)) > 0.1 * np.ptp(real_signals):
            result["success"] = np.array(False)

        if result["success"]:
            new_gain_factor = cur_gain * pi_len / self.ref_pi_product

            ctx.env_dict["pi_length"] = pi_len
            ctx.env_dict["pi2_length"] = pi2_len
            ctx.env_dict["gain_factor"] = (
                ctx.env_dict["gain_factor"] * new_gain_factor
            ) ** 0.5

        ctx.set_current_data(result)

    def get_default_result(self) -> LenRabiResult:
        return LenRabiResult(
            raw_signals=self.task.get_default_result(),
            pi_length=np.array(np.nan),
            pi2_length=np.array(np.nan),
            rabi_freq=np.array(np.nan),
            success=np.array(False),
        )

    def cleanup(self) -> None:
        self.task.cleanup()
