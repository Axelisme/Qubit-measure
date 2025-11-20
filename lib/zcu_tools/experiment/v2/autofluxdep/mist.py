from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Callable, Dict, Sequence, cast

import numpy as np
from numpy.typing import NDArray
from typing_extensions import NotRequired, TypedDict

from zcu_tools.experiment.utils import make_ge_sweep, sweep2array
from zcu_tools.experiment.v2.runner import HardTask, TaskConfig, TaskContext
from zcu_tools.experiment.v2.utils import wrap_earlystop_check
from zcu_tools.library import ModuleLibrary
from zcu_tools.liveplot import LivePlotter1D, LivePlotter2D
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

from .executor import MeasurementTask


def mist_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    # shape: (gains, ge, ge)
    avg_len = max(int(0.05 * signals.shape[0]), 1)

    real_signals = signals - np.mean(signals[:avg_len], axis=0, keepdims=True)
    mist_signals = np.abs(0.5 * (real_signals[..., 0] + real_signals[..., 1]))
    decay_signals = np.abs(0.5 * (real_signals[..., 0] - real_signals[..., 1]))

    # shape: (gains, ge, md)
    return np.stack([mist_signals, decay_signals], axis=-1)


def mist_fluxdep_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return np.array(list(map(mist_signal2real, signals)), dtype=np.float64)


class MistCfgTemplate(TypedDict):
    reset: NotRequired[ResetCfg]
    pi_pulse: PulseCfg
    mist_pulse: PulseCfg
    readout: ReadoutCfg


class MistCfg(TaskConfig, ModularProgramCfg):
    reset: NotRequired[ResetCfg]
    pre_pi_pulse: PulseCfg
    mist_pulse: PulseCfg
    post_pi_pulse: PulseCfg
    readout: ReadoutCfg


class MistResult(TypedDict, closed=True):
    raw_signals: NDArray[np.complex128]
    success: NDArray[np.bool_]


class PlotterDictType(TypedDict, closed=True):
    g_mist: LivePlotter2D
    e_mist: LivePlotter2D
    g_decay: LivePlotter2D
    e_decay: LivePlotter2D
    mist_signal: LivePlotter1D
    decay_signal: LivePlotter1D


class MistMeasurementTask(MeasurementTask[MistResult, MistCfg, PlotterDictType]):
    def __init__(
        self,
        gain_sweep: SweepCfg,
        cfg_maker: Callable[[TaskContext, ModuleLibrary], MistCfgTemplate],
    ) -> None:
        self.gain_sweep = gain_sweep
        self.cfg_maker = cfg_maker

        def measure_mist_fn(ctx: TaskContext, update_hook: Callable):
            import time

            def make_ge_signals(i, g_threshold: float, e_threshold: float):
                noise_level = 0.01 * (ctx.cfg["rounds"] - i)
                mist_signals = np.where(
                    gains > g_threshold, gains, 0
                ) + noise_level * np.random.randn(len(gains))
                decay_signals = np.where(
                    gains > e_threshold, gains, 0
                ) + noise_level * np.random.randn(len(gains))
                g_signals = mist_signals + decay_signals
                e_signals = mist_signals - decay_signals
                return np.stack([g_signals, e_signals], axis=-1)  # (gains, post_ge)

            gains = sweep2array(self.gain_sweep)
            for i in range(ctx.cfg["rounds"]):
                raw_signals = np.stack(
                    [
                        make_ge_signals(
                            i, ctx.env_dict["flx_value"], ctx.env_dict["flx_value"] ** 2
                        ),
                        make_ge_signals(
                            i,
                            2 * ctx.env_dict["flx_value"] - 1,
                            2 * ctx.env_dict["flx_value"] + 1,
                        ),
                    ],
                    axis=1,
                )
                raw_signals = [
                    np.array(
                        [np.stack([raw_signals, np.zeros_like(raw_signals)], axis=-1)],
                        dtype=np.complex128,
                    )
                ]

                update_hook(i, raw_signals)
                time.sleep(0.01)

            # [[(gains, ge, ge, iq)]]
            return cast(Sequence[NDArray[np.float64]], raw_signals)

        self.task = HardTask[Sequence[NDArray[np.float64]], MistCfg](
            measure_fn=measure_mist_fn,
            # measure_fn=lambda ctx, update_hook: ModularProgramV2(
            #     ctx.env_dict["soccfg"],
            #     ctx.cfg,
            #     modules=[
            #         Reset("reset", ctx.cfg.get("reset", {"type": "none"})),
            #         Pulse(name="pre_pi_pulse", cfg=ctx.cfg["pre_pi_pulse"]),
            #         Pulse(name="mist_pulse", cfg=ctx.cfg["mist_pulse"]),
            #         Pulse(name="post_pi_pulse", cfg=ctx.cfg["post_pi_pulse"]),
            #         Readout("readout", ctx.cfg["readout"]),
            #     ],
            # ).acquire(ctx.env_dict["soc"], progress=False, callback=update_hook),
            result_shape=(self.gain_sweep["expts"], 2, 2),
        )

    def num_axes(self) -> Dict[str, int]:
        return dict(
            g_mist=1,
            e_mist=1,
            g_decay=1,
            e_decay=1,
            mist_signal=1,
            decay_signal=1,
        )

    def make_plotter(self, name, axs) -> PlotterDictType:
        return PlotterDictType(
            g_mist=LivePlotter2D(
                "Flux device value",
                "Readout Gain (a.u.)",
                segment_kwargs=dict(title=name + "(g_mist)"),
                existed_axes=[axs["g_mist"]],
            ),
            e_mist=LivePlotter2D(
                "Flux device value",
                "Readout Gain (a.u.)",
                segment_kwargs=dict(title=name + "(e_mist)"),
                existed_axes=[axs["e_mist"]],
            ),
            g_decay=LivePlotter2D(
                "Flux device value",
                "Readout Gain (a.u.)",
                segment_kwargs=dict(title=name + "(g_decay)"),
                existed_axes=[axs["g_decay"]],
            ),
            e_decay=LivePlotter2D(
                "Flux device value",
                "Readout Gain (a.u.)",
                segment_kwargs=dict(title=name + "(e_decay)"),
                existed_axes=[axs["e_decay"]],
            ),
            mist_signal=LivePlotter1D(
                "Readout Gain (a.u.)",
                "Mist Signal",
                segment_kwargs=dict(
                    title=name + "(mist_signal)",
                    num_lines=2,
                    line_kwargs=[
                        dict(label="Ground", color="blue"),
                        dict(label="Excited", color="red"),
                    ],
                ),
                existed_axes=[axs["mist_signal"]],
            ),
            decay_signal=LivePlotter1D(
                "Readout Gain (a.u.)",
                "Decay Signal",
                segment_kwargs=dict(
                    title=name + "(decay_signal)",
                    num_lines=2,
                    line_kwargs=[
                        dict(label="Ground", color="blue"),
                        dict(label="Excited", color="red"),
                    ],
                ),
                existed_axes=[axs["decay_signal"]],
            ),
        )

    def update_plotter(self, plotters, ctx, signals) -> None:
        flx_idx: int = ctx.env_dict["flx_idx"]
        flx_values: np.ndarray = ctx.env_dict["flx_values"]
        gains: np.ndarray = sweep2array(self.gain_sweep)

        # shape: (flx, gains, ge, md)
        real_signals = mist_fluxdep_signal2real(signals["raw_signals"])

        std_len = max(int(0.3 * real_signals.shape[1]), 5)
        real_signals = np.clip(
            real_signals, 0, 5 * np.nanstd(real_signals[:, :std_len])
        )

        plotters["g_mist"].update(
            flx_values, gains, real_signals[..., 0, 0], refresh=False
        )
        plotters["e_mist"].update(
            flx_values, gains, real_signals[..., 1, 0], refresh=False
        )
        plotters["g_decay"].update(
            flx_values, gains, real_signals[..., 0, 1], refresh=False
        )
        plotters["e_decay"].update(
            flx_values, gains, real_signals[..., 1, 1], refresh=False
        )
        plotters["mist_signal"].update(
            gains, real_signals[flx_idx, ..., 0].T, refresh=False
        )
        plotters["decay_signal"].update(
            gains, real_signals[flx_idx, ..., 1].T, refresh=False
        )

    def save(self, filepath, flx_values, result, comment, prefix_tag) -> None:
        filepath = Path(filepath)

        gains = sweep2array(self.gain_sweep)
        np.savez_compressed(filepath, flx_values=flx_values, gains=gains, **result)

        x_info = {"name": "Flux value", "unit": "a.u.", "values": flx_values}

        # signals
        g_signals = result["raw_signals"][:, :, 0, :]
        e_signals = result["raw_signals"][:, :, 1, :]
        save_data(
            filepath=str(filepath.with_name(filepath.name + "_signals_gg")),
            x_info=x_info,
            y_info={"name": "Readout Gain", "unit": "a.u.", "values": gains},
            z_info={"name": "Signal", "unit": "a.u.", "values": g_signals[..., 0].T},
            comment=comment,
            tag=prefix_tag + "/signals_gg",
        )
        save_data(
            filepath=str(filepath.with_name(filepath.name + "_signals_ge")),
            x_info=x_info,
            y_info={"name": "Readout Gain", "unit": "a.u.", "values": gains},
            z_info={"name": "Signal", "unit": "a.u.", "values": g_signals[..., 1].T},
            comment=comment,
            tag=prefix_tag + "/signals_ge",
        )
        save_data(
            filepath=str(filepath.with_name(filepath.name + "_signals_eg")),
            x_info=x_info,
            y_info={"name": "Readout Gain", "unit": "a.u.", "values": gains},
            z_info={"name": "Signal", "unit": "a.u.", "values": e_signals[..., 0].T},
            comment=comment,
            tag=prefix_tag + "/signals_eg",
        )
        save_data(
            filepath=str(filepath.with_name(filepath.name + "_signals_ee")),
            x_info=x_info,
            y_info={"name": "Readout Gain", "unit": "a.u.", "values": gains},
            z_info={"name": "Signal", "unit": "a.u.", "values": e_signals[..., 1].T},
            comment=comment,
            tag=prefix_tag + "/signals_ee",
        )

    def init(self, ctx, dynamic_pbar=False) -> None:
        self.task.init(ctx, dynamic_pbar=dynamic_pbar)

    def run(self, ctx) -> None:
        ml: ModuleLibrary = ctx.env_dict["ml"]

        cfg_temp = self.cfg_maker(ctx, ml)
        deepupdate(
            cfg_temp,  # type: ignore
            {
                "pre_pi_pulse": deepcopy(cfg_temp["pi_pulse"]),
                "post_pi_pulse": deepcopy(cfg_temp["pi_pulse"]),
                "dev": ctx.cfg["dev"],
                "sweep": {
                    "gain": self.gain_sweep,
                    "pre_ge": make_ge_sweep(),
                    "post_ge": make_ge_sweep(),
                },
            },
        )
        cfg_temp = ml.make_cfg(dict(cfg_temp))
        del cfg_temp["pi_pulse"]

        Pulse.set_param(
            cfg_temp["mist_pulse"],
            "gain",
            sweep2param("gain", cfg_temp["sweep"]["gain"]),
        )
        Pulse.set_param(
            cfg_temp["pre_pi_pulse"],
            "on/off",
            sweep2param("pre_ge", cfg_temp["sweep"]["pre_ge"]),
        )
        Pulse.set_param(
            cfg_temp["post_pi_pulse"],
            "on/off",
            sweep2param("post_ge", cfg_temp["sweep"]["post_ge"]),
        )

        cfg = cast(MistCfg, cfg_temp)
        self.task.run(ctx(addr="raw_signals", new_cfg=cfg))

        raw_signals = ctx.get_current_data(append_addr=["raw_signals"])
        assert isinstance(raw_signals, np.ndarray)

        result = MistResult(raw_signals=raw_signals, success=np.array(True))

        ctx.set_current_data(result)

    def get_default_result(self) -> MistResult:
        return MistResult(
            raw_signals=self.task.get_default_result(),
            success=np.array(True),
        )

    def cleanup(self) -> None:
        self.task.cleanup()
