from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Dict, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.utils import format_sweep1D, sweep2array
from zcu_tools.liveplot import LivePlotter1D, LivePlotter2DwithLine
from zcu_tools.program.v2 import (
    ModularProgramV2,
    Pulse,
    Readout,
    Repeat,
    Reset,
    sweep2param,
)
from zcu_tools.utils.datasaver import save_data
from zcu_tools.utils.process import rotate2real

from ..runner import HardTask, Runner, SoftTask, TaskContext


def zigzag_signal2real(signals: np.ndarray) -> np.ndarray:
    return rotate2real(signals).real  # type: ignore


ZigZagResultType = Tuple[np.ndarray, np.ndarray]  # (times, signals)


class ZigZagExperiment(AbsExperiment[ZigZagResultType]):
    """ZigZag oscillation by varying pi-pulse times following a pi/2-pulse."""

    def run(
        self,
        soc,
        soccfg,
        cfg: Dict[str, Any],
        *,
        repeat_on: Literal["X90_pulse", "X180_pulse"] = "X180_pulse",
        progress: bool = True,
    ) -> ZigZagResultType:
        cfg = deepcopy(cfg)  # avoid in-place modification

        X90_pulse = deepcopy(cfg["X90_pulse"])

        cfg["sweep"] = format_sweep1D(cfg["sweep"], "times")
        times = sweep2array(cfg["sweep"]["times"], allow_array=True)  # predicted

        del cfg["sweep"]

        def updateCfg(_: int, ctx: TaskContext, time: float) -> None:
            ctx.cfg["zigzag_time"] = time

        def measure_fn(
            ctx: TaskContext, update_hook: Callable[[int, Any], None]
        ) -> np.ndarray:
            zigzag_time = ctx.cfg["zigzag_time"]

            if repeat_on == "X90_pulse":
                repeat_time = 2 * zigzag_time
            elif repeat_on == "X180_pulse":
                repeat_time = zigzag_time

            return ModularProgramV2(
                soccfg,
                ctx.cfg,
                modules=[
                    Reset("reset", ctx.cfg.get("reset", {"type": "none"})),
                    Pulse(name="X90_pulse", cfg=X90_pulse),
                    Repeat(
                        name="zigzag_loop",
                        n=repeat_time,
                        sub_module=Pulse(
                            name=f"loop_{repeat_on}", cfg=ctx.cfg[repeat_on]
                        ),
                    ),
                    Readout("readout", ctx.cfg["readout"]),
                ],
            ).acquire(soc, progress=False, callback=update_hook)

        with LivePlotter1D(
            "Times", "Signal", segment_kwargs=dict(show_grid=True), disable=not progress
        ) as viewer:
            signals = Runner(
                task=SoftTask(
                    sweep_name="times",
                    sweep_values=times,
                    update_cfg_fn=updateCfg,
                    sub_task=HardTask(measure_fn=measure_fn),
                ),
                update_hook=lambda ctx: viewer.update(
                    times, zigzag_signal2real(np.asarray(ctx.get_data()))
                ),
            ).run(cfg)
            signals = np.asarray(signals)

        # record last cfg and result
        self.last_cfg = cfg
        self.last_result = (times, signals)

        return times, signals

    def analyze(
        self,
        result: Optional[ZigZagResultType] = None,
    ) -> Tuple[float, float]:
        raise NotImplementedError("Not implemented")

    def save(
        self,
        filepath: str,
        result: Optional[ZigZagResultType] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/ge/zigzag",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        times, signals = result
        save_data(
            filepath=filepath,
            x_info={"name": "Times", "unit": "a.u.", "values": times},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals},
            comment=comment,
            tag=tag,
            **kwargs,
        )


ZigZagSweepResultType = Tuple[np.ndarray, np.ndarray, np.ndarray]  # (xs,times, signals)


class ZigZagSweepExperiment(AbsExperiment[ZigZagSweepResultType]):
    SWEEP_MAP = {
        "gain": {"name": "Gain (a.u.)", "param_key": "gain"},
        "freq": {"name": "Frequency (MHz)", "param_key": "freq"},
    }

    def run(
        self,
        soc,
        soccfg,
        cfg: Dict[str, Any],
        *,
        repeat_on: Literal["X90_pulse", "X180_pulse"] = "X180_pulse",
        progress: bool = True,
    ) -> ZigZagSweepResultType:
        cfg = deepcopy(cfg)  # avoid in-place modification

        X90_pulse = deepcopy(cfg["X90_pulse"])

        time_sweep = cfg["sweep"].pop("times")
        times = sweep2array(time_sweep, allow_array=True)  # predicted

        # extract sweep parameters
        x_key = list(cfg["sweep"].keys())[0]
        if x_key not in self.SWEEP_MAP:
            raise ValueError(f"Unsupported sweep key: {x_key}")
        x_info = self.SWEEP_MAP[x_key]
        values = sweep2array(cfg["sweep"][x_key])  # predicted

        cfg[repeat_on][x_info["param_key"]] = sweep2param(
            x_info["param_key"], cfg["sweep"][x_key]
        )

        def updateCfg(_: int, ctx: TaskContext, time: float) -> None:
            ctx.cfg["zigzag_time"] = time

        def measure_fn(
            ctx: TaskContext, update_hook: Callable[[int, Any], None]
        ) -> np.ndarray:
            zigzag_time = ctx.cfg["zigzag_time"]

            if repeat_on == "X90_pulse":
                repeat_time = 2 * zigzag_time
            elif repeat_on == "X180_pulse":
                repeat_time = zigzag_time

            return ModularProgramV2(
                soccfg,
                ctx.cfg,
                modules=[
                    Reset("reset", ctx.cfg.get("reset", {"type": "none"})),
                    Pulse(name="X90_pulse", cfg=X90_pulse),
                    Repeat(
                        name="zigzag_loop",
                        n=repeat_time,
                        sub_module=Pulse(
                            name=f"loop_{repeat_on}", cfg=ctx.cfg[repeat_on]
                        ),
                    ),
                    Readout("readout", ctx.cfg["readout"]),
                ],
            ).acquire(soc, progress=False, callback=update_hook)

        signals = Runner(
            task=SoftTask(
                sweep_name="times",
                sweep_values=times,
                update_cfg_fn=updateCfg,
                sub_task=HardTask(
                    measure_fn=measure_fn,
                    result_shape=(len(values),),
                ),
            ),
            liveplotter=LivePlotter2DwithLine(
                "Times",
                x_info["name"],
                line_axis=1,
                num_lines=3,
                disable=not progress,
            ),
            update_hook=lambda viewer, ctx: viewer.update(
                times, values, zigzag_signal2real(np.asarray(ctx.get_data()))
            ),
        ).run(cfg)
        signals = np.asarray(signals)

        # record last cfg and result
        self.last_cfg = cfg
        self.last_result = (times, values, signals)

        return times, values, signals

    def analyze(
        self,
        result: Optional[ZigZagResultType] = None,
    ) -> Tuple[float, float]:
        if result is None:
            result = self.last_result

        times, values, signals = result

        real_signals = zigzag_signal2real(signals)
        valid_cutoff = np.min(np.sum(~np.isnan(real_signals), axis=0))

        if valid_cutoff < 2:
            raise ValueError("Not enough valid data points for analysis")

        times = times[:valid_cutoff]
        real_signals = real_signals[:valid_cutoff]

        cum_diff = np.sum(np.abs(np.diff(real_signals, axis=0)), axis=0)
        cum_diff = gaussian_filter1d(cum_diff, sigma=1)
        min_value = values[np.argmin(cum_diff)]

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        ax1.imshow(
            zigzag_signal2real(signals),
            aspect="auto",
            extent=[values[0], values[-1], times[0], times[-1]],
            origin="lower",
            interpolation="none",
        )
        ax1.set_ylabel("Number of gate")
        ax2.plot(values, cum_diff, marker=".")
        ax2.axvline(
            x=min_value,
            color="red",
            linestyle="--",
            label=f"x = {min_value:.3f}",
        )
        ax2.legend()
        ax2.set_xlabel("Sweep value (a.u.)")
        ax2.set_ylabel("Cumulative difference (a.u.)")

        return min_value

    def save(
        self,
        filepath: str,
        result: Optional[ZigZagResultType] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/ge/zigzag_sweep",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        times, values, signals = result
        save_data(
            filepath=filepath,
            x_info={"name": "Times", "unit": "a.u.", "values": times},
            y_info={"name": "Sweep value", "unit": "a.u.", "values": values},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals.T},
            comment=comment,
            tag=tag,
            **kwargs,
        )
