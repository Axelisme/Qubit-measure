from __future__ import annotations

from copy import deepcopy
from typing import Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d
from matplotlib.figure import Figure

from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.utils import format_sweep1D, sweep2array
from zcu_tools.liveplot import LivePlotter1D, LivePlotter2DwithLine
from zcu_tools.program.v2 import (
    ModularProgramCfg,
    ModularProgramV2,
    Pulse,
    PulseCfg,
    Readout,
    ReadoutCfg,
    Repeat,
    Reset,
    sweep2param,
)
from zcu_tools.utils.datasaver import save_data
from zcu_tools.utils.process import rotate2real

from ..runner import HardTask, SoftTask, TaskConfig, TaskContext, run_task

# (times, signals)
ZigZagResultType = Tuple[NDArray[np.float64], NDArray[np.complex128]]


def zigzag_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return rotate2real(signals).real  # type: ignore


class ZigZagTaskConfig(TaskConfig, ModularProgramCfg):
    X90_pulse: PulseCfg
    X180_pulse: PulseCfg
    readout: ReadoutCfg


class ZigZagExperiment(AbsExperiment):
    def run(
        self,
        soc,
        soccfg,
        cfg: ZigZagTaskConfig,
        *,
        repeat_on: Literal["X90_pulse", "X180_pulse"] = "X180_pulse",
    ) -> ZigZagResultType:
        cfg = deepcopy(cfg)  # avoid in-place modification

        X90_pulse = deepcopy(cfg["X90_pulse"])

        assert "sweep" in cfg
        cfg["sweep"] = format_sweep1D(cfg["sweep"], "times")
        times = sweep2array(cfg["sweep"]["times"], allow_array=True)  # predicted

        del cfg["sweep"]

        with LivePlotter1D(
            "Times", "Signal", segment_kwargs=dict(show_grid=True)
        ) as viewer:

            def measure_fn(ctx: TaskContext, update_hook):
                zigzag_time = ctx.env_dict["zigzag_time"]
                if repeat_on == "X90_pulse":
                    repeat_time = 2 * zigzag_time
                else:
                    repeat_time = zigzag_time

                return ModularProgramV2(
                    soccfg,
                    ctx.cfg,
                    modules=[
                        Reset(
                            "reset",
                            ctx.cfg.get("reset", {"type": "none"}),
                        ),
                        Pulse(name="X90_pulse", cfg=X90_pulse),
                        Repeat(
                            name="zigzag_loop",
                            n=repeat_time,
                            sub_module=Pulse(
                                name=f"loop_{repeat_on}",
                                cfg=ctx.cfg[repeat_on],
                            ),
                        ),
                        Readout("readout", ctx.cfg["readout"]),
                    ],
                ).acquire(soc, progress=False, callback=update_hook)

            signals = run_task(
                task=SoftTask(
                    sweep_name="times",
                    sweep_values=times.tolist(),
                    update_cfg_fn=lambda _, ctx, time: (
                        ctx.env_dict.update(zigzag_time=time)
                    ),
                    sub_task=HardTask(measure_fn=measure_fn),
                ),
                init_cfg=cfg,
                update_hook=lambda ctx: viewer.update(
                    times, zigzag_signal2real(np.asarray(ctx.data))
                ),
            )
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


# (times, values, signals)
ZigZagSweepResultType = Tuple[
    NDArray[np.int64], NDArray[np.float64], NDArray[np.complex128]
]


class ZigZagSweepExperiment(AbsExperiment):
    SWEEP_MAP = {
        "gain": {"name": "Gain (a.u.)", "param_key": "gain"},
        "freq": {"name": "Frequency (MHz)", "param_key": "freq"},
    }

    def run(
        self,
        soc,
        soccfg,
        cfg: ZigZagTaskConfig,
        *,
        repeat_on: Literal["X90_pulse", "X180_pulse"] = "X180_pulse",
    ) -> ZigZagSweepResultType:
        cfg = deepcopy(cfg)  # avoid in-place modification

        X90_pulse = deepcopy(cfg["X90_pulse"])

        assert "sweep" in cfg
        assert isinstance(cfg["sweep"], dict)
        time_sweep = cfg["sweep"].pop("times")
        times = sweep2array(time_sweep, allow_array=True).astype(np.int64)

        # extract sweep parameters
        x_key = list(cfg["sweep"].keys())[0]
        if x_key not in self.SWEEP_MAP:
            raise ValueError(f"Unsupported sweep key: {x_key}")
        x_info = self.SWEEP_MAP[x_key]
        values: NDArray[np.float64] = sweep2array(cfg["sweep"][x_key])  # predicted

        Pulse.set_param(
            cfg[repeat_on],
            x_info["param_key"],
            sweep2param(x_info["param_key"], cfg["sweep"][x_key]),
        )

        with LivePlotter2DwithLine(
            "Times", x_info["name"], line_axis=1, num_lines=3
        ) as viewer:

            def measure_fn(ctx: TaskContext, update_hook):
                zigzag_time = ctx.env_dict["zigzag_time"]
                if repeat_on == "X90_pulse":
                    repeat_time = 2 * zigzag_time
                else:
                    repeat_time = zigzag_time

                return ModularProgramV2(
                    soccfg,
                    ctx.cfg,
                    modules=[
                        Reset(
                            "reset",
                            ctx.cfg.get("reset", {"type": "none"}),
                        ),
                        Pulse(name="X90_pulse", cfg=X90_pulse),
                        Repeat(
                            name="zigzag_loop",
                            n=repeat_time,
                            sub_module=Pulse(
                                name=f"loop_{repeat_on}",
                                cfg=ctx.cfg[repeat_on],
                            ),
                        ),
                        Readout("readout", ctx.cfg["readout"]),
                    ],
                ).acquire(soc, progress=False, callback=update_hook)

            signals = run_task(
                task=SoftTask(
                    sweep_name="times",
                    sweep_values=times.tolist(),
                    update_cfg_fn=lambda _, ctx, time: ctx.env_dict.update(
                        zigzag_time=time
                    ),
                    sub_task=HardTask(
                        measure_fn=measure_fn,
                        result_shape=(len(values),),
                    ),
                ),
                init_cfg=cfg,
                update_hook=lambda ctx: viewer.update(
                    times.astype(np.float64),
                    values,
                    zigzag_signal2real(np.asarray(ctx.data)),
                ),
            )
            signals = np.asarray(signals)

        # record last cfg and result
        self.last_cfg = cfg
        self.last_result = (times, values, signals)

        return times, values, signals

    def analyze(
        self,
        result: Optional[ZigZagSweepResultType] = None,
    ) -> Tuple[float, Figure]:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

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

        return min_value, fig

    def save(
        self,
        filepath: str,
        result: Optional[ZigZagSweepResultType] = None,
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
