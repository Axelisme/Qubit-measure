from __future__ import annotations

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d
from typeguard import check_type
from typing_extensions import (
    Any,
    Literal,
    NotRequired,
    Optional,
    Sequence,
    TypeAlias,
    TypedDict,
    Union,
)

from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.v2.runner import Task, TaskCfg, TaskState, run_task
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.liveplot import LivePlotter2DwithLine
from zcu_tools.program import SweepCfg
from zcu_tools.program.v2 import (
    ModularProgramCfg,
    ModularProgramV2,
    Pulse,
    PulseCfg,
    Readout,
    ReadoutCfg,
    Repeat,
    Reset,
    ResetCfg,
    sweep2param,
)
from zcu_tools.utils.datasaver import load_data, save_data
from zcu_tools.utils.process import rotate2real

# (times, values, signals)
ZigZagSweepResult: TypeAlias = tuple[
    NDArray[np.int64], NDArray[np.float64], NDArray[np.complex128]
]


def zigzag_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return rotate2real(signals).real  # type: ignore


class ZigZagModuleCfg(TypedDict, closed=True):
    reset: NotRequired[ResetCfg]
    X90_pulse: PulseCfg
    X180_pulse: NotRequired[PulseCfg]
    readout: ReadoutCfg


class ZigZagSweepCfg(TypedDict, closed=True):
    times: Union[SweepCfg, Sequence]
    gain: SweepCfg


class ZigZagCfg(ModularProgramCfg, TaskCfg):
    modules: ZigZagModuleCfg
    sweep: ZigZagSweepCfg


class ZigZagSweepExp(AbsExperiment[ZigZagSweepResult, ZigZagCfg]):
    SWEEP_MAP = {
        "gain": {"name": "Gain (a.u.)", "param_key": "gain"},
        "freq": {"name": "Frequency (MHz)", "param_key": "freq"},
    }

    def run(
        self,
        soc,
        soccfg,
        cfg: dict[str, Any],
        *,
        repeat_on: Literal["X90_pulse", "X180_pulse"] = "X180_pulse",
    ) -> ZigZagSweepResult:
        _cfg = check_type(deepcopy(cfg), ZigZagCfg)
        modules = _cfg["modules"]

        X90_pulse = deepcopy(modules["X90_pulse"])

        time_sweep: SweepCfg = _cfg["sweep"].pop("times")  # type: ignore
        times = sweep2array(time_sweep, allow_array=True).astype(np.int64)

        # extract sweep parameters
        x_key = list(_cfg["sweep"].keys())[0]
        if x_key not in self.SWEEP_MAP:
            raise ValueError(f"Unsupported sweep key: {x_key}")
        x_info = self.SWEEP_MAP[x_key]
        values: NDArray[np.float64] = sweep2array(_cfg["sweep"][x_key])  # predicted

        if repeat_on not in modules:
            raise ValueError(f"Repeat on pulse {repeat_on} not found")

        Pulse.set_param(
            modules[repeat_on],  # type: ignore
            x_info["param_key"],
            sweep2param(x_info["param_key"], _cfg["sweep"][x_key]),  # type: ignore
        )

        with LivePlotter2DwithLine(
            "Times", x_info["name"], line_axis=1, num_lines=3
        ) as viewer:

            def measure_fn(ctx: TaskState, update_hook):
                modules = ctx.cfg["modules"]
                zigzag_time = ctx.env["zigzag_time"]
                if repeat_on == "X90_pulse":
                    repeat_time = 2 * zigzag_time
                else:
                    repeat_time = zigzag_time

                return ModularProgramV2(
                    soccfg,
                    ctx.cfg,
                    modules=[
                        Reset("reset", modules.get("reset")),
                        Pulse(name="X90_pulse", cfg=X90_pulse),
                        Repeat(
                            name="zigzag_loop",
                            n=repeat_time,
                            sub_module=Pulse(
                                name=f"loop_{repeat_on}",
                                cfg=modules[repeat_on],
                            ),
                        ),
                        Readout("readout", modules["readout"]),
                    ],
                ).acquire(soc, progress=False, callback=update_hook)

            signals = run_task(
                task=Task(
                    measure_fn=measure_fn,
                    result_shape=(len(values),),
                ).scan(
                    "times",
                    times.tolist(),
                    before_each=lambda _, ctx, time: ctx.env.update(zigzag_time=time),
                ),
                init_cfg=_cfg,
                on_update=lambda ctx: viewer.update(
                    times.astype(np.float64),
                    values,
                    zigzag_signal2real(np.asarray(ctx.root_data)),
                ),
            )
            signals = np.asarray(signals)

        # record last cfg and result
        self.last_cfg = _cfg
        self.last_result = (times, values, signals)

        return times, values, signals

    def analyze(
        self,
        result: Optional[ZigZagSweepResult] = None,
    ) -> tuple[float, Figure]:
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
        ax2.grid(True)
        ax2.legend()
        ax2.set_xlabel("Sweep value (a.u.)")
        ax2.set_ylabel("Cumulative difference (a.u.)")

        return min_value, fig

    def save(
        self,
        filepath: str,
        result: Optional[ZigZagSweepResult] = None,
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

    def load(self, filepath: str, **kwargs) -> ZigZagSweepResult:
        signals, times, values = load_data(filepath, **kwargs)
        assert times is not None and values is not None
        assert len(times.shape) == 1 and len(values.shape) == 1
        assert signals.shape == (len(values), len(times))

        signals = signals.T  # transpose back

        times = times.astype(np.int64)
        values = values.astype(np.float64)
        signals = signals.astype(np.complex128)

        self.last_cfg = None
        self.last_result = (times, values, signals)

        return times, values, signals
