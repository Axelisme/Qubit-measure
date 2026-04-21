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
    Callable,
    Literal,
    NotRequired,
    Optional,
    TypeAlias,
    TypedDict,
    cast,
)

from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.utils import setup_devices
from zcu_tools.experiment.v2.runner import Task, TaskCfg, TaskState, run_task
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.liveplot import LivePlot2D
from zcu_tools.program import SweepCfg
from zcu_tools.program.v2 import (
    LoadValue,
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


class ZigZagSweepCfg(TypedDict):
    gain: NotRequired[SweepCfg]
    freq: NotRequired[SweepCfg]


class ZigZagCfg(ModularProgramCfg, TaskCfg):
    modules: ZigZagModuleCfg
    sweep: ZigZagSweepCfg
    n_times: int


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
        acquire_kwargs: Optional[dict[str, Any]] = None,
    ) -> ZigZagSweepResult:
        _cfg = check_type(deepcopy(cfg), ZigZagCfg)
        setup_devices(_cfg, progress=True)
        modules = _cfg["modules"]

        times = np.arange(_cfg["n_times"])

        if len(_cfg["sweep"]) != 1:
            raise ValueError("Expected exactly one sweep key")

        x_key = next(k for k in _cfg["sweep"])
        x_sweep = _cfg["sweep"][x_key]
        if x_key not in ZigZagSweepExp.SWEEP_MAP:
            raise ValueError(f"Unsupported sweep key: {x_key}")
        x_info = ZigZagSweepExp.SWEEP_MAP[x_key]

        repeat_pulse = modules.get(repeat_on)
        if repeat_pulse is None:
            raise ValueError(f"Repeat on pulse {repeat_on} not found")

        values = sweep2array(
            x_sweep,
            x_key,  # type: ignore
            {"soccfg": soccfg, "gen_ch": repeat_pulse.ch},
        )

        def measure_fn(
            ctx: TaskState, update_hook: Optional[Callable]
        ) -> list[NDArray[np.float64]]:
            cfg = cast(ZigZagCfg, ctx.cfg)
            modules = cfg["modules"]

            X90_pulse = deepcopy(modules["X90_pulse"])
            repeat_pulse = modules.get(repeat_on)
            if repeat_pulse is None:
                raise ValueError(f"Repeat on pulse {repeat_on} not found")

            x_sweep = cfg["sweep"][x_key]
            x_param = sweep2param(x_info["param_key"], x_sweep)
            repeat_pulse.set_param(x_info["param_key"], x_param)

            loop_n = 2 * times if repeat_on == "X90_pulse" else times

            return ModularProgramV2(
                soccfg,
                cfg,
                modules=[
                    Reset("reset", modules.get("reset")),
                    Pulse("X90_pulse", X90_pulse),
                    LoadValue(
                        "load_repeat_count",
                        values=list(loop_n),
                        idx_reg="times",
                        val_reg="repeat_count",
                    ),
                    Repeat("zigzag_loop", n="repeat_count").add_content(
                        Pulse(f"loop_{repeat_on}", repeat_pulse)
                    ),
                    Readout("readout", modules["readout"]),
                ],
                sweep=[("times", len(times)), (x_key, x_sweep)],
            ).acquire(
                soc, progress=False, round_hook=update_hook, **(acquire_kwargs or {})
            )

        with LivePlot2D("Times", x_info["name"]) as viewer:
            signals = run_task(
                task=Task(
                    measure_fn=measure_fn,
                    result_shape=(len(times), len(values)),
                    pbar_n=_cfg["rounds"],
                ),
                init_cfg=_cfg,
                on_update=lambda ctx: viewer.update(
                    times.astype(np.float64),
                    values,
                    zigzag_signal2real(np.asarray(ctx.root_data)),
                ),
            )
            signals = np.asarray(signals, dtype=np.complex128)

        # record last cfg and result
        self.last_cfg = cast(ZigZagCfg, deepcopy(cfg))
        self.last_result = (times, values, signals)

        return times, values, signals

    def analyze(
        self,
        result: Optional[ZigZagSweepResult] = None,
        find_range: tuple[Optional[float], Optional[float]] = (None, None),
    ) -> tuple[float, Figure]:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        times, values, signals = result

        real_signals = zigzag_signal2real(signals)  # (times , values)
        valid_cutoff = np.min(np.sum(~np.isnan(real_signals), axis=0))

        if valid_cutoff < 2:
            raise ValueError("Not enough valid data points for analysis")

        times = times[:valid_cutoff]
        real_signals = real_signals[:valid_cutoff]

        val_diff = np.sum(np.abs(np.diff(real_signals, axis=0)), axis=0)
        loss = gaussian_filter1d(val_diff, sigma=1)

        if find_range[0] is not None:
            loss[values < find_range[0]] = np.nan
        if find_range[1] is not None:
            loss[values > find_range[1]] = np.nan
        min_value = values[np.nanargmin(loss)]

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

        dx = (values[1] - values[0]) * 0.5
        dy = (times[1] - times[0]) * 0.5
        ax1.imshow(
            zigzag_signal2real(signals),
            aspect="auto",
            extent=[values[0] - dx, values[-1] + dx, times[0] - dy, times[-1] + dy],
            origin="lower",
            interpolation="none",
        )
        ax1.set_ylabel("Number of gate")
        ax2.plot(values, loss, marker=".")
        ax2.axvline(
            x=min_value, color="red", linestyle="--", label=f"x = {min_value:.3f}"
        )
        ax2.grid(True)
        ax2.legend()
        ax2.set_xlabel("Sweep value (a.u.)")
        ax2.set_ylabel("Loss (a.u.)")

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

        times = times.astype(np.float64)

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
