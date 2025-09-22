from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.utils import format_sweep1D, sweep2array
from zcu_tools.liveplot import LivePlotter1D, LivePlotter2DwithLine
from zcu_tools.program.v2 import (
    ModularProgramV2,
    Pulse,
    Repeat,
    make_readout,
    make_reset,
    sweep2param,
)
from zcu_tools.utils.datasaver import save_data
from zcu_tools.utils.process import rotate2real

from ..template import sweep1D_soft_template, sweep2D_soft_hard_template


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

        cfg["zigzag_time"] = times[0]  # initial value

        def updateCfg(cfg: Dict[str, Any], _: int, time: Any) -> None:
            cfg["zigzag_time"] = time

        def measure_fn(cfg: Dict[str, Any], callback) -> np.ndarray:
            repeat_time = cfg["zigzag_time"]

            if repeat_on == "X90_pulse":
                sequence = list(range(2 * repeat_time))
            elif repeat_on == "X180_pulse":
                sequence = list(range(repeat_time))

            prog = ModularProgramV2(
                soccfg,
                cfg,
                modules=[
                    make_reset("reset", cfg.get("reset")),
                    Pulse(name="X90_pulse", cfg=X90_pulse),
                    *[
                        Pulse(
                            name=f"{repeat_on}_{i}",
                            cfg=cfg[repeat_on],
                            pulse_name=f"{repeat_on}_0",
                        )
                        for i in sequence
                    ],
                    make_readout("readout", cfg["readout"]),
                ],
            )

            return prog.acquire(soc, progress=False, callback=callback)[0][0].dot(
                [1, 1j]
            )

        signals = sweep1D_soft_template(
            cfg,
            measure_fn,
            LivePlotter1D("Times", "Signal", disable=not progress),
            xs=times,
            updateCfg=updateCfg,
            signal2real=zigzag_signal2real,
            progress=progress,
        )

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
        "length": {"name": "Length (us)", "param_key": "length"},
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
        if x_key == "length" and cfg[repeat_on]["style"] == "gauss":
            cfg[repeat_on]["sigma"] = cfg[repeat_on]["length"] / 5

        def updateCfg(cfg: Dict[str, Any], _: int, time: Any) -> None:
            cfg["zigzag_time"] = time

        updateCfg(cfg, 0, times[0])  # initial update

        def make_prog(cfg: Dict[str, Any], repeat_time: int) -> ModularProgramV2:
            if repeat_on == "X90_pulse":
                pulse_num = 2 * repeat_time
            elif repeat_on == "X180_pulse":
                pulse_num = repeat_time

            prog = ModularProgramV2(
                soccfg,
                cfg,
                modules=[
                    make_reset("reset", cfg.get("reset")),
                    Pulse(name="X90_pulse", cfg=X90_pulse),
                    Repeat(
                        name="zigzag_loop",
                        n=pulse_num,
                        sub_module=Pulse(name=f"loop_{repeat_on}", cfg=cfg[repeat_on]),
                    ),
                    make_readout("readout", cfg["readout"]),
                ],
            )

            return prog

        def measure_fn(cfg: Dict[str, Any], callback) -> np.ndarray:
            prog = make_prog(cfg, repeat_time=cfg["zigzag_time"])
            return prog.acquire(soc, progress=False, callback=callback)[0][0].dot(
                [1, 1j]
            )

        make_prog(cfg, np.max(times))  # may raise error

        signals = sweep2D_soft_hard_template(
            cfg,
            measure_fn,
            LivePlotter2DwithLine(
                "Times",
                x_info["name"],
                line_axis=1,
                num_lines=3,
                disable=not progress,
            ),
            xs=times,
            ys=values,
            updateCfg=updateCfg,
            signal2real=zigzag_signal2real,
            progress=progress,
        )

        prog = ModularProgramV2(
            soccfg, cfg, modules=[Pulse(name=repeat_on, cfg=cfg[repeat_on])]
        )
        real_values = prog.get_pulse_param(
            repeat_on, x_info["param_key"], as_array=True
        )
        real_values += values[0] - real_values[0]

        # record last cfg and result
        self.last_cfg = cfg
        self.last_result = (times, real_values, signals)

        return times, real_values, signals

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
