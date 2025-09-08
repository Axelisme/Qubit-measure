from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional, Tuple, Literal

import numpy as np

from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.utils import format_sweep1D, sweep2array
from zcu_tools.liveplot import LivePlotter1D, LivePlotter2D
from zcu_tools.program.v2 import (
    ModularProgramV2,
    Pulse,
    make_readout,
    make_reset,
    sweep2param,
)
from zcu_tools.utils.datasaver import save_data
from zcu_tools.utils.process import rotate2real

from ..template import (
    sweep1D_soft_template,
    sweep2D_soft_hard_template,
    sweep2D_soft_template,
)


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
        progress: bool = True,
    ) -> ZigZagResultType:
        cfg = deepcopy(cfg)  # avoid in-place modification

        cfg["sweep"] = format_sweep1D(cfg["sweep"], "times")
        times = sweep2array(cfg["sweep"]["times"], allow_array=True)  # predicted
        del cfg["sweep"]

        cfg["zigzag_pi_time"] = times[0]  # initial value

        def updateCfg(cfg: Dict[str, Any], _: int, time: Any) -> None:
            cfg["zigzag_pi_time"] = time

        def measure_fn(cfg: Dict[str, Any], callback) -> np.ndarray:
            pi_time = cfg["zigzag_pi_time"]

            prog = ModularProgramV2(
                soccfg,
                cfg,
                modules=[
                    make_reset("reset", cfg.get("reset")),
                    Pulse(name="X90_pulse", cfg=cfg["X90_pulse"]),
                    *[
                        Pulse(
                            name=f"X180_pulse_{i}",
                            cfg=cfg["X180_pulse"],
                        )
                        for i in range(pi_time)
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

    def run_hard(
        self,
        soc,
        soccfg,
        cfg: Dict[str, Any],
        *,
        progress: bool = True,
    ) -> ZigZagSweepResultType:
        cfg = deepcopy(cfg)  # avoid in-place modification

        time_sweep = cfg["sweep"].pop("times")
        times = sweep2array(time_sweep, allow_array=True)  # predicted

        # extract sweep parameters
        x_key = list(cfg["sweep"].keys())[0]
        if x_key not in self.SWEEP_MAP:
            raise ValueError(f"Unsupported sweep key: {x_key}")
        x_info = self.SWEEP_MAP[x_key]
        values = sweep2array(cfg["sweep"][x_key])  # predicted

        cfg["X180_pulse"][x_info["param_key"]] = sweep2param(
            x_info["param_key"], cfg["sweep"][x_key]
        )

        def updateCfg(cfg: Dict[str, Any], _: int, time: Any) -> None:
            cfg["zigzag_pi_time"] = time

        updateCfg(cfg, 0, times[0])  # initial update

        def measure_fn(cfg: Dict[str, Any], callback) -> np.ndarray:
            pi_time = cfg["zigzag_pi_time"]

            prog = ModularProgramV2(
                soccfg,
                cfg,
                modules=[
                    make_reset("reset", cfg.get("reset")),
                    Pulse(name="X90_pulse", cfg=cfg["X90_pulse"]),
                    *[
                        Pulse(
                            name=f"X180_pulse_{i}",
                            cfg=cfg["X180_pulse"],
                            pulse_name="X180_pulse",
                        )
                        for i in range(pi_time)
                    ],
                    make_readout("readout", cfg["readout"]),
                ],
            )

            return prog.acquire(soc, progress=False, callback=callback)[0][0].dot(
                [1, 1j]
            )

        signals = sweep2D_soft_hard_template(
            cfg,
            measure_fn,
            LivePlotter2D(
                "Times",
                x_info["name"],
                disable=not progress,
            ),
            xs=times,
            ys=values,
            updateCfg=updateCfg,
            signal2real=zigzag_signal2real,
            progress=progress,
        )

        prog = ModularProgramV2(
            soccfg, cfg, modules=[Pulse(name="X180_pulse", cfg=cfg["X180_pulse"])]
        )
        real_values = prog.get_pulse_param(
            "X180_pulse", x_info["param_key"], as_array=True
        )
        real_values += values[0] - real_values[0]

        # record last cfg and result
        self.last_cfg = cfg
        self.last_result = (times, real_values, signals)

        return times, real_values, signals

    def run_soft(
        self,
        soc,
        soccfg,
        cfg: Dict[str, Any],
        *,
        progress: bool = True,
    ) -> ZigZagSweepResultType:
        cfg = deepcopy(cfg)  # avoid in-place modification

        time_sweep = cfg["sweep"].pop("times")
        times = sweep2array(time_sweep, allow_array=True)  # predicted

        # extract sweep parameters
        x_key = list(cfg["sweep"].keys())[0]
        if x_key not in self.SWEEP_MAP:
            raise ValueError(f"Unsupported sweep key: {x_key}")
        x_info = self.SWEEP_MAP[x_key]
        values = sweep2array(cfg["sweep"][x_key])  # predicted

        del cfg["sweep"]  # no hard sweep

        def updateCfg_x(cfg: Dict[str, Any], _: int, time: Any) -> None:
            cfg["zigzag_pi_time"] = time

        def updateCfg_y(cfg: Dict[str, Any], _: int, value: Any) -> None:
            cfg["X180_pulse"][x_info["param_key"]] = value

        updateCfg_x(cfg, 0, times[0])  # initial update
        updateCfg_y(cfg, 0, values[0])  # initial update

        def measure_fn(cfg: Dict[str, Any], callback) -> np.ndarray:
            pi_time = cfg["zigzag_pi_time"]

            prog = ModularProgramV2(
                soccfg,
                cfg,
                modules=[
                    make_reset("reset", cfg.get("reset")),
                    Pulse(name="X90_pulse", cfg=cfg["X90_pulse"]),
                    *[
                        Pulse(
                            name=f"X180_pulse_{i}",
                            cfg=cfg["X180_pulse"],
                            pulse_name="X180_pulse",
                        )
                        for i in range(pi_time)
                    ],
                    make_readout("readout", cfg["readout"]),
                ],
            )

            return prog.acquire(soc, progress=False, callback=callback)[0][0].dot(
                [1, 1j]
            )

        signals = sweep2D_soft_template(
            cfg,
            measure_fn,
            LivePlotter2D(
                "Times",
                x_info["name"],
                disable=not progress,
            ),
            xs=times,
            ys=values,
            updateCfg_x=updateCfg_x,
            updateCfg_y=updateCfg_y,
            signal2real=zigzag_signal2real,
            progress=progress,
        )

        # record last cfg and result
        self.last_cfg = cfg
        self.last_result = (times, values, signals)

        return times, values, signals

    def run(
        self,
        soc,
        soccfg,
        cfg: Dict[str, Any],
        *,
        method: Literal["hard", "soft"] = "hard",
        progress: bool = True,
    ) -> ZigZagSweepResultType:
        if method == "hard":
            return self.run_hard(soc, soccfg, cfg, progress=progress)
        elif method == "soft":
            return self.run_soft(soc, soccfg, cfg, progress=progress)
        else:
            raise ValueError(f"Unsupported method: {method}")

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
