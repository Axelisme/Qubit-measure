from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from numpy import ndarray
from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.utils import map2adcfreq, sweep2array
from zcu_tools.liveplot import LivePlotter2DwithLine
from zcu_tools.program.v1 import OneToneProgram
from zcu_tools.utils.datasaver import save_data
from zcu_tools.utils.process import minus_background, rescale

from ..template import sweep2D_soft_soft_template

PowerDepResultType = Tuple[np.ndarray, np.ndarray, np.ndarray]


def pdrdep_signal2real(signals: np.ndarray) -> np.ndarray:
    return rescale(minus_background(np.abs(signals), axis=1), axis=1)


class PowerDepExperiment(AbsExperiment[PowerDepResultType]):
    def run(
        self,
        soc,
        soccfg,
        cfg: Dict[str, Any],
        *,
        dynamic_avg: bool = False,
        gain_ref: float = 1000,
        progress: bool = True,
    ) -> PowerDepResultType:
        cfg = deepcopy(cfg)

        res_pulse = cfg["dac"]["res_pulse"]
        reps_ref = cfg["reps"]
        rounds_ref = cfg.get("rounds", 1)

        fpt_sweep = cfg["sweep"]["freq"]
        pdr_sweep = cfg["sweep"]["gain"]

        fpts = sweep2array(fpt_sweep, allow_array=True)
        fpts = map2adcfreq(soccfg, fpts, res_pulse["ch"], cfg["adc"]["chs"][0])
        pdrs = sweep2array(pdr_sweep, allow_array=True)

        del cfg["sweep"]

        def x_updateCfg(cfg, _, pdr) -> None:
            cfg["dac"]["res_pulse"]["gain"] = pdr

            if dynamic_avg:
                dyn_factor = (gain_ref / pdr) ** 2
                if dyn_factor > 1:
                    cfg["reps"] = int(reps_ref * dyn_factor)
                    max_reps = min(100 * reps_ref, 1000000)
                    if cfg["reps"] > max_reps:
                        cfg["reps"] = max_reps
                elif rounds_ref > 1:  # soft_avgs is rounds
                    cfg["rounds"] = int(rounds_ref * dyn_factor)
                    min_avgs = max(int(0.1 * rounds_ref), 1)
                    if cfg["rounds"] < min_avgs:
                        cfg["rounds"] = min_avgs
                else:
                    cfg["reps"] = int(reps_ref * dyn_factor)
                    min_reps = max(int(0.1 * reps_ref), 1)
                    if cfg["reps"] < min_reps:
                        cfg["reps"] = min_reps

        def y_updateCfg(cfg, _, fpt) -> None:
            cfg["dac"]["res_pulse"]["freq"] = fpt

        def result_fn(ir: int, sum_d, sum2_d) -> ndarray:
            avg_d = [d / (ir + 1) for d in sum_d]
            return avg_d[0].dot([1, 1j])

        def measure_fn(
            cfg: Dict[str, Any], cb: Optional[Callable[..., None]]
        ) -> np.ndarray:
            prog = OneToneProgram(soccfg, cfg)
            sum_d, sum2_d = prog.acquire(soc, progress=False, callback=cb)
            return result_fn(cfg["reps"] - 1, sum_d, sum2_d)

        signals2D = sweep2D_soft_soft_template(
            cfg,
            measure_fn,
            LivePlotter2DwithLine(
                "Power (a.u.)",
                "Frequency (MHz)",
                line_axis=1,
                num_lines=10,
                disable=not progress,
            ),
            xs=pdrs,
            ys=fpts,
            x_updateCfg=x_updateCfg,
            y_updateCfg=y_updateCfg,
            signal2real=pdrdep_signal2real,
            progress=progress,
        )

        self.last_cfg = cfg
        self.last_result = (pdrs, fpts, signals2D)

        return pdrs, fpts, signals2D

    def analyze(
        self,
        result: Optional[PowerDepResultType] = None,
    ) -> None:
        raise NotImplementedError("Not implemented")

    def save(
        self,
        filepath: str,
        result: Optional[PowerDepResultType] = None,
        comment: Optional[str] = None,
        tag: str = "onetone/power_dep",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        pdrs, fpts, signals2D = result

        save_data(
            filepath=filepath,
            x_info={"name": "Frequency", "unit": "Hz", "values": fpts * 1e6},
            y_info={"name": "Power", "unit": "a.u.", "values": pdrs},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals2D},
            comment=comment,
            tag=tag,
            **kwargs,
        )
