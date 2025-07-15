from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.utils import sweep2array
from zcu_tools.liveplot import LivePlotter2DwithLine
from zcu_tools.program.v2 import OneToneProgram, sweep2param
from zcu_tools.utils.datasaver import save_data
from zcu_tools.utils.process import minus_background, rescale

from ..template import sweep2D_soft_hard_template

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
        gain_ref: float = 0.1,
        progress: bool = True,
    ) -> PowerDepResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        res_pulse = cfg["readout"]["pulse_cfg"]
        pdr_sweep = cfg["sweep"]["gain"]
        fpt_sweep = cfg["sweep"]["freq"]
        reps_ref = cfg["reps"]
        avgs_ref = cfg["rounds"]

        del cfg["sweep"]["gain"]  # use soft for loop here

        res_pulse["freq"] = sweep2param("freq", fpt_sweep)

        pdrs = sweep2array(pdr_sweep, allow_array=True)
        fpts = sweep2array(fpt_sweep)  # predicted frequency points

        res_pulse["gain"] = pdrs[0]  # set initial power

        def updateCfg(cfg, _, pdr) -> None:
            cfg["readout"]["pulse_cfg"]["gain"] = pdr

            # change reps and rounds based on power
            if dynamic_avg:
                dyn_factor = (gain_ref / pdr) ** 2
                if dyn_factor > 1:
                    # increase reps
                    cfg["reps"] = int(reps_ref * dyn_factor)
                    max_reps = min(100 * reps_ref, 1000000)
                    if cfg["reps"] > max_reps:
                        cfg["reps"] = max_reps
                elif cfg["rounds"] > 1:
                    # decrease rounds
                    cfg["rounds"] = int(avgs_ref * dyn_factor)
                    min_avgs = max(int(0.1 * avgs_ref), 1)
                    if cfg["rounds"] < min_avgs:
                        cfg["rounds"] = min_avgs
                else:
                    # decrease reps
                    cfg["reps"] = int(reps_ref * dyn_factor)
                    min_reps = max(int(0.1 * reps_ref), 1)
                    if cfg["reps"] < min_reps:
                        cfg["reps"] = min_reps

        def measure_fn(
            cfg: Dict[str, Any], cb: Optional[Callable[..., None]]
        ) -> np.ndarray:
            prog = OneToneProgram(soccfg, cfg)
            return prog.acquire(soc, progress=False, callback=cb)[0][0].dot([1, 1j])

        signals2D = sweep2D_soft_hard_template(
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
            updateCfg=updateCfg,
            signal2real=pdrdep_signal2real,
            progress=progress,
        )

        # get the actual frequency points
        prog = OneToneProgram(soccfg, cfg)
        fpts = prog.get_pulse_param("readout_pulse", "freq", as_array=True)
        assert isinstance(fpts, np.ndarray), "fpts should be an array"

        # rescale signals2D
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
