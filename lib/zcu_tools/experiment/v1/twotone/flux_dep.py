from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.utils import sweep2array
from zcu_tools.liveplot import LivePlotter2DwithLine
from zcu_tools.program.v1.twotone import RFreqTwoToneProgram
from zcu_tools.utils.datasaver import save_data
from zcu_tools.utils.process import minus_background

from ..template import sweep2D_soft_hard_template

FluxDepResultType = Tuple[np.ndarray, np.ndarray, np.ndarray]


def fluxdep_signal2real(signals: np.ndarray) -> np.ndarray:
    return np.abs(minus_background(signals, axis=1))


class FluxDepExperiment(AbsExperiment[FluxDepResultType]):
    """Two-tone flux dependence experiment.

    Sweeps flux bias and qubit frequency to map out the
    qubit transition as a function of magnetic flux.
    """

    def run(
        self,
        soc,
        soccfg,
        cfg: Dict[str, Any],
        *,
        progress: bool = True,
    ) -> FluxDepResultType:
        cfg = deepcopy(cfg)

        flx_sweep = cfg["sweep"]["flux"]
        fpt_sweep = cfg["sweep"]["freq"]

        cfg["sweep"] = {"freq": fpt_sweep}

        flxs = sweep2array(flx_sweep, allow_array=True)
        fpts = sweep2array(fpt_sweep)

        if cfg["dev"]["flux_dev"] == "none":
            raise ValueError("Flux sweep requires flux_dev != 'none'")

        cfg["dev"]["flux"] = flxs[0]

        def updateCfg(cfg: Dict[str, Any], _: int, flx: float) -> None:
            cfg["dev"]["flux"] = flx

        def measure_fn(
            cfg: Dict[str, Any], cb: Optional[Callable[..., None]]
        ) -> np.ndarray:
            prog = RFreqTwoToneProgram(soccfg, cfg)
            (avgi, avgq), _ = prog.acquire(soc, progress=False, callback=cb)
            return avgi[0] + 1j * avgq[0]

        signals2D = sweep2D_soft_hard_template(
            cfg,
            measure_fn,
            LivePlotter2DwithLine(
                "Flux (a.u.)",
                "Frequency (MHz)",
                line_axis=1,
                num_lines=2,
                disable=not progress,
            ),
            xs=flxs,
            ys=fpts,
            updateCfg=updateCfg,
            signal2real=fluxdep_signal2real,
            progress=progress,
        )

        prog = RFreqTwoToneProgram(soccfg, cfg)
        fpts_real = prog.get_sweep_pts()

        self.last_cfg = cfg
        self.last_result = (flxs, fpts_real, signals2D)

        return flxs, fpts_real, signals2D

    def analyze(
        self,
        result: Optional[FluxDepResultType] = None,
    ) -> None:
        raise NotImplementedError(
            "Analysis not implemented for two-tone flux dependence"
        )

    def save(
        self,
        filepath: str,
        result: Optional[FluxDepResultType] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/flux_dep",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        As, fpts, signals2D = result

        save_data(
            filepath=filepath,
            x_info={"name": "Frequency", "unit": "Hz", "values": fpts * 1e6},
            y_info={"name": "Current", "unit": "A", "values": As},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals2D},
            comment=comment,
            tag=tag,
            **kwargs,
        )
