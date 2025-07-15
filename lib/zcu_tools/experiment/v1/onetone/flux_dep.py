from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from numpy import ndarray
from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.utils import map2adcfreq, sweep2array
from zcu_tools.liveplot import LivePlotter2DwithLine
from zcu_tools.notebook.single_qubit.process import minus_background, rescale
from zcu_tools.program.v1 import OneToneProgram
from zcu_tools.utils.datasaver import save_data

from ..template import sweep2D_soft_soft_template

FluxDepResultType = Tuple[np.ndarray, np.ndarray, np.ndarray]


def fluxdep_signal2real(signals: np.ndarray) -> np.ndarray:
    return rescale(minus_background(np.abs(signals), axis=1), axis=1)


class FluxDepExperiment(AbsExperiment[FluxDepResultType]):
    def run(
        self,
        soc,
        soccfg,
        cfg: Dict[str, Any],
        *,
        progress: bool = True,
    ) -> FluxDepResultType:
        cfg = deepcopy(cfg)

        if cfg["dev"]["flux_dev"] == "none":
            raise NotImplementedError("Flux sweep but get flux_dev == 'none'")

        res_pulse = cfg["dac"]["res_pulse"]

        fpt_sweep = cfg["sweep"]["freq"]
        flx_sweep = cfg["sweep"]["flux"]

        fpts = sweep2array(fpt_sweep, allow_array=True)
        fpts = map2adcfreq(soccfg, fpts, res_pulse["ch"], cfg["adc"]["chs"][0])
        flxs = sweep2array(flx_sweep, allow_array=True)

        del cfg["sweep"]

        def x_updateCfg(cfg, _, flx) -> None:
            cfg["dev"]["flux"] = flx

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
                "Flux (a.u.)",
                "Frequency (MHz)",
                line_axis=1,
                num_lines=2,
                disable=not progress,
            ),
            xs=flxs,
            ys=fpts,
            x_updateCfg=x_updateCfg,
            y_updateCfg=y_updateCfg,
            signal2real=fluxdep_signal2real,
            progress=progress,
        )

        self.last_cfg = cfg
        self.last_result = (flxs, fpts, signals2D)

        return flxs, fpts, signals2D

    def analyze(
        self,
        result: Optional[FluxDepResultType] = None,
        mA_c: Optional[float] = None,
        mA_e: Optional[float] = None,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        As, fpts, signals2D = result

        from zcu_tools.notebook.analysis.fluxdep.interactive import InteractiveLines

        actline = InteractiveLines(
            signals2D,
            mAs=1e3 * As,
            fpts=fpts,
            mA_c=mA_c,
            mA_e=mA_e,
            use_phase=False,
        )

        return actline

    def save(
        self,
        filepath: str,
        result: Optional[FluxDepResultType] = None,
        comment: Optional[str] = None,
        tag: str = "onetone/flux_dep",
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
