from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.utils import sweep2array
from zcu_tools.liveplot import LivePlotter2DwithLine
from zcu_tools.program.v2 import OneToneProgram, sweep2param
from zcu_tools.utils.datasaver import save_data
from zcu_tools.utils.process import minus_background
from zcu_tools.notebook.analysis.fluxdep.interactive import InteractiveLines

from ..template import sweep2D_soft_hard_template

FluxDepResultType = Tuple[np.ndarray, np.ndarray, np.ndarray]


def fluxdep_signal2real(signals: np.ndarray) -> np.ndarray:
    return minus_background(np.abs(signals), axis=1)


class FluxDepExperiment(AbsExperiment[FluxDepResultType]):
    def run(
        self,
        soc,
        soccfg,
        cfg: Dict[str, Any],
        *,
        progress: bool = True,
    ) -> FluxDepResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        res_pulse = cfg["readout"]["pulse_cfg"]
        fpt_sweep = cfg["sweep"]["freq"]
        flx_sweep = cfg["sweep"]["flux"]

        # remove flux from sweep dict, will be handled by soft loop
        cfg["sweep"] = {
            "freq": fpt_sweep,
        }

        As = sweep2array(flx_sweep, allow_array=True)
        fpts = sweep2array(fpt_sweep)  # predicted frequency points

        res_pulse["freq"] = sweep2param("freq", fpt_sweep)

        cfg["dev"]["flux"] = As[0]  # set initial flux

        def updateCfg(cfg, _, mA) -> None:
            cfg["dev"]["flux"] = mA * 1e-3

        def measure_fn(
            cfg: Dict[str, Any], cb: Optional[Callable[..., None]]
        ) -> np.ndarray:
            prog = OneToneProgram(soccfg, cfg)
            return prog.acquire(soc, progress=False, callback=cb)[0][0].dot([1, 1j])

        signals2D = sweep2D_soft_hard_template(
            cfg,
            measure_fn,
            LivePlotter2DwithLine(
                "Flux (mA)",
                "Frequency (MHz)",
                line_axis=1,
                num_lines=2,
                disable=not progress,
            ),
            xs=1e3 * As,
            ys=fpts,
            updateCfg=updateCfg,
            signal2real=fluxdep_signal2real,
            progress=progress,
        )

        # get the actual frequency points
        prog = OneToneProgram(soccfg, cfg)
        fpts = prog.get_pulse_param("readout_pulse", "freq", as_array=True)
        assert isinstance(fpts, np.ndarray), "fpts should be an array"

        # record last cfg and result
        self.last_cfg = cfg
        self.last_result = (As, fpts, signals2D)

        return As, fpts, signals2D

    def analyze(
        self,
        result: Optional[FluxDepResultType] = None,
        mA_c: Optional[float] = None,
        mA_e: Optional[float] = None,
    ) -> InteractiveLines:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        As, fpts, signals2D = result

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
