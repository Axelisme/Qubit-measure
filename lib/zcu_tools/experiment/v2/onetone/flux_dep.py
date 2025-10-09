from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import numpy as np

from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.utils import set_flux_in_dev_cfg, sweep2array
from zcu_tools.liveplot import LivePlotter2DwithLine
from zcu_tools.notebook.analysis.fluxdep.interactive import InteractiveLines
from zcu_tools.program.v2 import OneToneProgram, set_readout_cfg, sweep2param
from zcu_tools.utils.datasaver import save_data
from zcu_tools.utils.process import minus_background

from ..runner import HardTask, Runner, SoftTask

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

        fpt_sweep = cfg["sweep"]["freq"]
        flx_sweep = cfg["sweep"]["flux"]

        # remove flux from sweep dict, will be handled by soft loop
        cfg["sweep"] = {"freq": fpt_sweep}

        dev_values = sweep2array(flx_sweep, allow_array=True)
        fpts = sweep2array(fpt_sweep)  # predicted frequency points

        set_readout_cfg(cfg["readout"], "freq", sweep2param("freq", fpt_sweep))

        with LivePlotter2DwithLine(
            "Flux device value",
            "Frequency (MHz)",
            line_axis=1,
            num_lines=2,
            disable=not progress,
        ) as viewer:
            signals = Runner(
                task=SoftTask(
                    sweep_name="flux",
                    sweep_values=dev_values,
                    update_cfg_fn=lambda i, ctx, flx: set_flux_in_dev_cfg(
                        ctx.cfg["dev"], flx
                    ),
                    sub_task=HardTask(
                        measure_fn=lambda ctx, update_hook: (
                            OneToneProgram(soccfg, ctx.cfg).acquire(
                                soc, progress=False, callback=update_hook
                            )
                        ),
                        result_shape=(len(fpts),),
                    ),
                ),
                update_hook=lambda ctx: viewer.update(
                    dev_values, fpts, fluxdep_signal2real(np.asarray(ctx.get_data()))
                ),
            ).run(cfg)
            signals = np.asarray(signals)

        # record last cfg and result
        self.last_cfg = cfg
        self.last_result = (dev_values, fpts, signals)

        return dev_values, fpts, signals

    def analyze(
        self,
        result: Optional[FluxDepResultType] = None,
        mA_c: Optional[float] = None,
        mA_e: Optional[float] = None,
    ) -> InteractiveLines:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        values, fpts, signals2D = result

        actline = InteractiveLines(
            signals2D.T,
            mAs=values,
            fpts=fpts,
            mA_c=mA_c,
            mA_e=mA_e,
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

        values, fpts, signals2D = result

        save_data(
            filepath=filepath,
            x_info={"name": "Frequency", "unit": "Hz", "values": fpts * 1e6},
            y_info={"name": "Flux device value", "unit": "a.u.", "values": values},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals2D},
            comment=comment,
            tag=tag,
            **kwargs,
        )
