from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import numpy as np

from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.utils import sweep2array
from zcu_tools.liveplot import LivePlotter2DwithLine
from zcu_tools.program.v2 import OneToneProgram, Readout, sweep2param
from zcu_tools.utils.datasaver import save_data
from zcu_tools.utils.process import minus_background, rescale

from ..runner import HardTask, Runner, SoftTask
from ..utils import wrap_earlystop_check

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
        progress: bool = True,
        earlystop_snr: Optional[float] = None,
    ) -> PowerDepResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        pdr_sweep = cfg["sweep"].pop("gain")

        pdrs = sweep2array(pdr_sweep, allow_array=True)
        fpts = sweep2array(cfg["sweep"]["freq"])  # predicted frequency points

        Readout.set_param(
            cfg["readout"], "freq", sweep2param("freq", cfg["sweep"]["freq"])
        )

        # run experiment
        with LivePlotter2DwithLine(
            "Power (a.u.)",
            "Frequency (MHz)",
            line_axis=1,
            num_lines=10,
            disable=not progress,
        ) as viewer:
            ax1d = viewer.get_ax("1d")

            def measure_fn(ctx, update_hook):
                prog = OneToneProgram(soccfg, ctx.cfg)
                return prog.acquire(
                    soc,
                    progress=False,
                    callback=wrap_earlystop_check(
                        prog,
                        update_hook,
                        earlystop_snr,
                        signal2real_fn=np.abs,
                        snr_hook=lambda snr: ax1d.set_title(f"snr = {snr:.1f}"),  # type: ignore
                    ),
                )

            signals = Runner(
                task=SoftTask(
                    sweep_name="gain",
                    sweep_values=pdrs,
                    update_cfg_fn=lambda _, ctx, pdr: (
                        Readout.set_param(ctx.cfg["readout"], "gain", pdr)
                    ),
                    sub_task=HardTask(
                        measure_fn=measure_fn,
                        result_shape=(len(fpts),),
                    ),
                ),
                update_hook=lambda ctx: viewer.update(
                    pdrs,
                    fpts,
                    pdrdep_signal2real(np.asarray(ctx.get_data())),  # type: ignore
                ),
            ).run(cfg)
            signals = np.asarray(signals)  # type: ignore

        # record last cfg and result
        self.last_cfg = cfg
        self.last_result = (pdrs, fpts, signals)

        return pdrs, fpts, signals

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
