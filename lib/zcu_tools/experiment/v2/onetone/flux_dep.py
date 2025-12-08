from __future__ import annotations

from copy import deepcopy

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Mapping, Optional, Tuple

from zcu_tools.device import DeviceInfo
from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.utils import set_flux_in_dev_cfg, sweep2array
from zcu_tools.liveplot import LivePlotter2DwithLine
from zcu_tools.notebook.analysis.fluxdep.interactive import InteractiveLines
from zcu_tools.program.v2 import OneToneProgram, OneToneProgramCfg, Readout, sweep2param
from zcu_tools.utils.datasaver import save_data

from ..runner import HardTask, SoftTask, TaskConfig, run_task

FluxDepResultType = Tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.complex128]
]


def fluxdep_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return np.abs(signals)


class FluxDepTaskConfig(TaskConfig, OneToneProgramCfg):
    dev: Mapping[str, DeviceInfo]


class FluxDepExperiment(AbsExperiment):
    def run(self, soc, soccfg, cfg: FluxDepTaskConfig) -> FluxDepResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        assert "sweep" in cfg
        fpt_sweep = cfg["sweep"]["freq"]
        flx_sweep = cfg["sweep"]["flux"]

        # remove flux from sweep dict, will be handled by soft loop
        cfg["sweep"] = {"freq": fpt_sweep}

        dev_values: NDArray[np.float64] = sweep2array(flx_sweep, allow_array=True)
        fpts = sweep2array(fpt_sweep)  # predicted frequency points

        Readout.set_param(cfg["readout"], "freq", sweep2param("freq", fpt_sweep))

        with LivePlotter2DwithLine(
            "Flux device value", "Frequency (MHz)", line_axis=1, num_lines=10
        ) as viewer:
            signals = run_task(
                task=SoftTask(
                    sweep_name="flux",
                    sweep_values=dev_values.tolist(),
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
                init_cfg=cfg,
                update_hook=lambda ctx: viewer.update(
                    dev_values,
                    fpts,
                    fluxdep_signal2real(np.asarray(ctx.data)),
                ),
            )
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
            signals2D,
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
