from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.utils import set_flux_in_dev_cfg, sweep2array
from zcu_tools.experiment.v2.runner import HardTask, Runner, SoftTask
from zcu_tools.liveplot import LivePlotter2DwithLine
from zcu_tools.program.v2 import OneToneProgram, Readout, sweep2param
from zcu_tools.utils.datasaver import save_data

JPAFluxResultType = Tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.complex128]
]


def jpa_flux_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return np.abs(signals)


class JPAFluxExperiment(AbsExperiment[JPAFluxResultType]):
    def run(self, soc, soccfg, cfg: Dict[str, Any]) -> JPAFluxResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        fpt_sweep = cfg["sweep"]["freq"]
        flx_sweep = cfg["sweep"]["flux"]

        # remove flux from sweep dict, will be handled by soft loop
        cfg["sweep"] = {"freq": fpt_sweep}

        flxs = sweep2array(flx_sweep, allow_array=True)
        freqs = sweep2array(fpt_sweep)  # predicted frequency points

        Readout.set_param(cfg["readout"], "freq", sweep2param("freq", fpt_sweep))

        with LivePlotter2DwithLine(
            "JPA Flux value", "Probe Frequency (MHz)", line_axis=1, num_lines=10
        ) as viewer:
            signals = Runner(
                task=SoftTask(
                    sweep_name="JPA Flux value",
                    sweep_values=flxs,
                    update_cfg_fn=lambda i, ctx, flx: set_flux_in_dev_cfg(
                        ctx.cfg["dev"], flx, label="jpa_flux_dev"
                    ),
                    sub_task=HardTask(
                        measure_fn=lambda ctx, update_hook: (
                            OneToneProgram(soccfg, ctx.cfg).acquire(
                                soc, progress=False, callback=update_hook
                            )
                        ),
                        result_shape=(len(freqs),),
                    ),
                ),
                update_hook=lambda ctx: viewer.update(
                    flxs,
                    freqs,
                    jpa_flux_signal2real(np.asarray(ctx.get_data())),  # type: ignore
                ),
            ).run(cfg)
            signals = np.asarray(signals)  # type: ignore

        # record last cfg and result
        self.last_cfg = cfg
        self.last_result = (flxs, freqs, signals)

        return flxs, freqs, signals

    def analyze(self, result: Optional[JPAFluxResultType] = None) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        values, fpts, signals2D = result

        raise NotImplementedError("analysis not yet implemented")

    def save(
        self,
        filepath: str,
        result: Optional[JPAFluxResultType] = None,
        comment: Optional[str] = None,
        tag: str = "jpa/flux",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        flxs, freqs, signals2D = result

        save_data(
            filepath=filepath,
            x_info={"name": "Probe Frequency", "unit": "Hz", "values": freqs * 1e6},
            y_info={"name": "JPA Flux value", "unit": "a.u.", "values": flxs},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals2D},
            comment=comment,
            tag=tag,
            **kwargs,
        )
