from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import numpy as np

from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.utils import (
    format_sweep1D,
    set_output_in_dev_cfg,
    sweep2array,
)
from zcu_tools.experiment.v2.runner import HardTask, Runner, SoftTask
from zcu_tools.liveplot import LivePlotter1D
from zcu_tools.program.v2 import OneToneProgram, Readout, sweep2param
from zcu_tools.utils.datasaver import save_data

JPACheckResultType = Tuple[np.ndarray, np.ndarray, np.ndarray]


def jpa_check_signal2real(signals: np.ndarray) -> np.ndarray:
    return np.abs(signals)


class JPACheckExperiment(AbsExperiment[JPACheckResultType]):
    def run(self, soc, soccfg, cfg: Dict[str, Any]) -> JPACheckResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        cfg["sweep"] = format_sweep1D(cfg["sweep"], "freq")

        fpts = sweep2array(cfg["sweep"]["freq"])  # predicted frequency points

        Readout.set_param(
            cfg["readout"], "freq", sweep2param("freq", cfg["sweep"]["freq"])
        )

        outputs = np.array([0, 1])
        OUTPUT_MAP = {0: "off", 1: "on"}

        with LivePlotter1D(
            "Frequency (MHz)", "Magnitude", segment_kwargs=dict(num_lines=2)
        ) as viewer:
            signals = Runner(
                task=SoftTask(
                    sweep_name="JPA on/off",
                    sweep_values=outputs,
                    update_cfg_fn=lambda i, ctx, output: set_output_in_dev_cfg(
                        ctx.cfg["dev"], OUTPUT_MAP[output], label="jpa_rf_dev"
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
                    fpts,
                    jpa_check_signal2real(np.asarray(ctx.get_data())),  # type: ignore
                ),
            ).run(cfg)
            signals = np.asarray(signals)  # type: ignore

        # record last cfg and result
        self.last_cfg = cfg
        self.last_result = (outputs, fpts, signals)

        return outputs, fpts, signals

    def analyze(self, result: Optional[JPACheckResultType] = None) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        outputs, fpts, signals2D = result

        raise NotImplementedError("analysis not yet implemented")

    def save(
        self,
        filepath: str,
        result: Optional[JPACheckResultType] = None,
        comment: Optional[str] = None,
        tag: str = "jpa/check",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        outputs, fpts, signals2D = result

        save_data(
            filepath=filepath,
            x_info={"name": "Frequency", "unit": "Hz", "values": fpts * 1e6},
            y_info={"name": "JPA Output", "unit": "a.u.", "values": outputs},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals2D},
            comment=comment,
            tag=tag,
            **kwargs,
        )
