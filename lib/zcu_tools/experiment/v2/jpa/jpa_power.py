from __future__ import annotations

from copy import deepcopy
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.utils import set_power_in_dev_cfg, sweep2array
from zcu_tools.experiment.v2.runner import HardTask, SoftTask, TaskConfig, run_task
from zcu_tools.liveplot import LivePlotter2DwithLine
from zcu_tools.program.v2 import OneToneProgram, OneToneProgramCfg, Readout, sweep2param
from zcu_tools.utils.datasaver import save_data

JPAPowerResultType = Tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.complex128]
]


def jpa_power_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return np.abs(signals)


class JPAPowerTaskConfig(TaskConfig, OneToneProgramCfg): ...


class JPAPowerExperiment(AbsExperiment):
    def run(self, soc, soccfg, cfg: JPAPowerTaskConfig) -> JPAPowerResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        assert "sweep" in cfg
        fpt_sweep = cfg["sweep"]["freq"]
        pdr_sweep = cfg["sweep"]["power_dBm"]

        # remove flux from sweep dict, will be handled by soft loop
        cfg["sweep"] = {"freq": fpt_sweep}

        powers = sweep2array(pdr_sweep, allow_array=True)
        fpts = sweep2array(fpt_sweep)  # predicted frequency points

        Readout.set_param(cfg["readout"], "freq", sweep2param("freq", fpt_sweep))

        with LivePlotter2DwithLine(
            "Power (dBm)", "Frequency (MHz)", line_axis=1, num_lines=10
        ) as viewer:
            signals = run_task(
                task=SoftTask(
                    sweep_name="power (dBm)",
                    sweep_values=powers.tolist(),
                    update_cfg_fn=lambda i, ctx, pdr: set_power_in_dev_cfg(
                        ctx.cfg["dev"], pdr, label="jpa_rf_dev"
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
                    powers,
                    fpts,
                    jpa_power_signal2real(np.asarray(ctx.data)),
                ),
            )
            signals = np.asarray(signals)

        # record last cfg and result
        self.last_cfg = cfg
        self.last_result = (powers, fpts, signals)

        return powers, fpts, signals

    def analyze(self, result: Optional[JPAPowerResultType] = None) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        values, fpts, signals2D = result

        raise NotImplementedError("analysis not yet implemented")

    def save(
        self,
        filepath: str,
        result: Optional[JPAPowerResultType] = None,
        comment: Optional[str] = None,
        tag: str = "jpa/power",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        values, fpts, signals2D = result

        save_data(
            filepath=filepath,
            x_info={"name": "Frequency", "unit": "Hz", "values": fpts * 1e6},
            y_info={"name": "JPA Power", "unit": "dBm", "values": values},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals2D},
            comment=comment,
            tag=tag,
            **kwargs,
        )
