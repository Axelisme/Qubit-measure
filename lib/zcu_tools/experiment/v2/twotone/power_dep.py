from __future__ import annotations

from copy import deepcopy
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.utils import sweep2array
from zcu_tools.liveplot import LivePlotter2D
from zcu_tools.program.v2 import TwoToneProgram, TwoToneProgramCfg, sweep2param
from zcu_tools.utils.datasaver import load_data, save_data
from zcu_tools.utils.process import minus_background

from ..runner import HardTask, TaskConfig, run_task

PowerDepResultType = Tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.complex128]
]


def pdrdep_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return np.abs(minus_background(signals, axis=1))


class PowerDepTaskConfig(TaskConfig, TwoToneProgramCfg): ...


class PowerDepExperiment(AbsExperiment):
    def run(self, soc, soccfg, cfg: PowerDepTaskConfig) -> PowerDepResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        # Ensure gain is the outer loop for better visualization
        assert "sweep" in cfg
        cfg["sweep"] = {
            "gain": cfg["sweep"]["gain"],
            "freq": cfg["sweep"]["freq"],
        }

        pdrs = sweep2array(cfg["sweep"]["gain"])  # predicted pulse gains
        fpts = sweep2array(cfg["sweep"]["freq"])  # predicted frequency points

        # Attach both sweep parameters to the qubit pulse
        cfg["qub_pulse"]["gain"] = sweep2param("gain", cfg["sweep"]["gain"])
        cfg["qub_pulse"]["freq"] = sweep2param("freq", cfg["sweep"]["freq"])

        with LivePlotter2D("Pulse Gain (a.u.)", "Frequency (MHz)") as viewer:
            signals = run_task(
                task=HardTask(
                    measure_fn=lambda ctx, update_hook: (
                        TwoToneProgram(soccfg, ctx.cfg).acquire(
                            soc, progress=False, callback=update_hook
                        )
                    ),
                    result_shape=(len(pdrs), len(fpts)),
                ),
                init_cfg=cfg,
                update_hook=lambda ctx: viewer.update(
                    pdrs, fpts, pdrdep_signal2real(ctx.data)
                ),
            )

        # Cache results
        self.last_cfg = cfg
        self.last_result = (pdrs, fpts, signals)

        return pdrs, fpts, signals

    def analyze(
        self,
        result: Optional[PowerDepResultType] = None,
    ) -> None:
        raise NotImplementedError(
            "Analysis not implemented for two-tone power dependence"
        )

    def save(
        self,
        filepath: str,
        result: Optional[PowerDepResultType] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/power_dep",
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

    def load(self, filepath: str, **kwargs) -> PowerDepResultType:
        signals2D, fpts, pdrs = load_data(filepath, **kwargs)
        assert fpts is not None and pdrs is not None
        assert len(fpts.shape) == 1 and len(pdrs.shape) == 1
        assert signals2D.shape == (len(pdrs), len(fpts))

        fpts = fpts * 1e-6  # Hz -> MHz

        pdrs = pdrs.astype(np.float64)
        fpts = fpts.astype(np.float64)
        signals2D = signals2D.astype(np.complex128)

        self.last_cfg = None
        self.last_result = (pdrs, fpts, signals2D)

        return pdrs, fpts, signals2D
