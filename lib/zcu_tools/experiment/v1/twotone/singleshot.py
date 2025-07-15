from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.notebook.single_qubit.single_shot import singleshot_ge_analysis
from zcu_tools.program.v1 import SingleShotProgram
from zcu_tools.utils.datasaver import save_data

from ...flux import set_flux

SingleShotResultType = np.ndarray


class SingleShotExperiment(AbsExperiment[SingleShotResultType]):
    def run(
        self, soc, soccfg, cfg: Dict[str, Any], *, progress: bool = True
    ) -> SingleShotResultType:
        cfg = deepcopy(cfg)

        set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"])
        prog = SingleShotProgram(soccfg, cfg)
        i0, q0 = prog.acquire(soc, progress=progress)

        signals = np.array(i0 + 1j * q0)  # (shots, ge)
        signals = np.swapaxes(signals, 0, 1)  # (ge, shots)

        self.last_cfg = cfg
        self.last_result = signals

        return signals

    def analyze(
        self,
        result: Optional[SingleShotResultType] = None,
        backend: Literal["center", "regression", "pca"] = "pca",
    ) -> Tuple[float, np.ndarray]:
        if result is None:
            result = self.last_result
        assert result is not None, (
            "No measurement data available. Run experiment first."
        )

        signals = result

        return singleshot_ge_analysis(signals, backend=backend)

    def save(
        self,
        filepath: str,
        result: Optional[SingleShotResultType] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/ge/singleshot",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, (
            "No measurement data available. Run experiment first."
        )

        signals = result

        save_data(
            filepath=filepath,
            x_info={
                "name": "shot",
                "unit": "point",
                "values": np.arange(signals.shape[1]),
            },
            y_info={"name": "ge", "unit": "", "values": np.array([0, 1])},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals},
            comment=comment,
            tag=tag,
            **kwargs,
        )
