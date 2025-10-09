from __future__ import annotations

import warnings
from copy import deepcopy
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np

from zcu_tools.device import GlobalDeviceManager
from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.utils.single_shot import singleshot_ge_analysis
from zcu_tools.program.v2 import TwoToneProgram, sweep2param
from zcu_tools.utils.datasaver import save_data

# (signals)
SingleShotResultType = np.ndarray


class SingleShotExperiment(AbsExperiment[SingleShotResultType]):
    """Single-shot measurement experiment.

    Performs single-shot readout measurements to characterize the fidelity of
    ground and excited state discrimination. The experiment initializes the qubit
    in either ground or excited state and measures the readout signal for each shot.

    The experiment performs:
    1. Initial reset (optional)
    2. Qubit pulse with variable gain (0 for ground, full gain for excited)
    3. Readout to measure single-shot signals

    The measurement produces raw IQ data for statistical analysis of state discrimination.
    """

    def run(
        self, soc, soccfg, cfg: Dict[str, Any], *, progress: bool = True
    ) -> SingleShotResultType:
        cfg = deepcopy(cfg)  # avoid in-place modification

        # Validate and setup configuration
        if cfg.setdefault("rounds", 1) != 1:
            warnings.warn("rounds will be overwritten to 1 for singleshot measurement")
            cfg["rounds"] = 1

        if "reps" in cfg:
            warnings.warn("reps will be overwritten by singleshot measurement shots")
        cfg["reps"] = cfg["shots"]

        if "sweep" in cfg:
            warnings.warn("sweep will be overwritten by singleshot measurement")

        # Create ge sweep: 0 (ground) and full gain (excited)
        cfg["sweep"] = {
            "ge": {"start": 0, "stop": cfg["qub_pulse"]["gain"], "expts": 2}
        }

        # Set qubit pulse gain from sweep parameter
        cfg["qub_pulse"]["gain"] = sweep2param("ge", cfg["sweep"]["ge"])

        # Set flux device
        GlobalDeviceManager.setup_devices(cfg["dev"], progress=True)

        # Create program and acquire data
        prog = TwoToneProgram(soccfg, deepcopy(cfg))
        prog.acquire(soc, progress=progress)

        # Get raw IQ data
        acc_buf = prog.get_raw()
        assert acc_buf is not None, "acc_buf should not be None"

        avgiq = (
            acc_buf[0] / list(prog.ro_chs.values())[0]["length"]
        )  # (reps, *sweep, 1, 2)
        i0, q0 = avgiq[..., 0, 0], avgiq[..., 0, 1]  # (reps, *sweep)
        signals = np.array(i0 + 1j * q0)  # (reps, *sweep)

        # Swap axes to (*sweep, reps) format: (ge, shots)
        signals = np.swapaxes(signals, 0, -1)

        # Cache results
        self.last_cfg = cfg
        self.last_result = signals

        return signals

    def analyze(
        self,
        result: Optional[SingleShotResultType] = None,
        backend: Literal["center", "regression", "pca"] = "pca",
    ) -> Tuple[float, float, float, np.ndarray]:
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
