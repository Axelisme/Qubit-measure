from __future__ import annotations

import warnings
from copy import deepcopy
from typing import Literal, Optional, Tuple, cast

import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.utils import make_ge_sweep
from zcu_tools.experiment.utils.single_shot import singleshot_ge_analysis
from zcu_tools.experiment.v2.runner import (
    HardTask,
    TaskConfig,
    TaskContextView,
    run_task,
)
from zcu_tools.program.v2 import (
    ModularProgramV2,
    Pulse,
    PulseCfg,
    Readout,
    Reset,
    TwoToneProgramCfg,
    sweep2param,
)
from zcu_tools.utils.datasaver import save_data

# (signals)
SingleShotResultType = NDArray[np.complex128]


class SingleShotTaskConfig(TaskConfig, TwoToneProgramCfg):
    probe_pulse: PulseCfg
    shots: int


class SingleShotExperiment(AbsExperiment):
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

    def run(self, soc, soccfg, cfg: SingleShotTaskConfig) -> SingleShotResultType:
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
        cfg["sweep"] = {"ge": make_ge_sweep()}

        # Set qubit pulse gain from sweep parameter
        Pulse.set_param(
            cfg["probe_pulse"], "on/off", sweep2param("ge", cfg["sweep"]["ge"])
        )

        def measure_fn(ctx: TaskContextView, _):
            prog = ModularProgramV2(
                soccfg,
                ctx.cfg,
                modules=[
                    Reset("reset", ctx.cfg.get("reset", {"type": "none"})),
                    Pulse("init_pulse", ctx.cfg.get("init_pulse")),
                    Pulse("probe_pulse", ctx.cfg["probe_pulse"]),
                    Readout("readout", ctx.cfg["readout"]),
                ],
            )
            prog.acquire(soc, progress=True)

            acc_buf = prog.get_raw()
            assert acc_buf is not None

            length = cast(int, list(prog.ro_chs.values())[0]["length"])
            avgiq = acc_buf[0] / length  # (reps, *sweep, 1, 2)

            return avgiq

        def raw2signal_fn(avgiq: NDArray[np.float64]) -> NDArray[np.complex128]:
            i0, q0 = avgiq[..., 0, 0], avgiq[..., 0, 1]  # (reps, *sweep)
            signals = np.array(i0 + 1j * q0)  # (reps, *sweep)

            # Swap axes to (*sweep, reps) format: (ge, shots)
            signals = np.swapaxes(signals, 0, -1)

            return signals

        signals = run_task(
            task=HardTask(
                measure_fn=measure_fn,
                raw2signal_fn=raw2signal_fn,
                result_shape=(2, cfg["shots"]),
            ),
            init_cfg=cfg,
        )

        # Cache results
        self.last_cfg = cfg
        self.last_result = signals

        return signals

    def analyze(
        self,
        result: Optional[SingleShotResultType] = None,
        backend: Literal["center", "regression", "pca"] = "pca",
        **kwargs,
    ) -> Tuple[float, float, float, np.ndarray, dict, Figure]:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        signals = result

        return singleshot_ge_analysis(signals, backend=backend, **kwargs)

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
