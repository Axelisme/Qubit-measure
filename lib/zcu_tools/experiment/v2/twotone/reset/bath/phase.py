from __future__ import annotations

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from typing_extensions import NotRequired, Optional, Tuple

from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.utils import format_sweep1D, sweep2array
from zcu_tools.experiment.v2.runner import HardTask, TaskConfig, run_task
from zcu_tools.liveplot import LivePlotter1D
from zcu_tools.program.v2 import (
    ModularProgramCfg,
    ModularProgramV2,
    Pulse,
    PulseCfg,
    Readout,
    ReadoutCfg,
    Reset,
    ResetCfg,
    sweep2param,
)
from zcu_tools.utils.datasaver import load_data, save_data
from zcu_tools.utils.fitting.base import cosfunc, fitcos
from zcu_tools.utils.process import rotate2real

# (phases, signals)
PhaseResultType = Tuple[NDArray[np.float64], NDArray[np.complex128]]


def bathreset_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return rotate2real(signals).real


class PhaseTaskConfig(TaskConfig, ModularProgramCfg):
    reset: NotRequired[ResetCfg]
    init_pulse: PulseCfg
    tested_reset: ResetCfg
    readout: ReadoutCfg


class PhaseExp(AbsExperiment[PhaseResultType, PhaseTaskConfig]):
    def run(self, soc, soccfg, cfg: PhaseTaskConfig) -> PhaseResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        # Check that reset pulse is dual pulse type
        if cfg["tested_reset"]["type"] != "bath":
            raise ValueError("This experiment only supports bath reset")

        assert "sweep" in cfg
        cfg["sweep"] = format_sweep1D(cfg["sweep"], "phase")

        phases = sweep2array(cfg["sweep"]["phase"])  # predicted phase points

        phase_param = sweep2param("phase", cfg["sweep"]["phase"])
        Reset.set_param(cfg["tested_reset"], "pi2_phase", phase_param)

        with LivePlotter1D("Phase (deg)", "Signal (a.u.)") as viewer:
            signals = run_task(
                task=HardTask(
                    measure_fn=lambda ctx, update_hook: (
                        ModularProgramV2(
                            soccfg,
                            ctx.cfg,
                            modules=[
                                Reset("reset", ctx.cfg.get("reset", {"type": "none"})),
                                Pulse("init_pulse", ctx.cfg.get("init_pulse")),
                                Reset("tested_reset", ctx.cfg["tested_reset"]),
                                Readout("readout", ctx.cfg["readout"]),
                            ],
                        ).acquire(soc, progress=False, callback=update_hook)
                    ),
                    result_shape=(len(phases),),
                ),
                init_cfg=cfg,
                update_hook=lambda ctx: viewer.update(
                    phases, bathreset_signal2real(ctx.data)
                ),
            )

        # Cache results
        self.last_cfg = cfg
        self.last_result = (phases, signals)

        return phases, signals

    def analyze(
        self, result: Optional[PhaseResultType] = None
    ) -> Tuple[float, float, Figure]:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        phases, signals = result

        real_signals = bathreset_signal2real(signals)

        pOpt, _ = fitcos(phases, real_signals, fixedparams=[None, None, 1 / 360, None])
        y_fit = cosfunc(phases, *pOpt)

        max_phase = phases[np.argmax(y_fit)]
        min_phase = phases[np.argmin(y_fit)]

        fig, ax = plt.subplots()
        assert isinstance(fig, Figure)

        ax.plot(phases, real_signals, ".-", label="data")
        ax.plot(phases, y_fit, "-", label="fit")
        ax.axvline(
            max_phase, color="C1", linestyle="--", label=f"max: {max_phase:.2f} deg"
        )
        ax.axvline(
            min_phase, color="C2", linestyle="--", label=f"min: {min_phase:.2f} deg"
        )
        ax.set_xlabel("Phase (deg)")
        ax.set_ylabel("Signal (a.u.)")
        ax.legend()
        ax.grid(True)

        fig.tight_layout()

        return max_phase, min_phase, fig

    def save(
        self,
        filepath: str,
        result: Optional[PhaseResultType] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/reset/bath/phase",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        phases, signals = result

        save_data(
            filepath=filepath,
            x_info={"name": "Phase", "unit": "deg", "values": phases},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals},
            comment=comment,
            tag=tag,
            **kwargs,
        )

    def load(self, filepath: str, **kwargs) -> PhaseResultType:
        signals, phases, _ = load_data(filepath, **kwargs)
        assert phases is not None
        assert len(phases.shape) == 1 and len(signals.shape) == 1
        assert phases.shape == signals.shape

        phases = phases.astype(np.float64)
        signals = signals.astype(np.complex128)

        self.last_cfg = None
        self.last_result = (phases, signals)

        return phases, signals
