from __future__ import annotations

from copy import deepcopy
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from typing_extensions import NotRequired
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

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
from zcu_tools.utils.process import rotate2real

# (lens, signals)
LengthResultType = Tuple[NDArray[np.float64], NDArray[np.complex128]]


def bathreset_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return rotate2real(signals).real


class LengthTaskConfig(TaskConfig, ModularProgramCfg):
    reset: NotRequired[ResetCfg]
    init_pulse: PulseCfg
    tested_reset: ResetCfg
    readout: ReadoutCfg


class LengthExperiment(AbsExperiment):
    def run(
        self, soc, soccfg, cfg: LengthTaskConfig, detune: float = 0.0
    ) -> LengthResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        # Check that reset pulse is dual pulse type
        if cfg["tested_reset"]["type"] != "bath":
            raise ValueError("This experiment only supports bath reset")

        assert "sweep" in cfg
        cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")

        lens = sweep2array(cfg["sweep"]["length"])  # predicted frequency points

        len_spans = sweep2param("length", cfg["sweep"]["length"])
        Reset.set_param(cfg["tested_reset"], "length", len_spans)
        Reset.set_param(cfg["tested_reset"], "pi2_phase", 360 * detune * len_spans)

        with LivePlotter1D("Length (us)", "Signal (a.u.)") as viewer:
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
                    result_shape=(len(lens),),
                ),
                init_cfg=cfg,
                update_hook=lambda ctx: viewer.update(
                    lens, bathreset_signal2real(ctx.data)
                ),
            )

        # Cache results
        self.last_cfg = cfg
        self.last_result = (lens, signals)

        return lens, signals

    def analyze(self, result: Optional[LengthResultType] = None) -> Figure:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        lens, signals = result

        real_signals = bathreset_signal2real(signals)

        fig, ax = plt.subplots()

        ax.plot(lens, real_signals, marker=".")
        ax.set_xlabel("Length (us)")
        ax.set_ylabel("Signal (a.u.)")
        ax.set_title("Bath Reset Length Measurement")
        ax.grid(True)

        return fig

    def save(
        self,
        filepath: str,
        result: Optional[LengthResultType] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/reset/bath/length",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        lens, signals = result

        save_data(
            filepath=filepath,
            x_info={"name": "Length", "unit": "s", "values": lens * 1e-6},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals},
            comment=comment,
            tag=tag,
            **kwargs,
        )

    def load(self, filepath: str, **kwargs) -> LengthResultType:
        signals, lens, _ = load_data(filepath, **kwargs)
        assert lens is not None
        assert len(lens.shape) == 1 and len(signals.shape) == 1
        assert lens.shape == signals.shape

        lens = lens * 1e6  # s -> us

        lens = lens.astype(np.float64)
        signals = signals.astype(np.complex128)

        self.last_cfg = None
        self.last_result = (lens, signals)

        return lens, signals
