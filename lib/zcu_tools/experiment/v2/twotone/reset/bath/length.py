from __future__ import annotations

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from typeguard import check_type
from typing_extensions import Any, NotRequired, Optional, TypeAlias, TypedDict

from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.utils import format_sweep1D
from zcu_tools.experiment.v2.runner import Task, TaskCfg, run_task
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.liveplot import LivePlotter1D
from zcu_tools.program import SweepCfg
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
from zcu_tools.program.v2.modules import BathResetCfg
from zcu_tools.utils.datasaver import load_data, save_data
from zcu_tools.utils.process import rotate2real

# (lens, signals)
LengthResult: TypeAlias = tuple[NDArray[np.float64], NDArray[np.complex128]]


def bathreset_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return rotate2real(signals).real


class LengthModuleCfg(TypedDict, closed=True):
    reset: NotRequired[ResetCfg]
    init_pulse: NotRequired[PulseCfg]
    tested_reset: BathResetCfg
    readout: ReadoutCfg


class LengthCfg(ModularProgramCfg, TaskCfg):
    modules: LengthModuleCfg
    sweep: dict[str, SweepCfg]


class LengthExp(AbsExperiment[LengthResult, LengthCfg]):
    def run(
        self, soc, soccfg, cfg: dict[str, Any], detune: float = 0.0
    ) -> LengthResult:
        cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")
        _cfg = check_type(deepcopy(cfg), LengthCfg)
        modules = _cfg["modules"]

        # TODO: align qubit pulse length with cavity pulse length
        lengths = sweep2array(
            _cfg["sweep"]["length"],
            "time",
            {
                "soccfg": soccfg,
                "gen_ch": modules["tested_reset"]["qubit_tone_cfg"]["ch"],
            },
        )

        length_param = sweep2param("length", _cfg["sweep"]["length"])
        phase_param = 360 * detune * length_param
        Reset.set_param(modules["tested_reset"], "length", length_param)
        Reset.set_param(modules["tested_reset"], "pi2_phase", phase_param)

        with LivePlotter1D("Length (us)", "Signal (a.u.)") as viewer:
            signals = run_task(
                task=Task(
                    measure_fn=lambda ctx, update_hook: (
                        (modules := ctx.cfg["modules"])
                        and (
                            ModularProgramV2(
                                soccfg,
                                ctx.cfg,
                                sweep=[("length", ctx.cfg["sweep"]["length"])],
                                modules=[
                                    Reset("reset", modules.get("reset")),
                                    Pulse("init_pulse", modules.get("init_pulse")),
                                    Reset("tested_reset", modules["tested_reset"]),
                                    Readout("readout", modules["readout"]),
                                ],
                            ).acquire(soc, progress=False, callback=update_hook)
                        )
                    ),
                    result_shape=(len(lengths),),
                ),
                init_cfg=_cfg,
                on_update=lambda ctx: viewer.update(
                    lengths, bathreset_signal2real(ctx.root_data)
                ),
            )

        # Cache results
        self.last_cfg = _cfg
        self.last_result = (lengths, signals)

        return lengths, signals

    def analyze(self, result: Optional[LengthResult] = None) -> Figure:
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
        result: Optional[LengthResult] = None,
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

    def load(self, filepath: str, **kwargs) -> LengthResult:
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
