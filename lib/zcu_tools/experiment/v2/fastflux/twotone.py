from __future__ import annotations

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from typeguard import check_type
from typing_extensions import Any, NotRequired, Optional, TypeAlias, TypedDict

from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.v2.runner import Task, TaskCfg, run_task
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.liveplot import LivePlotter2D
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
from zcu_tools.utils.datasaver import load_data, save_data
from zcu_tools.utils.process import rotate2real

# (gains, freqs, signals2D)
TwoToneResult: TypeAlias = tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.complex128]
]


def twotone_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return rotate2real(signals).real


class TwoToneModuleCfg(TypedDict, closed=True):
    reset: NotRequired[ResetCfg]
    flux_pulse: PulseCfg
    qub_pulse: PulseCfg
    readout: ReadoutCfg


class TwotoneCfg(ModularProgramCfg, TaskCfg):
    modules: TwoToneModuleCfg
    sweep: dict[str, SweepCfg]


class TwoToneExp(AbsExperiment[TwoToneResult, TwotoneCfg]):
    def run(self, soc, soccfg, cfg: dict[str, Any]) -> TwoToneResult:
        _cfg = check_type(deepcopy(cfg), TwotoneCfg)
        modules = _cfg["modules"]

        # uniform in square space
        gains = sweep2array(
            _cfg["sweep"]["gain"],
            "gain",
            {"soccfg": soccfg, "gen_ch": modules["flux_pulse"]["ch"]},
        )
        freqs = sweep2array(
            _cfg["sweep"]["freq"],
            "freq",
            {"soccfg": soccfg, "gen_ch": modules["qub_pulse"]["ch"]},
        )

        gain_param = sweep2param("gain", _cfg["sweep"]["gain"])
        freq_param = sweep2param("freq", _cfg["sweep"]["freq"])
        Pulse.set_param(modules["flux_pulse"], "gain", gain_param)
        Pulse.set_param(modules["qub_pulse"], "freq", freq_param)

        with LivePlotter2D("Flux Pulse Gain (a.u.)", "Frequency (MHz)") as viewer:
            signals = run_task(
                task=Task(
                    measure_fn=lambda ctx, update_hook: (
                        (modules := ctx.cfg["modules"])
                        and ModularProgramV2(
                            soccfg,
                            ctx.cfg,
                            modules=[
                                Reset("reset", modules.get("reset")),
                                Pulse(
                                    "flux_pulse",
                                    modules["flux_pulse"],
                                    block_mode=False,
                                ),
                                Pulse("qub_pulse", modules["qub_pulse"]),
                                Readout("readout", modules["readout"]),
                            ],
                        ).acquire(soc, progress=False, callback=update_hook)
                    ),
                    result_shape=(len(gains), len(freqs)),
                ),
                init_cfg=_cfg,
                on_update=lambda ctx: viewer.update(
                    gains, freqs, twotone_signal2real(ctx.root_data)
                ),
            )

        # Cache results
        self.last_cfg = _cfg
        self.last_result = (gains, freqs, signals)

        return gains, freqs, signals

    def analyze(self, result: Optional[TwoToneResult] = None) -> Figure:
        if result is None:
            result = self.last_result
        assert result is not None, "No result found"

        gains, freqs, signals2D = result

        real_signals = twotone_signal2real(signals2D)

        fig, ax = plt.subplots()

        ax.imshow(
            real_signals.T,
            extent=[gains[0], gains[-1], freqs[0], freqs[-1]],
            aspect="auto",
            origin="lower",
            interpolation="none",
            cmap="RdBu_r",
        )
        ax.set_xlabel("Flux Pulse Gain (a.u.)")
        ax.set_ylabel("Frequency (MHz)")

        fig.tight_layout()

        return fig

    def save(
        self,
        filepath: str,
        result: Optional[TwoToneResult] = None,
        comment: Optional[str] = None,
        tag: str = "fastflux/twotone",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "No result found"

        gains, freqs, signals2D = result

        save_data(
            filepath=filepath,
            x_info={"name": "Flux Pulse Gain", "unit": "a.u.", "values": gains},
            y_info={"name": "Frequency", "unit": "Hz", "values": freqs * 1e6},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals2D.T},
            comment=comment,
            tag=tag,
            **kwargs,
        )

    def load(self, filepath: str, **kwargs) -> TwoToneResult:
        signals2D, gains, freqs = load_data(filepath, **kwargs)
        assert freqs is not None
        assert len(gains.shape) == 1 and len(freqs.shape) == 1
        assert signals2D.shape == (len(gains), len(freqs))

        freqs = freqs * 1e-6  # Hz -> MHz

        gains = gains.astype(np.float64)
        freqs = freqs.astype(np.float64)
        signals2D = signals2D.astype(np.complex128)

        self.last_cfg = None
        self.last_result = (gains, freqs, signals2D)

        return gains, freqs, signals2D
