from __future__ import annotations

from copy import deepcopy
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from typing_extensions import NotRequired, TypedDict

from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.utils import format_sweep1D, sweep2array
from zcu_tools.experiment.v2.runner import HardTask, run_task
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
LengthResult = Tuple[NDArray[np.float64], NDArray[np.complex128]]


def reset_length_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return rotate2real(signals).real


class LengthModuleCfg(TypedDict, closed=True):
    reset: NotRequired[ResetCfg]
    init_pulse: NotRequired[PulseCfg]
    tested_reset: ResetCfg
    readout: ReadoutCfg


class LengthCfg(ModularProgramCfg):
    modules: LengthModuleCfg


class LengthExp(AbsExperiment[LengthResult, LengthCfg]):
    def run(self, soc, soccfg, cfg: LengthCfg) -> LengthResult:
        cfg = deepcopy(cfg)  # prevent in-place modification

        # Check that reset pulse is dual pulse type
        modules = cfg["modules"]
        if modules["tested_reset"]["type"] != "two_pulse":
            raise ValueError("This experiment only supports dual-tone reset")

        # Canonicalise sweep section to single-axis form
        assert "sweep" in cfg
        cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")

        lens = sweep2array(cfg["sweep"]["length"])  # predicted pulse lengths

        pulse1_cfg = modules["tested_reset"]["pulse1_cfg"]  # type: ignore
        pulse2_cfg = modules["tested_reset"]["pulse2_cfg"]  # type: ignore

        len_diff = pulse2_cfg["waveform"]["length"] - pulse1_cfg["waveform"]["length"]
        len1_span = sweep2param("length", cfg["sweep"]["length"])

        Pulse.set_param(pulse1_cfg, "length", len1_span)
        Pulse.set_param(pulse2_cfg, "length", len1_span + len_diff)

        with LivePlotter1D("Length (us)", "Amplitude") as viewer:
            signals = run_task(
                task=HardTask(
                    measure_fn=lambda ctx, update_hook: ModularProgramV2(
                        soccfg,
                        ctx.cfg,
                        modules=[
                            Reset(
                                "reset",
                                (modules := ctx.cfg["modules"]).get(
                                    "reset", {"type": "none"}
                                ),
                            ),
                            Pulse("init_pulse", modules.get("init_pulse")),
                            Reset("tested_reset", modules["tested_reset"]),
                            Readout("readout", modules["readout"]),
                        ],
                    ).acquire(soc, progress=False, callback=update_hook),
                    result_shape=(len(lens),),
                ),
                init_cfg=cfg,
                update_hook=lambda ctx: viewer.update(
                    lens, reset_length_signal2real(ctx.data)
                ),
            )

        # Cache results
        self.last_cfg = cfg
        self.last_result = (lens, signals)

        return lens, signals

    def analyze(self, result: Optional[LengthResult] = None) -> Figure:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        lens, signals = result

        # Discard NaNs (possible early abort)
        val_mask = ~np.isnan(signals)
        lens = lens[val_mask]
        signals = signals[val_mask]

        real_signals = reset_length_signal2real(signals)

        fig, ax = plt.subplots(figsize=config.figsize)
        assert isinstance(fig, Figure)

        ax.plot(lens, real_signals, marker=".")
        ax.set_xlabel("ProbeTime (us)", fontsize=14)
        ax.set_ylabel("Signal (a.u.)", fontsize=14)
        ax.grid(True)
        ax.tick_params(axis="both", which="major", labelsize=12)
        plt.show()

        fig.tight_layout()

        return fig

    def save(
        self,
        filepath: str,
        result: Optional[LengthResult] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/reset/dual_tone/length",
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
