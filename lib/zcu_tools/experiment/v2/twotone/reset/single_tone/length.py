from __future__ import annotations

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from typeguard import check_type
from typing_extensions import (
    Any,
    Callable,
    NotRequired,
    Optional,
    TypeAlias,
    TypedDict,
    cast,
)

from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.utils import format_sweep1D, setup_devices
from zcu_tools.experiment.v2.runner import Task, TaskCfg, TaskState, run_task
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.liveplot import LivePlot1D
from zcu_tools.program import SweepCfg
from zcu_tools.program.v2 import (
    ModularProgramCfg,
    ModularProgramV2,
    Pulse,
    PulseCfg,
    PulseReset,
    Readout,
    ReadoutCfg,
    Reset,
    ResetCfg,
    sweep2param,
)
from zcu_tools.program.v2.modules import PulseResetCfg
from zcu_tools.utils.datasaver import load_data, save_data
from zcu_tools.utils.process import rotate2real

# (lens, signals)
LengthResult: TypeAlias = tuple[NDArray[np.float64], NDArray[np.complex128]]


def reset_length_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return rotate2real(signals).real


class LengthModuleCfg(TypedDict, closed=True):
    reset: NotRequired[ResetCfg]
    init_pulse: NotRequired[PulseCfg]
    tested_reset: PulseResetCfg
    readout: ReadoutCfg


class LengthCfg(ModularProgramCfg, TaskCfg):
    modules: LengthModuleCfg
    sweep: dict[str, SweepCfg]


class LengthExp(AbsExperiment[LengthResult, LengthCfg]):
    def run(
        self,
        soc,
        soccfg,
        cfg: dict[str, Any],
        *,
        acquire_kwargs: Optional[dict[str, Any]] = None,
    ) -> LengthResult:
        cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")
        _cfg = check_type(deepcopy(cfg), LengthCfg)
        setup_devices(_cfg, progress=True)
        modules = _cfg["modules"]

        length_sweep = _cfg["sweep"]["length"]

        pulse_cfg = modules["tested_reset"].pulse_cfg

        lengths = sweep2array(
            length_sweep, "time", dict(soccfg=soccfg, gen_ch=pulse_cfg.ch)
        )

        def measure_fn(
            ctx: TaskState, update_hook: Optional[Callable]
        ) -> list[NDArray[np.float64]]:
            cfg = cast(LengthCfg, ctx.cfg)
            modules = cfg["modules"]

            length_sweep = cfg["sweep"]["length"]
            length_param = sweep2param("length", length_sweep)

            modules["tested_reset"].set_param("length", length_param)

            return ModularProgramV2(
                soccfg,
                cfg,
                sweep=[("length", length_sweep)],
                modules=[
                    Reset("reset", modules.get("reset")),
                    Pulse("init_pulse", modules.get("init_pulse")),
                    PulseReset("tested_reset", modules["tested_reset"]),
                    Readout("readout", modules["readout"]),
                ],
            ).acquire(
                soc,
                progress=False,
                round_hook=update_hook,
                **(acquire_kwargs or {}),
            )

        with LivePlot1D("Length (us)", "Amplitude") as viewer:
            signals = run_task(
                task=Task(
                    measure_fn=measure_fn,
                    result_shape=(len(lengths),),
                    pbar_n=_cfg["rounds"],
                ),
                init_cfg=_cfg,
                on_update=lambda ctx: viewer.update(
                    lengths, reset_length_signal2real(ctx.root_data)
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

        # Discard NaNs (possible early abort)
        val_mask = ~np.isnan(signals)
        lens = lens[val_mask]
        signals = signals[val_mask]

        real_signals = reset_length_signal2real(signals)

        fig, ax = plt.subplots(figsize=config.figsize)
        ax.plot(lens, real_signals, marker=".")
        ax.set_xlabel("ProbeTime (us)", fontsize=14)
        ax.set_ylabel("Signal (a.u.)", fontsize=14)
        ax.grid(True)
        ax.tick_params(axis="both", which="major", labelsize=12)

        fig.tight_layout()

        return fig

    def save(
        self,
        filepath: str,
        result: Optional[LengthResult] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/reset/single_tone/length",
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
