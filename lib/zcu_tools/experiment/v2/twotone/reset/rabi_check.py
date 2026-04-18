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
    Branch,
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

# (gains, signals_2d)  # signals shape: (2, len(gains)) for [w/o reset, w/ reset]
RabiCheckResult: TypeAlias = tuple[NDArray[np.float64], NDArray[np.complex128]]


def reset_rabi_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return rotate2real(signals).real


class RabiCheckModuleCfg(TypedDict, closed=True):
    reset: NotRequired[ResetCfg]
    rabi_pulse: PulseCfg
    tested_reset: ResetCfg
    pi_pulse: PulseCfg
    readout: ReadoutCfg


class RabiCheckCfg(ModularProgramCfg, TaskCfg):
    modules: RabiCheckModuleCfg
    sweep: dict[str, SweepCfg]


class RabiCheckExp(AbsExperiment[RabiCheckResult, RabiCheckCfg]):
    def run(
        self,
        soc,
        soccfg,
        cfg: dict[str, Any],
        *,
        acquire_kwargs: Optional[dict[str, Any]] = None,
    ) -> RabiCheckResult:
        cfg["sweep"] = format_sweep1D(cfg["sweep"], "gain")
        _cfg = check_type(deepcopy(cfg), RabiCheckCfg)
        setup_devices(_cfg, progress=True)
        modules = _cfg["modules"]

        gains = sweep2array(
            _cfg["sweep"]["gain"],
            "gain",
            {"soccfg": soccfg, "gen_ch": modules["rabi_pulse"].ch},
        )

        def measure_fn(
            ctx: TaskState, update_hook: Optional[Callable]
        ) -> list[NDArray[np.float64]]:
            cfg = cast(RabiCheckCfg, ctx.cfg)
            modules = cfg["modules"]

            # Attach gain sweep to initialization pulse
            gain_param = sweep2param("gain", _cfg["sweep"]["gain"])
            modules["rabi_pulse"].set_param("gain", gain_param)

            return ModularProgramV2(
                soccfg,
                cfg,
                sweep=[
                    ("w/o_reset", 3),
                    ("gain", cfg["sweep"]["gain"]),
                ],
                modules=[
                    Reset("reset", modules.get("reset")),
                    Pulse("rabi_pulse", modules["rabi_pulse"]),
                    Branch(
                        "w/o_reset",
                        [],
                        Reset("tested_reset_1", modules["tested_reset"]),
                        [
                            Reset("tested_reset_2", modules["tested_reset"]),
                            Pulse("pi_pulse", modules["pi_pulse"]),
                        ],
                    ),
                    Readout("readout", modules["readout"]),
                ],
            ).acquire(
                soc,
                progress=False,
                callback=update_hook,
                **(acquire_kwargs or {}),
            )

        with LivePlot1D(
            "Pulse gain", "Amplitude", segment_kwargs=dict(num_lines=3)
        ) as viewer:
            signals = run_task(
                task=Task(
                    measure_fn=measure_fn,
                    result_shape=(3, len(gains)),
                    pbar_n=_cfg["rounds"],
                ),
                init_cfg=_cfg,
                on_update=lambda ctx: viewer.update(
                    gains, reset_rabi_signal2real(ctx.root_data)
                ),
            )

        # Cache results
        self.last_cfg = _cfg
        self.last_result = (gains, signals)

        return gains, signals

    def analyze(self, result: Optional[RabiCheckResult] = None) -> Figure:
        """Analyze reset rabi check results. (No specific analysis implemented)"""
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        gains, signals = result
        real_signals = reset_rabi_signal2real(signals)

        wo_signals, w_signals, wp_signals = real_signals

        fig, ax = plt.subplots(figsize=config.figsize)

        ax.plot(gains, wo_signals, label="Without Reset", marker=".")
        ax.plot(gains, w_signals, label="With Reset", marker=".")
        ax.plot(gains, wp_signals, label="  + Pi Pulse", marker=".")
        ax.legend()
        ax.grid(True)

        return fig

    def save(
        self,
        filepath: str,
        result: Optional[RabiCheckResult] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/reset/rabi_check",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        gains, signals = result

        save_data(
            filepath=filepath,
            x_info={"name": "Amplitude", "unit": "a.u.", "values": gains},
            y_info={"name": "Reset", "unit": "None", "values": np.array([0, 1, 2])},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals},
            comment=comment,
            tag=tag,
            **kwargs,
        )

    def load(self, filepath: str, **kwargs) -> RabiCheckResult:
        signals, gains, y_values = load_data(filepath, **kwargs)
        assert gains is not None and y_values is not None
        assert len(gains.shape) == 1 and len(y_values.shape) == 1
        assert signals.shape == (len(y_values), len(gains))

        gains = gains.astype(np.float64)
        signals = signals.astype(np.complex128)

        self.last_cfg = None
        self.last_result = (gains, signals)

        return gains, signals
