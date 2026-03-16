from __future__ import annotations

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from typeguard import check_type
from typing_extensions import (
    Any,
    Dict,
    NotRequired,
    Optional,
    Tuple,
    TypeAlias,
    TypedDict,
    cast,
)

from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.utils import format_sweep1D, make_ge_sweep, sweep2array
from zcu_tools.experiment.v2.runner import Task, TaskCfg, run_task
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
from zcu_tools.utils.datasaver import load_data, save_data
from zcu_tools.utils.process import rotate2real

# (pdrs, signals_2d)  # signals shape: (2, len(pdrs)) for [w/o reset, w/ reset]
RabiCheckResult: TypeAlias = Tuple[NDArray[np.float64], NDArray[np.complex128]]


def reset_rabi_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return rotate2real(signals).real


class RabiCheckModuleCfg(TypedDict, closed=True):
    reset: NotRequired[ResetCfg]
    rabi_pulse: PulseCfg
    tested_reset: ResetCfg
    post_pulse: NotRequired[PulseCfg]
    readout: ReadoutCfg


class RabiCheckCfg(ModularProgramCfg, TaskCfg):
    modules: RabiCheckModuleCfg
    sweep: Dict[str, SweepCfg]


class RabiCheckExp(AbsExperiment[RabiCheckResult, RabiCheckCfg]):
    def run(self, soc, soccfg, cfg: Dict[str, Any]) -> RabiCheckResult:
        cfg["sweep"] = format_sweep1D(cfg["sweep"], "gain")
        _cfg = check_type(deepcopy(cfg), RabiCheckCfg)

        _cfg["sweep"] = {
            "w/o_reset": make_ge_sweep(),
            "gain": _cfg["sweep"]["gain"],
        }

        pdrs = sweep2array(_cfg["sweep"]["gain"])  # predicted amplitudes

        # Attach gain sweep to initialization pulse
        modules = _cfg["modules"]
        Pulse.set_param(
            modules["rabi_pulse"], "gain", sweep2param("gain", _cfg["sweep"]["gain"])
        )
        Reset.set_param(
            modules["tested_reset"],
            "on/off",
            sweep2param("w/o_reset", _cfg["sweep"]["w/o_reset"]),
        )

        def measure_fn(ctx, update_hook):
            nonlocal pdrs
            modules = ctx.cfg["modules"]
            prog = ModularProgramV2(
                soccfg,
                ctx.cfg,
                modules=[
                    Reset("reset", modules.get("reset")),
                    Pulse("rabi_pulse", modules["rabi_pulse"]),
                    Reset("tested_reset", modules["tested_reset"]),
                    Pulse("post_pulse", modules.get("post_pulse")),
                    Readout("readout", modules["readout"]),
                ],
            )
            _pdrs = prog.get_pulse_param("rabi_pulse", "gain", as_array=True)
            pdrs = cast(NDArray[np.float64], _pdrs)
            return prog.acquire(soc, progress=False, callback=update_hook)

        with LivePlotter1D(
            "Pulse gain", "Amplitude", segment_kwargs=dict(num_lines=2)
        ) as viewer:
            signals = run_task(
                task=Task(
                    measure_fn=measure_fn,
                    result_shape=(2, len(pdrs)),
                ),
                init_cfg=_cfg,
                on_update=lambda ctx: viewer.update(
                    pdrs, reset_rabi_signal2real(ctx.root_data)
                ),
            )

        # Cache results
        self.last_cfg = _cfg
        self.last_result = (pdrs, signals)

        return pdrs, signals

    def analyze(self, result: Optional[RabiCheckResult] = None) -> Figure:
        """Analyze reset rabi check results. (No specific analysis implemented)"""
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        pdrs, signals = result
        real_signals = reset_rabi_signal2real(signals)

        wo_signals, w_signals = real_signals

        fig, ax = plt.subplots(figsize=config.figsize)

        ax.plot(pdrs, w_signals, label="With Reset", marker=".")
        ax.plot(pdrs, wo_signals, label="Without Reset", marker=".")
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

        pdrs, signals = result

        save_data(
            filepath=filepath,
            x_info={"name": "Amplitude", "unit": "a.u.", "values": pdrs},
            y_info={"name": "Reset", "unit": "None", "values": np.array([0, 1])},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals},
            comment=comment,
            tag=tag,
            **kwargs,
        )

    def load(self, filepath: str, **kwargs) -> RabiCheckResult:
        signals, pdrs, y_values = load_data(filepath, **kwargs)
        assert pdrs is not None and y_values is not None
        assert len(pdrs.shape) == 1 and len(y_values.shape) == 1
        assert signals.shape == (len(y_values), len(pdrs))

        pdrs = pdrs.astype(np.float64)
        signals = signals.astype(np.complex128)

        self.last_cfg = None
        self.last_result = (pdrs, signals)

        return pdrs, signals
