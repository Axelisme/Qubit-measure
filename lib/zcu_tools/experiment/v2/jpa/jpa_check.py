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
    Mapping,
    NotRequired,
    Optional,
    TypeAlias,
    TypedDict,
    cast,
)

from zcu_tools.device import DeviceInfo
from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.utils import format_sweep1D, set_output_in_dev_cfg
from zcu_tools.experiment.v2.runner import Task, TaskCfg, TaskState, run_task
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.liveplot import LivePlotter1D
from zcu_tools.program import SweepCfg
from zcu_tools.program.v2 import (
    ModularProgramCfg,
    ModularProgramV2,
    PulseReadout,
    PulseReadoutCfg,
    Reset,
    ResetCfg,
    sweep2param,
)
from zcu_tools.utils.datasaver import load_data, save_data

CheckResult: TypeAlias = tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.complex128]
]


def check_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return np.abs(signals)

class CheckModuleCfg(TypedDict, closed=True):
    reset: NotRequired[ResetCfg]
    readout: PulseReadoutCfg

class CheckCfg(ModularProgramCfg, TaskCfg):
    modules: CheckModuleCfg
    dev: Mapping[str, DeviceInfo]
    sweep: dict[str, SweepCfg]


class CheckExp(AbsExperiment[CheckResult, CheckCfg]):
    OUTPUT_MAP = {0: "off", 1: "on"}

    def run(self, soc, soccfg, cfg: dict[str, Any]) -> CheckResult:
        cfg["sweep"] = format_sweep1D(cfg["sweep"], "freq")
        _cfg = check_type(deepcopy(cfg), CheckCfg)
        modules = _cfg["modules"]

        outputs = np.array([0, 1])
        freqs = sweep2array(
            _cfg["sweep"]["freq"],
            "freq",
            {"soccfg": soccfg, "gen_ch": modules["readout"]["pulse_cfg"]["ch"]},
        )

        def measure_fn(
            ctx: TaskState[NDArray[np.complex128], Any],
            update_hook: Optional[Callable[[int, list[NDArray[np.float64]]], None]],
        ) -> list[NDArray[np.float64]]:
            cfg: CheckCfg = cast(CheckCfg, ctx.cfg)
            modules = cfg["modules"]

            freq_sweep = cfg["sweep"]["freq"]
            freq_param = sweep2param("freq", freq_sweep)
            PulseReadout.set_param(modules["readout"], "freq", freq_param)

            return ModularProgramV2(
                soccfg,
                cfg,
                modules=[
                    Reset("reset", modules.get("reset")),
                    PulseReadout("readout", modules["readout"]),
                ],
                sweep=[("freq", freq_sweep)],
            ).acquire(ctx.env["soc"], progress=False, callback=update_hook)

        with LivePlotter1D(
            "Frequency (MHz)", "Magnitude", segment_kwargs=dict(num_lines=2)
        ) as viewer:
            signals = run_task(
                task=Task(measure_fn=measure_fn, result_shape=(len(freqs),)).scan(
                    "JPA on/off",
                    outputs.tolist(),
                    before_each=lambda i, ctx, output: set_output_in_dev_cfg(
                        ctx.cfg["dev"],
                        self.OUTPUT_MAP[output],  # type: ignore
                        label="jpa_rf_dev",
                    ),
                ),
                init_cfg=_cfg,
                on_update=lambda ctx: viewer.update(
                    freqs, check_signal2real(np.asarray(ctx.root_data))
                ),
            )
            signals = np.asarray(signals)

        # record last cfg and result
        self.last_cfg = _cfg
        self.last_result = (outputs, freqs, signals)

        return outputs, freqs, signals

    def analyze(self, result: Optional[CheckResult] = None) -> Figure:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        outputs, freqs, signals2D = result

        real_signals = check_signal2real(signals2D)

        fig, ax = plt.subplots(figsize=config.figsize)
        for i, output in enumerate(outputs):
            ax.plot(
                freqs,
                real_signals[i, :],
                label=f"JPA {self.OUTPUT_MAP[output]}",
                marker="o",
                markersize=4,
                linestyle="-",
            )

        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Signal Magnitude (a.u.)")
        ax.legend()
        ax.grid(True)

        return fig

    def save(
        self,
        filepath: str,
        result: Optional[CheckResult] = None,
        comment: Optional[str] = None,
        tag: str = "jpa/check",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        outputs, freqs, signals2D = result

        save_data(
            filepath=filepath,
            x_info={"name": "Frequency", "unit": "Hz", "values": freqs * 1e6},
            y_info={"name": "JPA Output", "unit": "a.u.", "values": outputs},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals2D},
            comment=comment,
            tag=tag,
            **kwargs,
        )

    def load(self, filepath: str, **kwargs) -> CheckResult:
        signals2D, freqs, outputs = load_data(filepath, **kwargs)
        assert freqs is not None and outputs is not None
        assert len(freqs.shape) == 1 and len(outputs.shape) == 1
        assert signals2D.shape == (len(outputs), len(freqs))

        freqs = freqs * 1e-6  # Hz -> MHz

        outputs = outputs.astype(np.float64)
        freqs = freqs.astype(np.float64)
        signals2D = signals2D.astype(np.complex128)

        self.last_cfg = None
        self.last_result = (outputs, freqs, signals2D)

        return outputs, freqs, signals2D
