from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from pydantic import Field

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.device import DeviceInfo
from zcu_tools.experiment import (
    MHZ_TO_HZ,
    AxesSpec,
    Axis,
    PersistableExperiment,
    ZSpec,
    config,
    record_result,
    retrieve_result,
)
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import (
    set_output_in_dev_cfg,
    setup_devices,
)
from zcu_tools.experiment.v2.runner import MeasureSession, TaskState
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.liveplot import LivePlot1D
from zcu_tools.program.v2 import (
    ModularProgramV2,
    ProgramV2Cfg,
    PulseReadout,
    PulseReadoutCfg,
    Reset,
    ResetCfg,
    SweepCfg,
    sweep2param,
)


@dataclass(frozen=True)
class CheckResult:
    outputs: NDArray[np.float64]
    freqs: NDArray[np.float64]
    signals: NDArray[np.complex128]
    cfg_snapshot: CheckCfg | None = None


def check_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return np.abs(signals)


class CheckModuleCfg(ConfigBase):
    reset: ResetCfg | None = None
    readout: PulseReadoutCfg


class CheckSweepCfg(ConfigBase):
    freq: SweepCfg


class CheckCfg(ProgramV2Cfg, ExpCfgModel):
    modules: CheckModuleCfg
    # Field(...) makes dev required in this subclass, overriding the Optional
    # default from ExpCfgModel — intentional Pydantic pattern (type: ignore[override]).
    dev: Mapping[str, DeviceInfo] = Field(...)  # type: ignore[override]
    sweep: CheckSweepCfg


class CheckExp(PersistableExperiment[CheckResult, CheckCfg]):
    OUTPUT_MAP = {0: "off", 1: "on"}

    # freqs stored as Hz on disk -> scale=MHZ_TO_HZ; outputs are int JPA labels.
    AXES_SPEC = AxesSpec(
        axes=(
            Axis("freqs", "Frequency", "Hz", scale=MHZ_TO_HZ),
            Axis("outputs", "JPA Output", "a.u.", dtype=np.int_),
        ),
        z=ZSpec("signals", "Signal", "a.u."),
        result_type=CheckResult,
        cfg_type=CheckCfg,
        tag="jpa/check",
    )

    @record_result
    def run(self, soc, soccfg, cfg: CheckCfg) -> CheckResult:
        cfg = deepcopy(cfg)
        setup_devices(cfg, progress=True)
        modules = cfg.modules

        outputs = np.array([0, 1])
        freqs = sweep2array(
            cfg.sweep.freq,
            "freq",
            {"soccfg": soccfg, "gen_ch": modules.readout.pulse_cfg.ch},
        )

        def measure_fn(
            ctx: TaskState[NDArray[np.complex128], Any, CheckCfg],
            update_hook: Callable[[int, list[NDArray[np.float64]]], None] | None,
        ) -> list[NDArray[np.float64]]:
            cfg = ctx.cfg
            setup_devices(cfg, progress=False)
            modules = cfg.modules

            freq_sweep = cfg.sweep.freq
            freq_param = sweep2param("freq", freq_sweep)
            modules.readout.set_param("freq", freq_param)

            return ModularProgramV2(
                soccfg,
                cfg,
                modules=[
                    Reset("reset", modules.reset),
                    PulseReadout("readout", modules.readout),
                ],
                sweep=[("freq", freq_sweep)],
            ).acquire(
                soc,
                progress=False,
                round_hook=update_hook,
                stop_checkers=[ctx.is_stop],
            )

        with LivePlot1D(
            "Frequency (MHz)", "Magnitude", segment_kwargs=dict(num_lines=2)
        ) as viewer:
            with MeasureSession(cfg) as run:
                signals_buffer = run.buffer(
                    (len(outputs), len(freqs)),
                    dtype=np.complex128,
                    on_update=lambda data: viewer.update(
                        freqs, check_signal2real(data)
                    ),
                )
                for step in run.scan("JPA on/off", outputs.tolist()):
                    if step.cfg.dev is not None:
                        set_output_in_dev_cfg(
                            step.cfg.dev,
                            self.OUTPUT_MAP[step.value],  # type: ignore[index]
                            label="jpa_rf_dev",
                        )
                    signals_buffer[step].measure(measure_fn, pbar_n=step.cfg.rounds)
                signals = signals_buffer.array

        return CheckResult(
            outputs=outputs, freqs=freqs, signals=signals, cfg_snapshot=cfg
        )

    @retrieve_result
    def analyze(self, result: CheckResult | None = None) -> Figure:
        assert result is not None, "no result found"

        outputs = result.outputs
        freqs = result.freqs
        signals2D = result.signals
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
