from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import ClassVar, Literal

import numpy as np
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
)
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.utils import (
    set_output_in_dev_cfg,
    setup_devices,
)
from zcu_tools.experiment.v2.runtime import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
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
    OUTPUT_MAP: ClassVar[dict[int, Literal["off", "on"]]] = {0: "off", 1: "on"}

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

    def run(self, cfg: CheckCfg, *, context: RunContext) -> CheckResult:
        cfg = deepcopy(cfg)
        soc, soccfg = context.soc, context.soccfg
        setup_devices(
            cfg,
            context.devices,
            progress=True,
            cancel_signal=context.cancel_signal.event,
        )
        modules = cfg.modules

        outputs = np.array([0, 1])
        freqs = sweep2array(
            cfg.sweep.freq,
            "freq",
            {"soccfg": soccfg, "gen_ch": modules.readout.pulse_cfg.ch},
        )

        viewer = context.plots.liveplot_1d(
            "measurement", "Frequency (MHz)", "Magnitude", num_lines=2
        )
        signals_buffer = SignalBuffer(
            (len(outputs), len(freqs)),
            on_update=lambda data: viewer.update(freqs, check_signal2real(data)),
        )
        with Schedule(cfg, signals_buffer, stop=context.cancel_signal) as sched:
            for output, step in sched.scan("JPA on/off", outputs.tolist()):
                set_output_in_dev_cfg(
                    step.cfg.dev,
                    self.OUTPUT_MAP[output],
                    label="jpa_rf_dev",
                )
                setup_devices(
                    step.cfg,
                    context.devices,
                    progress=False,
                    cancel_signal=context.cancel_signal.event,
                )
                modules = step.cfg.modules
                modules.readout.set_param(
                    "freq", sweep2param("freq", step.cfg.sweep.freq)
                )
                _ = (
                    step.prog_builder(soc, soccfg)
                    .add(
                        Reset("reset", modules.reset),
                        PulseReadout("readout", modules.readout),
                    )
                    .declare_sweep("freq", step.cfg.sweep.freq)
                    .build_and_acquire()
                )
            signals = signals_buffer.array

        return CheckResult(outputs=outputs, freqs=freqs, signals=signals)

    def analyze(
        self, source: RunRecord[CheckCfg, CheckResult], options: None, *, plots: Plots
    ) -> None:
        del options
        result = source.result

        outputs = result.outputs
        freqs = result.freqs
        signals2D = result.signals
        real_signals = check_signal2real(signals2D)

        _, ax = plots.subplots("fit", figsize=config.figsize)
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
