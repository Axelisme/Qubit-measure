from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray
from pydantic import Field

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.device import DeviceInfo
from zcu_tools.experiment import (
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
    set_power_in_dev_cfg,
    setup_devices,
)
from zcu_tools.experiment.v2.runtime import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils import snr_as_signal, sweep2array
from zcu_tools.experiment.v2.utils.tracker import MomentTracker
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    Branch,
    ProgramV2Cfg,
    Pulse,
    PulseCfg,
    Readout,
    ReadoutCfg,
    Reset,
    ResetCfg,
    SweepCfg,
)


@dataclass(frozen=True)
class PowerResult:
    powers: NDArray[np.float64]
    signals: NDArray[np.float64]


@dataclass(frozen=True)
class PowerAnalysis:
    best_power: float


class PowerModuleCfg(ConfigBase):
    reset: ResetCfg | None = None
    pi_pulse: PulseCfg
    readout: ReadoutCfg


class PowerSweepCfg(ConfigBase):
    jpa_power: SweepCfg


class PowerCfg(ProgramV2Cfg, ExpCfgModel):
    modules: PowerModuleCfg
    # Field(...) makes dev required in this subclass, overriding the Optional
    # default from ExpCfgModel — intentional Pydantic pattern (type: ignore[override]).
    dev: Mapping[str, DeviceInfo] = Field(...)  # type: ignore[override]
    sweep: PowerSweepCfg
    skew_penalty: float = Field(default=0.0, ge=0.0)


class PowerExp(PersistableExperiment[PowerResult, PowerCfg]):
    # powers stored in dBm on disk -> scale=IDENTITY (1.0); signals are real -> ZSpec dtype float64
    AXES_SPEC = AxesSpec(
        axes=(Axis("powers", "JPA Power", "dBm"),),
        z=ZSpec("signals", "Signal", "a.u.", dtype=np.float64),
        result_type=PowerResult,
        cfg_type=PowerCfg,
        tag="jpa/power",
    )

    def run(self, cfg: PowerCfg, *, context: RunContext) -> PowerResult:
        cfg = deepcopy(cfg)
        soc, soccfg = context.soc, context.soccfg
        jpa_powers = sweep2array(cfg.sweep.jpa_power, allow_array=True)
        np.random.shuffle(jpa_powers[1:-1])

        def configure_axes(ax: Axes) -> None:
            ax.lines[0].set_linestyle("None")
            ax.lines[0].set_marker("o")

        viewer = context.plots.liveplot_1d(
            "measurement",
            "Power (dBm)",
            "Signal Difference",
            configure_axes=configure_axes,
        )
        signals_buffer = SignalBuffer(
            (len(jpa_powers),),
            dtype=np.float64,
            on_update=lambda data: viewer.update(jpa_powers, np.abs(data)),
        )
        with Schedule(cfg, signals_buffer, stop=context.cancel_signal) as sched:
            for jpa_power, step in sched.scan("power (dBm)", jpa_powers.tolist()):
                set_power_in_dev_cfg(
                    step.cfg.dev,
                    jpa_power,
                    label="jpa_rf_dev",
                )
                setup_devices(
                    step.cfg,
                    context.devices,
                    progress=False,
                    cancel_signal=context.cancel_signal,
                )
                modules = step.cfg.modules
                tracker = MomentTracker()
                _ = (
                    step.prog_builder(soc, soccfg)
                    .add(
                        Reset("reset", modules.reset),
                        Branch("ge", [], Pulse("pi_pulse", modules.pi_pulse)),
                        Readout("readout", modules.readout),
                    )
                    .declare_sweep("ge", 2)
                    .build_and_acquire(
                        raw2signal_fn=lambda raw, tracker=tracker, skew_penalty=step.cfg.skew_penalty: (
                            snr_as_signal(
                                [tracker],
                                ge_axis=1,
                                skew_penalty=skew_penalty,
                            )
                        ),
                        trackers=[tracker],
                    )
                )
            signals = signals_buffer.array

        return PowerResult(powers=jpa_powers, signals=signals)

    def analyze(
        self, source: RunRecord[PowerCfg, PowerResult], options: None, *, plots: Plots
    ) -> PowerAnalysis:
        del options
        result = source.result

        jpa_powers = result.powers
        signals = result.signals
        snrs = np.abs(signals)

        max_idx = np.nanargmax(snrs)
        best_jpa_power = jpa_powers[max_idx]

        fig, ax = plots.subplots("fit", figsize=config.figsize)
        ax.scatter(jpa_powers, snrs, label="signal difference", s=1)
        ax.axvline(
            best_jpa_power,
            color="r",
            ls="--",
            label=f"best JPA power = {best_jpa_power:.2g} dBm",
        )
        ax.set_xlabel("JPA Frequency (MHz)")
        ax.set_ylabel("Signal Difference (a.u.)")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()

        return PowerAnalysis(best_power=float(best_jpa_power))
