from __future__ import annotations

from collections.abc import Mapping
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
    set_power_in_dev_cfg,
    setup_devices,
)
from zcu_tools.experiment.v2.runner import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils import snr_as_signal, sweep2array
from zcu_tools.experiment.v2.utils.tracker import MomentTracker
from zcu_tools.liveplot import LivePlotScatter
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
    cfg_snapshot: PowerCfg | None = None


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

    @record_result
    def run(self, soc, soccfg, cfg: PowerCfg) -> PowerResult:
        orig_cfg = deepcopy(cfg)
        jpa_powers = sweep2array(cfg.sweep.jpa_power, allow_array=True)
        np.random.shuffle(jpa_powers[1:-1])

        with LivePlotScatter("Power (dBm)", "Signal Difference") as viewer:
            signals_buffer = SignalBuffer(
                (len(jpa_powers),),
                dtype=np.float64,
                on_update=lambda data: viewer.update(jpa_powers, np.abs(data)),
            )
            with Schedule(cfg, signals_buffer) as sched:
                for jpa_power, step in sched.scan("power (dBm)", jpa_powers.tolist()):
                    set_power_in_dev_cfg(
                        step.cfg.dev,
                        jpa_power,
                        label="jpa_rf_dev",
                    )
                    setup_devices(step.cfg, progress=False)
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
                            raw2signal_fn=lambda raw: snr_as_signal(
                                [tracker],
                                ge_axis=0,
                                skew_penalty=sched.cfg.skew_penalty,
                            ),
                            trackers=[tracker],
                        )
                    )
                signals = signals_buffer.array

        return PowerResult(powers=jpa_powers, signals=signals, cfg_snapshot=orig_cfg)

    @retrieve_result
    def analyze(self, result: PowerResult | None = None) -> tuple[float, Figure]:
        assert result is not None, "no result found"

        jpa_powers = result.powers
        signals = result.signals
        snrs = np.abs(signals)

        max_idx = np.nanargmax(snrs)
        best_jpa_power = jpa_powers[max_idx]

        fig, ax = plt.subplots(figsize=config.figsize)
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

        return float(best_jpa_power), fig
