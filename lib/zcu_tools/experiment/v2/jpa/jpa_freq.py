from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from pydantic import Field

from zcu_tools.cfg_model import ConfigBase
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
from zcu_tools.experiment.utils import set_freq_in_dev_cfg, setup_devices
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
class FreqResult:
    freqs: NDArray[np.float64]
    signals: NDArray[np.float64]
    cfg_snapshot: FreqCfg | None = None


class FreqModuleCfg(ConfigBase):
    reset: ResetCfg | None = None
    pi_pulse: PulseCfg
    readout: ReadoutCfg


class FreqSweepCfg(ConfigBase):
    jpa_freq: SweepCfg


class FreqCfg(ProgramV2Cfg, ExpCfgModel):
    modules: FreqModuleCfg
    sweep: FreqSweepCfg
    skew_penalty: float = Field(default=0.0, ge=0.0)


class FreqExp(PersistableExperiment[FreqResult, FreqCfg]):
    # jpa_freq stored as Hz on disk -> scale=MHZ_TO_HZ; signals are SNR (float64)
    AXES_SPEC = AxesSpec(
        axes=(Axis("freqs", "JPA Frequency", "Hz", scale=MHZ_TO_HZ),),
        z=ZSpec("signals", "Signal", "a.u.", dtype=np.float64),
        result_type=FreqResult,
        cfg_type=FreqCfg,
        tag="jpa/freq",
    )

    @record_result
    def run(self, soc, soccfg, cfg: FreqCfg) -> FreqResult:
        orig_cfg = deepcopy(cfg)
        jpa_freqs = sweep2array(cfg.sweep.jpa_freq, allow_array=True)
        np.random.shuffle(jpa_freqs[1:-1])  # randomize permutation

        with LivePlotScatter("JPA Frequency (MHz)", "Signal Difference") as viewer:
            signals_buffer = SignalBuffer(
                (len(jpa_freqs),),
                dtype=np.float64,
                on_update=lambda data: viewer.update(jpa_freqs, np.abs(data)),
            )
            with Schedule(cfg, signals_buffer) as sched:
                for jpa_freq, step in sched.scan("JPA Frequency", jpa_freqs.tolist()):
                    if step.cfg.dev is not None:
                        set_freq_in_dev_cfg(
                            step.cfg.dev,
                            jpa_freq * 1e6,
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

        return FreqResult(freqs=jpa_freqs, signals=signals, cfg_snapshot=orig_cfg)

    @retrieve_result
    def analyze(self, result: FreqResult | None = None) -> tuple[float, Figure]:
        assert result is not None, "no result found"

        jpa_freqs = result.freqs
        signals = result.signals

        real_signals = np.abs(signals)

        max_idx = np.nanargmax(real_signals)
        best_jpa_freq = jpa_freqs[max_idx]

        fig, ax = plt.subplots(figsize=config.figsize)
        ax.scatter(jpa_freqs, real_signals, label="signal difference", s=2)
        ax.axvline(
            best_jpa_freq,
            color="r",
            ls="--",
            label=f"best JPA frequency = {best_jpa_freq:.2g} MHz",
        )
        ax.set_xlabel("JPA Frequency (MHz)")
        ax.set_ylabel("Signal Difference (a.u.)")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()

        return float(best_jpa_freq), fig
