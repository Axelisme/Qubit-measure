from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from matplotlib.axes import Axes
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
)
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.utils import set_freq_in_dev_cfg, setup_devices
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
class FreqResult:
    freqs: NDArray[np.float64]
    signals: NDArray[np.float64]


@dataclass(frozen=True)
class FreqAnalysis:
    best_freq: float


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

    def run(self, cfg: FreqCfg, *, context: RunContext) -> FreqResult:
        cfg = deepcopy(cfg)
        soc, soccfg = context.soc, context.soccfg
        jpa_freqs = sweep2array(cfg.sweep.jpa_freq, allow_array=True)
        np.random.shuffle(jpa_freqs[1:-1])  # randomize permutation

        def configure_axes(ax: Axes) -> None:
            ax.lines[0].set_linestyle("None")
            ax.lines[0].set_marker("o")

        viewer = context.plots.liveplot_1d(
            "measurement",
            "JPA Frequency (MHz)",
            "Signal Difference",
            configure_axes=configure_axes,
        )
        signals_buffer = SignalBuffer(
            (len(jpa_freqs),),
            dtype=np.float64,
            on_update=lambda data: viewer.update(jpa_freqs, np.abs(data)),
        )
        with Schedule(cfg, signals_buffer, stop=context.cancel_signal) as sched:
            for jpa_freq, step in sched.scan("JPA Frequency", jpa_freqs.tolist()):
                if step.cfg.dev is not None:
                    set_freq_in_dev_cfg(
                        step.cfg.dev,
                        jpa_freq * 1e6,
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

        return FreqResult(freqs=jpa_freqs, signals=signals)

    def analyze(
        self, source: RunRecord[FreqCfg, FreqResult], options: None, *, plots: Plots
    ) -> FreqAnalysis:
        del options
        result = source.result

        jpa_freqs = result.freqs
        signals = result.signals

        real_signals = np.abs(signals)

        max_idx = np.nanargmax(real_signals)
        best_jpa_freq = jpa_freqs[max_idx]

        fig, ax = plots.subplots("fit", figsize=config.figsize)
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

        return FreqAnalysis(best_freq=float(best_jpa_freq))
