from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from zcu_tools.analysis.fitting import fit_decay, fit_decay_fringe
from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment import (
    US_TO_S,
    AxesSpec,
    Axis,
    PersistableExperiment,
    ZSpec,
    config,
)
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.utils import setup_devices
from zcu_tools.experiment.v2.runtime import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    Delay,
    ProgramV2Cfg,
    Pulse,
    PulseCfg,
    Readout,
    ReadoutCfg,
    Reset,
    ResetCfg,
    SweepCfg,
    sweep2param,
)
from zcu_tools.utils.process import rotate2real


@dataclass(frozen=True)
class T2EchoResult:
    times: NDArray[np.float64]
    signals: NDArray[np.complex128]
    true_activate_detune: float | None = None


def t2echo_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return rotate2real(signals).real


class T2EchoModuleCfg(ConfigBase):
    reset: ResetCfg | None = None
    pi2_pulse: PulseCfg
    pi_pulse: PulseCfg
    readout: ReadoutCfg


class T2EchoSweepCfg(ConfigBase):
    length: SweepCfg


class T2EchoCfg(ProgramV2Cfg, ExpCfgModel):
    modules: T2EchoModuleCfg
    sweep: T2EchoSweepCfg
    detune: float = 0.0


@dataclass(frozen=True)
class T2EchoAnalyzeOptions:
    fit_method: Literal["fringe", "decay"] = "decay"
    fit_phase: bool = False


@dataclass(frozen=True)
class T2EchoAnalysis:
    t2e: float
    t2e_err: float
    detune: float
    detune_err: float


class T2EchoExp(PersistableExperiment[T2EchoResult, T2EchoCfg]):
    # times stores us in memory, s on disk -> scale=US_TO_S
    AXES_SPEC = AxesSpec(
        axes=(Axis("times", "Time", "s", scale=US_TO_S),),
        z=ZSpec("signals", "Signal", "a.u."),
        result_type=T2EchoResult,
        cfg_type=T2EchoCfg,
        tag="twotone/ge/t2echo",
    )

    def run(self, cfg: T2EchoCfg, *, context: RunContext) -> T2EchoResult:
        cfg = deepcopy(cfg)
        soc, soccfg = context.soc, context.soccfg
        detune = cfg.detune

        setup_devices(
            cfg,
            context.devices,
            progress=True,
            cancel_signal=context.cancel_signal,
        )

        lengths = sweep2array(
            cfg.sweep.length, "time", {"soccfg": soccfg, "scaler": 0.5}
        )

        # calculate true scanned detune based on rounding lengths
        if detune != 0.0:
            detune_lengths = sweep2array(
                cfg.sweep.length,
                "phase",
                {
                    "soccfg": soccfg,
                    "gen_ch": cfg.modules.pi2_pulse.ch,
                    "scaler": 360 * detune,
                },
            )
            mask = lengths > 0
            detune_ratio = np.mean(detune_lengths[mask] / lengths[mask]).item()
            true_detune = detune * detune_ratio
        else:
            true_detune = 0.0

        viewer = context.plots.liveplot_1d(
            "measurement",
            "Time (us)",
            "Amplitude",
            title=f"T2 Echo (detune={true_detune:.3f}MHz)",
        )
        signals_buffer = SignalBuffer(
            (len(lengths),),
            on_update=lambda data: viewer.update(lengths, t2echo_signal2real(data)),
        )
        with Schedule(cfg, signals_buffer, stop=context.cancel_signal) as sched:
            cfg = sched.cfg
            modules = cfg.modules

            length_sweep = cfg.sweep.length
            length_param = sweep2param("length", length_sweep)
            detune_param = 360 * detune * length_param

            _ = (
                sched.prog_builder(soc, soccfg)
                .add(
                    Reset("reset", modules.reset),
                    Pulse("pi2_pulse1", modules.pi2_pulse),
                    Delay("t2e_delay1", delay=0.5 * length_param),
                    Pulse("pi_pulse", modules.pi_pulse),
                    Delay("t2e_delay2", delay=0.5 * length_param),
                    Pulse(
                        name="pi2_pulse2",
                        cfg=modules.pi2_pulse.with_updates(
                            phase=modules.pi2_pulse.phase + detune_param
                        ),
                    ),
                    Readout("readout", modules.readout),
                )
                .declare_sweep("length", length_sweep)
                .build_and_acquire()
            )

        return T2EchoResult(
            times=lengths,
            signals=signals_buffer.array,
            true_activate_detune=true_detune,
        )

    def analyze(
        self,
        source: RunRecord[T2EchoCfg, T2EchoResult],
        options: T2EchoAnalyzeOptions,
        *,
        plots: Plots,
    ) -> T2EchoAnalysis:
        """fit_phase frees the fringe phase; decay-only fits ignore this option."""
        result = source.result
        fit_method = options.fit_method
        fit_phase = options.fit_phase

        xs, signals = result.times, result.signals

        xs = xs[1:]
        signals = signals[1:]

        real_signals = rotate2real(signals).real

        if fit_method == "fringe":
            fixedparams = None if fit_phase else [None, None, None, 0.0, None]
            t2e, t2eerr, detune, detune_err, y_fit, _ = fit_decay_fringe(
                xs, real_signals, fixedparams=fixedparams
            )
        elif fit_method == "decay":
            t2e, t2eerr, y_fit, _ = fit_decay(xs, real_signals)
            detune = 0.0
            detune_err = 0.0
        else:
            raise ValueError(f"Unknown fit_method: {fit_method}")

        fig, ax = plots.subplots("fit", figsize=config.figsize)

        ax.plot(xs, real_signals, label="data", ls="-", marker="o", markersize=5)
        ax.plot(xs, y_fit, label="fit", c="orange", zorder=1)

        t2e_str = f"{t2e:.2f}us ± {t2eerr:.2f}us"
        if fit_method == "fringe":
            detune_str = f"{detune:.2f}MHz ± {detune_err * 1e3:.2f}kHz"
            title = r"$T_{2echo}$ fringe = " + f"{t2e_str}, detune = {detune_str}"

        elif fit_method == "decay":
            title = r"$T_{2echo}$ decay = " + f"{t2e_str}"

        else:
            raise ValueError(f"Unknown fit_method: {fit_method}")

        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Delay Time (us)", fontsize=14)
        ax.set_ylabel("Signal (a.u.)", fontsize=14)
        ax.legend(loc="upper right")
        ax.grid(True)

        fig.tight_layout()

        return T2EchoAnalysis(
            t2e=float(t2e),
            t2e_err=float(t2eerr),
            detune=float(detune),
            detune_err=float(detune_err),
        )
