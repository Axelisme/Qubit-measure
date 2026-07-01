from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment import (
    US_TO_S,
    AxesSpec,
    Axis,
    PersistableExperiment,
    ZSpec,
    config,
    retrieve_result,
)
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import setup_devices
from zcu_tools.experiment.v2.runner import MeasureSession, TaskState
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.liveplot import LivePlot1D
from zcu_tools.program.v2 import (
    Delay,
    ModularProgramV2,
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
from zcu_tools.utils.fitting import fit_decay, fit_decay_fringe
from zcu_tools.utils.process import rotate2real


@dataclass(frozen=True)
class T2EchoResult:
    times: NDArray[np.float64]
    signals: NDArray[np.complex128]
    cfg_snapshot: T2EchoCfg | None = None


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


class T2EchoExp(PersistableExperiment[T2EchoResult, T2EchoCfg]):
    # times stores us in memory, s on disk -> scale=US_TO_S
    AXES_SPEC = AxesSpec(
        axes=(Axis("times", "Time", "s", scale=US_TO_S),),
        z=ZSpec("signals", "Signal", "a.u."),
        result_type=T2EchoResult,
        cfg_type=T2EchoCfg,
        tag="twotone/ge/t2echo",
    )

    def run(
        self,
        soc,
        soccfg,
        cfg: T2EchoCfg,
        *,
        detune: float = 0.0,
        acquire_kwargs: dict[str, Any] | None = None,
    ) -> tuple[T2EchoResult, float]:
        orig_cfg = deepcopy(cfg)

        setup_devices(cfg, progress=True)

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

        def measure_fn(
            ctx: TaskState[NDArray[np.complex128], Any, T2EchoCfg], update_hook
        ):
            cfg = ctx.cfg
            modules = cfg.modules

            length_sweep = cfg.sweep.length
            length_param = sweep2param("length", length_sweep)
            detune_param = 360 * detune * length_param

            return ModularProgramV2(
                soccfg,
                cfg,
                modules=[
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
                ],
                sweep=[("length", length_sweep)],
            ).acquire(
                soc,
                progress=False,
                round_hook=update_hook,
                stop_checkers=[ctx.is_stop],
                **(acquire_kwargs or {}),
            )

        with LivePlot1D(
            "Time (us)",
            "Amplitude",
            segment_kwargs={"title": f"T2 Echo (detune={true_detune:.3f}MHz)"},
        ) as viewer:
            with MeasureSession(cfg) as run:
                signals_buffer = run.buffer(
                    (len(lengths),),
                    dtype=np.complex128,
                    on_update=lambda data: viewer.update(
                        lengths, t2echo_signal2real(data)
                    ),
                )
                signals_buffer.measure(measure_fn, pbar_n=run.cfg.rounds)
                signals = signals_buffer.array

        # record result
        self.last_result = T2EchoResult(
            times=lengths, signals=signals, cfg_snapshot=orig_cfg
        )

        return self.last_result, true_detune

    @retrieve_result
    def analyze(
        self,
        result: T2EchoResult | None = None,
        *,
        fit_method: Literal["fringe", "decay"] = "decay",
    ) -> tuple[float, float, float, float, Figure]:
        assert result is not None, "no result found"

        xs, signals = result.times, result.signals

        xs = xs[1:]
        signals = signals[1:]

        real_signals = rotate2real(signals).real

        if fit_method == "fringe":
            fixedparams = [None, None, None, 0.0, None]
            t2e, t2eerr, detune, detune_err, y_fit, _ = fit_decay_fringe(
                xs, real_signals, fixedparams=fixedparams
            )
        elif fit_method == "decay":
            t2e, t2eerr, y_fit, _ = fit_decay(xs, real_signals)
            detune = 0.0
            detune_err = 0.0
        else:
            raise ValueError(f"Unknown fit_method: {fit_method}")

        fig, ax = plt.subplots(figsize=config.figsize)
        assert isinstance(fig, Figure)

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

        return t2e, t2eerr, detune, detune_err, fig
