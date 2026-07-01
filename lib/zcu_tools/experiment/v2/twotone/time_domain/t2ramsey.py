from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

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
    record_result,
    retrieve_result,
)
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import setup_devices
from zcu_tools.experiment.v2.runner import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.liveplot import LivePlot1D
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
from zcu_tools.utils.fitting import fit_decay, fit_decay_fringe
from zcu_tools.utils.process import rotate2real


def t2ramsey_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return rotate2real(signals).real


@dataclass(frozen=True)
class T2RamseyResult:
    times: NDArray[np.float64]
    signals: NDArray[np.complex128]
    true_activate_detune: float
    cfg_snapshot: T2RamseyCfg | None = None


class T2RamseyModuleCfg(ConfigBase):
    reset: ResetCfg | None = None
    pi2_pulse: PulseCfg
    readout: ReadoutCfg


class T2RamseySweepCfg(ConfigBase):
    length: SweepCfg


class T2RamseyCfg(ProgramV2Cfg, ExpCfgModel):
    modules: T2RamseyModuleCfg
    sweep: T2RamseySweepCfg


class T2RamseyExp(PersistableExperiment[T2RamseyResult, T2RamseyCfg]):
    # times stored as seconds on disk -> scale=US_TO_S
    AXES_SPEC = AxesSpec(
        axes=(Axis("times", "Time", "s", US_TO_S),),
        z=ZSpec("signals", "Signal", "a.u."),
        result_type=T2RamseyResult,
        cfg_type=T2RamseyCfg,
        tag="twotone/ge/t2ramsey",
    )

    @record_result
    def run(
        self,
        soc,
        soccfg,
        cfg: T2RamseyCfg,
        *,
        detune: float = 0.0,
        acquire_kwargs: dict[str, Any] | None = None,
    ) -> T2RamseyResult:
        orig_cfg = deepcopy(cfg)

        setup_devices(cfg, progress=True)
        modules = cfg.modules

        length_sweep = cfg.sweep.length
        lengths = sweep2array(length_sweep, "time", {"soccfg": soccfg})

        # calculate true scanned detune based on rounding lengths
        if detune != 0.0:
            detune_lengths = sweep2array(
                cfg.sweep.length,
                "phase",
                {
                    "soccfg": soccfg,
                    "gen_ch": modules.pi2_pulse.ch,
                    "scaler": 360 * detune,
                },
            )
            mask = lengths > 0
            detune_ratio = np.mean(detune_lengths[mask] / lengths[mask]).item()
            true_detune = detune * detune_ratio
        else:
            true_detune = 0.0

        title = f"T2 Ramsey - detune {detune:.2f} MHz"
        with LivePlot1D(
            "Time (us)", "Real Signal", segment_kwargs={"title": title}
        ) as viewer:
            signals_buffer = SignalBuffer(
                (len(lengths),),
                on_update=lambda data: viewer.update(
                    lengths, t2ramsey_signal2real(data)
                ),
            )
            with Schedule(cfg, signals_buffer) as sched:
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
                        Delay("t2_delay", delay=length_param),
                        Pulse(
                            name="pi2_pulse2",
                            cfg=modules.pi2_pulse.with_updates(
                                phase=modules.pi2_pulse.phase + detune_param
                            ),
                        ),
                        Readout("readout", modules.readout),
                    )
                    .declare_sweep("length", length_sweep)
                    .build_and_acquire(
                        **(acquire_kwargs or {}),
                    )
                )

        # record result
        return T2RamseyResult(
            times=lengths,
            signals=signals_buffer.array,
            cfg_snapshot=orig_cfg,
            true_activate_detune=true_detune,
        )

    @retrieve_result
    def analyze(
        self, result: T2RamseyResult | None = None, *, fit_fringe: bool = True
    ) -> tuple[float, float, float, float, Figure]:
        assert result is not None, "no result found"

        lengths, signals = result.times, result.signals

        real_signals = t2ramsey_signal2real(signals)

        if fit_fringe:
            zero_signal = real_signals[np.argmin(np.abs(lengths))]
            if zero_signal > 0.5 * (np.max(real_signals) + np.min(real_signals)):
                init_phase = 0.0
            else:
                init_phase = 180.0
            fixedparams = [None, None, None, init_phase, None]
            t2r, t2rerr, detune, detune_err, y_fit, _ = fit_decay_fringe(
                lengths, real_signals, fixedparams=fixedparams
            )
        else:
            t2r, t2rerr, y_fit, _ = fit_decay(lengths, real_signals)
            detune = 0.0
            detune_err = 0.0

        fig, ax = plt.subplots(figsize=config.figsize)
        assert isinstance(fig, Figure)

        ax.plot(lengths, real_signals, label="meas", ls="-", marker="o", markersize=3)
        ax.plot(lengths, y_fit, label="fit")
        t2r_str = f"{t2r:.2f}us ± {t2rerr:.2f}us"
        if fit_fringe:
            detune_str = f"{detune:.2f}MHz ± {detune_err * 1e3:.2f}kHz"
            ax.set_title(f"T2 fringe = {t2r_str}, detune = {detune_str}", fontsize=15)
        else:
            ax.set_title(f"T2 decay = {t2r_str}", fontsize=15)
        ax.set_xlabel("Delay Time (us)")
        ax.set_ylabel("Signal Real (a.u.)")
        ax.legend()
        ax.grid(True)

        fig.tight_layout()

        return t2r, t2rerr, detune, detune_err, fig
