from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import make_comment, parse_comment, setup_devices
from zcu_tools.experiment.v2.runner import Task, TaskState, run_task
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
from zcu_tools.utils.datasaver import load_data, save_data
from zcu_tools.utils.fitting import fit_decay, fit_decay_fringe
from zcu_tools.utils.process import rotate2real


def t2ramsey_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return rotate2real(signals).real


@dataclass(frozen=True)
class T2RamseyResult:
    times: NDArray[np.float64]
    signals: NDArray[np.complex128]
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


class T2RamseyExp(AbsExperiment[T2RamseyResult, T2RamseyCfg]):
    def run(
        self,
        soc,
        soccfg,
        cfg: T2RamseyCfg,
        *,
        detune: float = 0.0,
        acquire_kwargs: dict[str, Any] | None = None,
    ) -> tuple[T2RamseyResult, float]:
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

        def measure_fn(
            ctx: TaskState[NDArray[np.complex128], Any, T2RamseyCfg], update_hook
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
                    Delay("t2_delay", delay=length_param),
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

        title = f"T2 Ramsey - detune {detune:.2f} MHz"
        with LivePlot1D(
            "Time (us)", "Real Signal", segment_kwargs={"title": title}
        ) as viewer:
            signals = run_task(
                task=Task(
                    measure_fn=measure_fn,
                    result_shape=(len(lengths),),
                    pbar_n=cfg.rounds,
                ),
                init_cfg=cfg,
                on_update=lambda ctx: viewer.update(
                    lengths, t2ramsey_signal2real(ctx.root_data)
                ),
            )

        # record result
        self.last_result = T2RamseyResult(
            times=lengths, signals=signals, cfg_snapshot=orig_cfg
        )

        return self.last_result, true_detune

    def analyze(
        self, result: T2RamseyResult | None = None, *, fit_fringe: bool = True
    ) -> tuple[float, float, float, float, Figure]:
        if result is None:
            result = self.last_result
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

    def save(
        self,
        filepath: str,
        result: T2RamseyResult | None = None,
        comment: str | None = None,
        tag: str = "twotone/ge/t2ramsey",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        Ts, signals = result.times, result.signals
        cfg = result.cfg_snapshot
        if cfg is None:
            raise ValueError("cfg_snapshot is None")
        comment = make_comment(cfg, comment)

        save_data(
            filepath=filepath,
            x_info={"name": "Time", "unit": "s", "values": Ts * 1e-6},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals},
            comment=comment,
            tag=tag,
            **kwargs,
        )

    def load(self, filepath: str, **kwargs) -> T2RamseyResult:
        signals, Ts, _, comment = load_data(filepath, return_comment=True, **kwargs)
        assert Ts is not None
        assert len(Ts.shape) == 1 and len(signals.shape) == 1
        assert Ts.shape == signals.shape

        Ts = Ts * 1e6  # s -> us

        Ts = Ts.astype(np.float64)
        signals = signals.astype(np.complex128)

        cfg_snapshot = None
        if comment is not None:
            _cfg, _, _ = parse_comment(comment)

            if _cfg is not None:
                cfg_snapshot = T2RamseyCfg.validate_or_warn(_cfg, source=filepath)
        self.last_result = T2RamseyResult(
            times=Ts, signals=signals, cfg_snapshot=cfg_snapshot
        )

        return self.last_result
