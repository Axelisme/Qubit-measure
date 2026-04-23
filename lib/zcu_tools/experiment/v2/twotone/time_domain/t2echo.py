from __future__ import annotations

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from pydantic import BaseModel
from typing_extensions import (
    Any,
    Literal,
    Optional,
    TypeAlias,
)

from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import setup_devices
from zcu_tools.experiment.v2.runner import Task, TaskState, run_task
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.liveplot import LivePlot1D
from zcu_tools.program import SweepCfg
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
    sweep2param,
)
from zcu_tools.utils.datasaver import load_data, save_data
from zcu_tools.utils.fitting import fit_decay, fit_decay_fringe
from zcu_tools.utils.process import rotate2real

# (times, signals)
T2EchoResult: TypeAlias = tuple[NDArray[np.float64], NDArray[np.complex128]]


def t2echo_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return rotate2real(signals).real


class T2EchoModuleCfg(BaseModel):
    reset: Optional[ResetCfg] = None
    pi2_pulse: PulseCfg
    pi_pulse: PulseCfg
    readout: ReadoutCfg


class T2EchoSweepCfg(BaseModel):
    length: SweepCfg


class T2EchoCfg(ProgramV2Cfg, ExpCfgModel):
    modules: T2EchoModuleCfg
    sweep: T2EchoSweepCfg


class T2EchoExp(AbsExperiment[T2EchoResult, T2EchoCfg]):
    def run(
        self,
        soc,
        soccfg,
        cfg: T2EchoCfg,
        *,
        detune: float = 0.0,
        acquire_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[T2EchoResult, float]:
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
                **(acquire_kwargs or {}),
            )

        with LivePlot1D(
            "Time (us)", "Amplitude", segment_kwargs={"title": "T2 Echo"}
        ) as viewer:
            signals = run_task(
                task=Task(
                    measure_fn=measure_fn,
                    result_shape=(len(lengths),),
                    pbar_n=cfg.rounds,
                ),
                init_cfg=cfg,
                on_update=lambda ctx: viewer.update(
                    lengths, t2echo_signal2real(ctx.root_data)
                ),
            )

        # record last cfg and result
        self.last_cfg = deepcopy(cfg)
        self.last_result = (lengths, signals)

        return (lengths, signals), true_detune

    def analyze(
        self,
        result: Optional[T2EchoResult] = None,
        *,
        fit_method: Literal["fringe", "decay"] = "decay",
    ) -> tuple[float, float, float, float, Figure]:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        xs, signals = result

        xs = xs[1:]
        signals = signals[1:]

        real_signals = rotate2real(signals).real

        if fit_method == "fringe":
            t2e, t2eerr, detune, detune_err, y_fit, _ = fit_decay_fringe(
                xs, real_signals
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
            title = r"$T_{2echo}$ decay = {t2e_str}"

        else:
            raise ValueError(f"Unknown fit_method: {fit_method}")

        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Delay Time (us)", fontsize=14)
        ax.set_ylabel("Signal (a.u.)", fontsize=14)
        ax.legend(loc="upper right")
        ax.grid(True)

        fig.tight_layout()

        return t2e, t2eerr, detune, detune_err, fig

    def save(
        self,
        filepath: str,
        result: Optional[T2EchoResult] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/ge/t2echo",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        Ts, signals = result
        save_data(
            filepath=filepath,
            x_info={"name": "Time", "unit": "s", "values": Ts * 1e-6},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals},
            comment=comment,
            tag=tag,
            **kwargs,
        )

    def load(self, filepath: str, **kwargs) -> T2EchoResult:
        signals, Ts, _, cfg = load_data(filepath, return_cfg=True, **kwargs)
        assert Ts is not None
        assert len(Ts.shape) == 1 and len(signals.shape) == 1
        assert Ts.shape == signals.shape

        Ts = Ts * 1e6  # s -> us

        Ts = Ts.astype(np.float64)
        signals = signals.astype(np.complex128)

        self.last_cfg = T2EchoCfg.validate_or_warn(cfg, source=filepath)
        self.last_result = (Ts, signals)

        return Ts, signals
