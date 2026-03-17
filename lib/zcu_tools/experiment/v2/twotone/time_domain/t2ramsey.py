from __future__ import annotations

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from typeguard import check_type
from typing_extensions import Any, NotRequired, Optional, TypeAlias, TypedDict

from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.utils import format_sweep1D, sweep2array
from zcu_tools.experiment.v2.runner import Task, TaskCfg, run_task
from zcu_tools.experiment.v2.utils import round_zcu_time
from zcu_tools.liveplot import LivePlotter1D
from zcu_tools.program import SweepCfg
from zcu_tools.program.v2 import (
    Delay,
    ModularProgramCfg,
    ModularProgramV2,
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


def t2ramsey_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return rotate2real(signals).real


T2RamseyResult: TypeAlias = tuple[NDArray[np.float64], NDArray[np.complex128]]


class T2RamseyModuleCfg(TypedDict, closed=True):
    reset: NotRequired[ResetCfg]
    pi2_pulse: PulseCfg
    readout: ReadoutCfg


class T2RamseyCfg(ModularProgramCfg, TaskCfg):
    modules: T2RamseyModuleCfg
    sweep: dict[str, SweepCfg]


class T2RamseyExp(AbsExperiment[T2RamseyResult, T2RamseyCfg]):
    def run(
        self, soc, soccfg, cfg: dict[str, Any], *, detune: float = 0.0
    ) -> T2RamseyResult:
        cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")
        _cfg = check_type(deepcopy(cfg), T2RamseyCfg)

        ts = sweep2array(_cfg["sweep"]["length"])
        ts = round_zcu_time(ts, soccfg)

        t2r_spans = sweep2param("length", _cfg["sweep"]["length"])

        with LivePlotter1D(
            "Time (us)", "Amplitude", segment_kwargs={"title": "T2 Ramsey"}
        ) as viewer:
            signals = run_task(
                task=Task(
                    measure_fn=lambda ctx, update_hook: (
                        (modules := ctx.cfg["modules"])
                        and ModularProgramV2(
                            soccfg,
                            ctx.cfg,
                            modules=[
                                Reset("reset", modules.get("reset")),
                                Pulse("pi2_pulse1", modules["pi2_pulse"]),
                                Delay("t2_delay", delay=t2r_spans),
                                Pulse(
                                    name="pi2_pulse2",
                                    cfg=PulseCfg(
                                        **modules["pi2_pulse"],
                                        phase=modules["pi2_pulse"]["phase"]
                                        + 360 * detune * t2r_spans,
                                    ),
                                ),
                                Readout("readout", modules["readout"]),
                            ],
                        ).acquire(soc, progress=False, callback=update_hook)
                    ),
                    result_shape=(len(ts),),
                ),
                init_cfg=_cfg,
                on_update=lambda ctx: viewer.update(
                    ts, t2ramsey_signal2real(ctx.root_data)
                ),
            )

        # record last cfg and result
        self.last_cfg = _cfg
        self.last_result = (ts, signals)

        return ts, signals

    def analyze(
        self, result: Optional[T2RamseyResult] = None, *, fit_fringe: bool = True
    ) -> tuple[float, float, float, float, Figure]:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        xs, signals = result

        real_signals = t2ramsey_signal2real(signals)

        if fit_fringe:
            t2r, t2rerr, detune, detune_err, y_fit, _ = fit_decay_fringe(
                xs, real_signals
            )
        else:
            t2r, t2rerr, y_fit, _ = fit_decay(xs, real_signals)
            detune = 0.0
            detune_err = 0.0

        fig, ax = plt.subplots(figsize=config.figsize)
        assert isinstance(fig, Figure)

        ax.plot(xs, real_signals, label="meas", ls="-", marker="o", markersize=3)
        ax.plot(xs, y_fit, label="fit")
        t2r_str = f"{t2r:.2f}us ± {t2rerr:.2f}us"
        if fit_fringe:
            detune_str = f"{detune:.2f}MHz ± {detune_err * 1e3:.2f}kHz"
            ax.set_title(f"T2 fringe = {t2r_str}, detune = {detune_str}", fontsize=15)
        else:
            ax.set_title(f"T2 decay = {t2r_str}", fontsize=15)
        ax.set_xlabel("Time (us)")
        ax.set_ylabel("Signal Real (a.u.)")
        ax.legend()
        ax.grid(True)

        fig.tight_layout()

        return t2r, t2rerr, detune, detune_err, fig

    def save(
        self,
        filepath: str,
        result: Optional[T2RamseyResult] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/ge/t2ramsey",
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

    def load(self, filepath: str, **kwargs) -> T2RamseyResult:
        signals, Ts, _ = load_data(filepath, **kwargs)
        assert Ts is not None
        assert len(Ts.shape) == 1 and len(signals.shape) == 1
        assert Ts.shape == signals.shape

        Ts = Ts * 1e6  # s -> us

        Ts = Ts.astype(np.float64)
        signals = signals.astype(np.complex128)

        self.last_cfg = None
        self.last_result = (Ts, signals)

        return Ts, signals
