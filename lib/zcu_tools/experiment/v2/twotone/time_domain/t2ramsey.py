from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.utils import format_sweep1D, sweep2array
from zcu_tools.liveplot import LivePlotter1D
from zcu_tools.program.v2 import (
    Delay,
    ModularProgramV2,
    Pulse,
    make_readout,
    make_reset,
    sweep2param,
)
from zcu_tools.utils.datasaver import save_data
from zcu_tools.utils.fitting import fit_decay, fit_decay_fringe
from zcu_tools.utils.process import rotate2real

from ...runner import HardTask, Runner


def t2ramsey_signal2real(signals: np.ndarray) -> np.ndarray:
    return rotate2real(signals).real  # type: ignore


T2RamseyResultType = Tuple[np.ndarray, np.ndarray]  # (times, signals)


class T2RamseyExperiment(AbsExperiment[T2RamseyResultType]):
    def run(
        self,
        soc,
        soccfg,
        cfg: Dict[str, Any],
        *,
        detune: float = 0.0,
        progress: bool = True,
    ) -> T2RamseyResultType:
        cfg = deepcopy(cfg)

        cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")

        ts = sweep2array(cfg["sweep"]["length"])

        t2r_spans = sweep2param("length", cfg["sweep"]["length"])

        with LivePlotter1D(
            "Time (us)",
            "Amplitude",
            segment_kwargs={"title": "T2 Ramsey"},
            disable=not progress,
        ) as viewer:
            signals = Runner(
                task=HardTask(
                    measure_fn=lambda ctx, update_hook: (
                        ModularProgramV2(
                            soccfg,
                            ctx.cfg,
                            modules=[
                                make_reset("reset", ctx.cfg.get("reset")),
                                Pulse("pi2_pulse1", ctx.cfg["pi2_pulse"]),
                                Delay("t2_delay", delay=t2r_spans),
                                Pulse(
                                    name="pi2_pulse2",
                                    cfg={
                                        **ctx.cfg["pi2_pulse"],
                                        "phase": ctx.cfg["pi2_pulse"]["phase"]
                                        + 360 * detune * t2r_spans,
                                    },
                                ),
                                make_readout("readout", ctx.cfg["readout"]),
                            ],
                        ).acquire(soc, progress=False, callback=update_hook)
                    ),
                    result_shape=(len(ts),),
                ),
                update_hook=lambda ctx: viewer.update(
                    ts, t2ramsey_signal2real(np.asarray(ctx.get_data()))
                ),
            ).run(cfg)
            signals = np.asarray(signals)

        # record last cfg and result
        self.last_cfg = cfg
        self.last_result = (ts, signals)

        return ts, signals

    def analyze(
        self,
        result: Optional[T2RamseyResultType] = None,
        *,
        max_contrast: bool = True,
        fit_fringe: bool = True,
    ) -> Tuple[float, float, float, float, plt.Figure]:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        xs, signals = result

        if max_contrast:
            real_signals = rotate2real(signals).real
        else:
            real_signals = np.abs(signals)

        if fit_fringe:
            t2r, t2rerr, detune, detune_err, y_fit, _ = fit_decay_fringe(
                xs, real_signals
            )
        else:
            t2r, t2rerr, y_fit, _ = fit_decay(xs, real_signals)
            detune = 0.0
            detune_err = 0.0

        fig, ax = plt.subplots(figsize=config.figsize)
        ax.plot(xs, real_signals, label="meas", ls="-", marker="o", markersize=3)
        ax.plot(xs, y_fit, label="fit")
        t2r_str = f"{t2r:.2f}us ± {t2rerr:.2f}us"
        if fit_fringe:
            detune_str = f"{detune:.2f}MHz ± {detune_err * 1e3:.2f}kHz"
            ax.set_title(f"T2 fringe = {t2r_str}, detune = {detune_str}", fontsize=15)
        else:
            ax.set_title(f"T2 decay = {t2r_str}", fontsize=15)
        ax.set_xlabel("Time (us)")
        ax.set_ylabel("Signal Real (a.u.)" if max_contrast else "Magnitude (a.u.)")
        ax.legend()

        fig.tight_layout()

        return t2r, t2rerr, detune, detune_err, fig

    def save(
        self,
        filepath: str,
        result: Optional[T2RamseyResultType] = None,
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
