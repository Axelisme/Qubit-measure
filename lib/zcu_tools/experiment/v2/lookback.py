from __future__ import annotations

import warnings
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.liveplot import LivePlotter1D
from zcu_tools.program.v2 import OneToneProgram
from zcu_tools.utils.datasaver import save_data

from .runner import HardTask, Runner

LookbackResultType = Tuple[np.ndarray, np.ndarray]


def lookback_signal2real(signals: np.ndarray) -> np.ndarray:
    return np.abs(signals)


class LookbackExperiment(AbsExperiment[LookbackResultType]):
    def run(
        self, soc, soccfg, cfg: Dict[str, Any], *, progress: bool = True
    ) -> LookbackResultType:
        cfg = deepcopy(cfg)

        if cfg.setdefault("reps", 1) != 1:
            warnings.warn("reps is not 1 in config, this will be ignored.")
            cfg["reps"] = 1

        prog = OneToneProgram(soccfg, cfg)
        Ts = prog.get_time_axis(ro_index=0) + cfg["readout"]["ro_cfg"]["trig_offset"]

        with LivePlotter1D("Time (us)", "Amplitude", disable=not progress) as viewer:
            signals = Runner(
                task=HardTask(
                    measure_fn=lambda ctx, update_hook: (
                        OneToneProgram(soccfg, ctx.cfg).acquire_decimated(
                            soc, progress=False, callback=update_hook
                        )
                    ),
                    raw2signal_fn=lambda x: x[0].dot([1, 1j]),
                    result_shape=(len(Ts),),
                ),
                update_hook=lambda ctx: viewer.update(
                    Ts,
                    lookback_signal2real(np.asarray(ctx.get_data())),  # type: ignore
                ),
                update_interval=3.0,
            ).run(cfg)
            signals = np.asarray(signals)  # type: ignore

        # record last cfg and result
        self.last_cfg = cfg
        self.last_result = (Ts, signals)

        return Ts, signals

    def analyze(
        self,
        result: Optional[LookbackResultType] = None,
        *,
        ratio: float = 0.3,
        smooth: Optional[float] = None,
        ro_cfg: Optional[dict] = None,
        plot_fit: bool = True,
    ) -> float:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        Ts, signals = result

        if smooth is not None:
            signals = gaussian_filter1d(signals, smooth)
        y = np.abs(signals)

        # start from max point, find largest idx where y is smaller than ratio * max_y
        max_idx = np.argmax(y)
        candidate_mask = y[:max_idx] < ratio * y[max_idx]
        if not np.any(candidate_mask):
            offset = Ts[0]
        else:
            offset = Ts[np.nonzero(candidate_mask)[0][-1]]

        plt.figure(figsize=config.figsize)
        plt.plot(Ts, signals.real, label="I value")
        plt.plot(Ts, signals.imag, label="Q value")
        plt.plot(Ts, y, label="mag")
        if plot_fit:
            plt.axvline(offset, color="r", linestyle="--", label="predict_offset")
        if ro_cfg is not None:
            trig_offset = ro_cfg["trig_offset"]
            ro_length = ro_cfg["ro_length"]
            plt.axvline(trig_offset, color="g", linestyle="--", label="ro start")
            plt.axvline(
                trig_offset + ro_length, color="g", linestyle="--", label="ro end"
            )

        plt.xlabel("Time (us)")
        plt.ylabel("a.u.")
        plt.legend()
        plt.show()

        return offset

    def save(
        self,
        filepath: str,
        result: Optional[LookbackResultType] = None,
        comment: Optional[str] = None,
        tag: str = "lookback",
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
