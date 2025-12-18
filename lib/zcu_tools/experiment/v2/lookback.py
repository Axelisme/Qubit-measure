from __future__ import annotations

import warnings
from copy import deepcopy
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d

from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.liveplot import LivePlotter1D
from zcu_tools.program.v2 import OneToneProgram, OneToneProgramCfg
from zcu_tools.utils.datasaver import load_data, save_data

from .runner import HardTask, TaskConfig, run_task

LookbackResultType = Tuple[NDArray[np.float64], NDArray[np.complex128]]


class LookbackTaskConfig(TaskConfig, OneToneProgramCfg): ...


def lookback_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return np.abs(signals)


class LookbackExperiment(AbsExperiment):
    def run(self, soc, soccfg, cfg: LookbackTaskConfig) -> LookbackResultType:
        cfg = deepcopy(cfg)

        if cfg.setdefault("reps", 1) != 1:
            warnings.warn("reps is not 1 in config, this will be ignored.")
            cfg["reps"] = 1

        prog = OneToneProgram(soccfg, cfg)
        Ts = prog.get_time_axis(ro_index=0) + cfg["readout"]["ro_cfg"]["trig_offset"]
        assert isinstance(Ts, np.ndarray)

        with LivePlotter1D("Time (us)", "Amplitude") as viewer:
            signals = run_task(
                task=HardTask(
                    measure_fn=lambda ctx, update_hook: (
                        OneToneProgram(soccfg, ctx.cfg).acquire_decimated(
                            soc, progress=False, callback=update_hook
                        )
                    ),
                    raw2signal_fn=lambda raw: raw[0].dot([1, 1j]),
                    result_shape=(len(Ts),),
                ),
                init_cfg=cfg,
                update_hook=lambda ctx: viewer.update(
                    Ts, lookback_signal2real(ctx.data)
                ),
            )

        # record last cfg and result
        self.last_cfg = dict(cfg)
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
    ) -> Tuple[float, Figure]:
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

        fig, ax = plt.subplots(figsize=config.figsize)

        ax.plot(Ts, signals.real, label="I value")
        ax.plot(Ts, signals.imag, label="Q value")
        ax.plot(Ts, y, label="mag")
        if plot_fit:
            ax.axvline(offset, color="r", linestyle="--", label="predict_offset")
        if ro_cfg is not None:
            trig_offset = ro_cfg["trig_offset"]
            ro_length = ro_cfg["ro_length"]
            ax.axvline(trig_offset, color="g", linestyle="--", label="ro start")
            ax.axvline(
                trig_offset + ro_length, color="g", linestyle="--", label="ro end"
            )
        ax.set_xlabel("Time (us)")
        ax.set_ylabel("a.u.")
        ax.grid(True)
        ax.legend()

        fig.tight_layout()

        return offset, fig

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

    def load(self, filepath: str, **kwargs) -> LookbackResultType:
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
