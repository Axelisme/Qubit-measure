from __future__ import annotations

import warnings
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d
from typeguard import check_type
from typing_extensions import Any, Dict, NotRequired, Optional, Tuple, TypedDict

from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.v2.runner import Task, TaskCfg, run_task
from zcu_tools.liveplot import LivePlotter1D
from zcu_tools.program.v2 import ModularProgramCfg, OneToneProgram
from zcu_tools.program.v2.modules import PulseCfg, PulseReadoutCfg, ResetCfg
from zcu_tools.utils.datasaver import load_data, save_data

LookbackResult = Tuple[NDArray[np.float64], NDArray[np.complex128]]


class LookbackModuleCfg(TypedDict, closed=True):
    reset: NotRequired[ResetCfg]
    init_pulse: NotRequired[PulseCfg]
    readout: PulseReadoutCfg


class LookbackCfg(ModularProgramCfg, TaskCfg):
    modules: LookbackModuleCfg


def lookback_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return np.abs(signals)


class LookbackExp(AbsExperiment[LookbackResult, LookbackCfg]):
    def run(self, soc, soccfg, cfg: Dict[str, Any]) -> LookbackResult:
        _cfg = check_type(deepcopy(cfg), LookbackCfg)

        if _cfg.setdefault("reps", 1) != 1:
            warnings.warn("reps is not 1 in config, this will be ignored.")
            _cfg["reps"] = 1

        prog = OneToneProgram(soccfg, _cfg)
        Ts = (
            prog.get_time_axis(ro_index=0)
            + _cfg["modules"]["readout"]["ro_cfg"]["trig_offset"]
        )
        assert isinstance(Ts, np.ndarray)

        with LivePlotter1D("Time (us)", "Amplitude") as viewer:

            def measure_fn(ctx, update_hook):
                return OneToneProgram(soccfg, ctx.cfg).acquire_decimated(
                    soc, progress=False, callback=update_hook
                )

            signals = run_task(
                task=Task(
                    measure_fn=measure_fn,
                    raw2signal_fn=lambda raw: raw[0].dot([1, 1j]),
                    result_shape=(len(Ts),),
                ),
                init_cfg=_cfg,
                on_update=lambda ctx: viewer.update(
                    Ts, lookback_signal2real(ctx.root_data)
                ),
            )

        # record last cfg and result
        self.last_cfg = _cfg
        self.last_result = (Ts, signals)

        return Ts, signals

    def analyze(
        self,
        result: Optional[LookbackResult] = None,
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
        result: Optional[LookbackResult] = None,
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

    def load(self, filepath: str, **kwargs) -> LookbackResult:
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
