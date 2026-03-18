from __future__ import annotations

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray
from typeguard import check_type
from typing_extensions import (
    Any,
    Callable,
    NotRequired,
    Optional,
    TypeAlias,
    TypedDict,
    Union,
    cast,
)

from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.utils import sweep2array
from zcu_tools.experiment.v2.runner import (
    Task,
    TaskCfg,
    TaskState,
    run_task,
)
from zcu_tools.experiment.v2.utils import round_zcu_time
from zcu_tools.liveplot import LivePlotter2DwithLine
from zcu_tools.program import SweepCfg
from zcu_tools.program.v2 import (
    Delay,
    ModularProgramCfg,
    ModularProgramV2,
    Pulse,
    PulseCfg,
    Readout,
    ReadoutCfg,
    Repeat,
    Reset,
    ResetCfg,
    sweep2param,
)
from zcu_tools.utils.datasaver import load_data, save_data
from zcu_tools.utils.fitting import fit_decay
from zcu_tools.utils.process import rotate2real


def cpmg_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    real_signals = rotate2real(signals).real
    max_vals = np.max(real_signals, axis=1, keepdims=True)
    min_vals = np.min(real_signals, axis=1, keepdims=True)
    return (real_signals - min_vals) / (max_vals - min_vals)


CPMG_Result: TypeAlias = tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.complex128]
]


class CPMG_ModuleCfg(TypedDict, closed=True):
    reset: NotRequired[ResetCfg]
    pi2_pulse: PulseCfg
    pi_pulse: PulseCfg
    readout: ReadoutCfg


class CPMG_SweepCfg(TypedDict, closed=True):
    times: Union[SweepCfg, NDArray, list]
    length: SweepCfg


class CPMG_Cfg(ModularProgramCfg, TaskCfg):
    modules: CPMG_ModuleCfg
    sweep: CPMG_SweepCfg


class CPMG_Exp(AbsExperiment[CPMG_Result, CPMG_Cfg]):
    def run(self, soc, soccfg, cfg: dict[str, Any]) -> CPMG_Result:
        _cfg = check_type(deepcopy(cfg), CPMG_Cfg)

        times_sweep = _cfg["sweep"]["times"]
        len_sweep = _cfg["sweep"]["length"]
        _cfg["sweep"].pop("times")  # type: ignore

        times = sweep2array(times_sweep, allow_array=True)
        ts = sweep2array(len_sweep)  # predicted times
        ts = round_zcu_time(ts, soccfg)

        cpmg_spans = sweep2param("length", len_sweep)

        if np.min(times) <= 0:
            raise ValueError("times should be larger than 0")

        def measure_fn(ctx: TaskState, update_hook: Callable[[int, Any], None]):
            cfg = ctx.cfg
            modules = cfg["modules"]
            interval = cpmg_spans / ctx.env["time"]
            return ModularProgramV2(
                soccfg,
                cfg,
                modules=[
                    Reset("reset", modules.get("reset")),
                    Pulse("pi2_pulse1", modules["pi2_pulse"], pulse_name="pi2_pulse"),
                    Delay("initial_cpmg_delay", delay=0.5 * interval),
                    Repeat(
                        name="cpmg_pi_loop",
                        n=ctx.env["time"] - 1,
                        sub_module=[
                            Pulse("pi_pulse", modules["pi_pulse"]),
                            Delay("interval_cpmg_delay", delay=interval),
                        ],
                    ),
                    Pulse("last_pi_pulse", modules["pi_pulse"], pulse_name="pi_pulse"),
                    Delay("final_cpmg_delay", delay=0.5 * interval),
                    Pulse("pi2_pulse2", modules["pi2_pulse"], pulse_name="pi2_pulse"),
                    Readout("readout", modules["readout"]),
                ],
            ).acquire(soc, progress=False, callback=update_hook)

        with LivePlotter2DwithLine(
            "Number of Pi", "Time (us)", line_axis=1, num_lines=2, title="CPMG"
        ) as viewer:
            signals = run_task(
                task=Task(measure_fn=measure_fn, result_shape=(len(ts),)).scan(
                    "times",
                    times.tolist(),
                    before_each=lambda _, ctx, time: ctx.env.update(time=time),
                ),
                init_cfg=_cfg,
                on_update=lambda ctx: viewer.update(
                    times, ts, cpmg_signal2real(np.asarray(ctx.root_data))
                ),
            )
            signals = np.asarray(signals)

        # record last cfg and result
        self.last_cfg = _cfg
        self.last_result = (times, ts, signals)

        return times, ts, signals

    def analyze(
        self, result: Optional[CPMG_Result] = None
    ) -> tuple[np.ndarray, np.ndarray, Figure]:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        times, Ts, signals2D = result

        real_signals2D = rotate2real(signals2D).real

        t2s = np.full(len(times), np.nan, dtype=np.float64)
        t2errs = np.zeros_like(t2s)

        fit_params = None
        for i in range(len(times)):
            real_signals = real_signals2D[i, :]

            # skip if have nan data
            if np.any(np.isnan(real_signals)):
                continue

            t2r, t2err, _, (pOpt, _) = fit_decay(
                Ts, real_signals, fit_params=fit_params
            )

            if t2err > 0.5 * t2r:
                continue

            fit_params = cast(tuple[float, float, float], tuple(pOpt))

            t2s[i] = t2r
            t2errs[i] = t2err

        if np.all(np.isnan(t2s)):
            raise ValueError("No valid Fitting T2 found. Please check the data.")

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        assert isinstance(fig, Figure)
        assert isinstance(ax1, Axes)
        assert isinstance(ax2, Axes)

        ax1.errorbar(times, t2s, yerr=t2errs, label="Fitting T2")
        ax1.set_ylabel("T2 (us)")
        ax1.set_ylim(bottom=0)
        ax1.grid()
        ax2.imshow(
            cpmg_signal2real(signals2D).T,
            aspect="auto",
            origin="lower",
            interpolation="none",
            extent=(times[0], times[-1], Ts[0], Ts[-1]),
        )
        ax2.set_ylabel("Time (us)")
        ax2.set_xlabel("Number of Pi")

        fig.tight_layout()

        return t2s, t2errs, fig

    def save(
        self,
        filepath: str,
        result: Optional[CPMG_Result] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/ge/cpmg",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        times, Ts, signals2D = result
        save_data(
            filepath=filepath,
            x_info={"name": "Number of pi", "unit": "a.u.", "values": times},
            y_info={"name": "Time", "unit": "s", "values": Ts * 1e-6},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals2D.T},
            comment=comment,
            tag=tag,
            **kwargs,
        )

    def load(self, filepath: str, **kwargs) -> CPMG_Result:
        signals2D, times, Ts = load_data(filepath, **kwargs)
        assert times is not None and Ts is not None
        assert len(times.shape) == 1 and len(Ts.shape) == 1
        assert signals2D.shape == (len(Ts), len(times))

        Ts = Ts * 1e6  # s -> us
        signals2D = signals2D.T  # transpose back

        times = times.astype(np.float64)
        Ts = Ts.astype(np.float64)
        signals2D = signals2D.astype(np.complex128)

        self.last_cfg = None
        self.last_result = (times, Ts, signals2D)

        return times, Ts, signals2D
