from __future__ import annotations

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
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
    cast,
)

from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.utils import format_sweep1D
from zcu_tools.experiment.v2.runner import Task, TaskCfg, TaskState, run_task
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.liveplot import LivePlotter2DwithLine
from zcu_tools.program import SweepCfg
from zcu_tools.program.v2 import (
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

PowerDepOvernightResult: TypeAlias = tuple[
    NDArray[np.int64], NDArray[np.float64], NDArray[np.complex128]
]


def mist_overnight_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    avg_len = max(int(0.05 * signals.shape[1]), 1)

    mist_signals = signals - np.mean(signals[:, :avg_len], axis=1, keepdims=True)

    return np.abs(mist_signals)


class PowerDepOvernightModuleCfg(TypedDict, closed=True):
    reset: NotRequired[ResetCfg]
    init_pulse: NotRequired[PulseCfg]
    probe_pulse: PulseCfg
    readout: ReadoutCfg


class PowerDepOvernightCfg(ModularProgramCfg, TaskCfg):
    modules: PowerDepOvernightModuleCfg
    interval: float
    sweep: dict[str, SweepCfg]


class PowerDepOvernightExp(
    AbsExperiment[PowerDepOvernightResult, PowerDepOvernightCfg]
):
    def run(
        self, soc, soccfg, cfg: dict[str, Any], *, num_times=50, fail_retry=3
    ) -> PowerDepOvernightResult:
        cfg["sweep"] = format_sweep1D(cfg["sweep"], "gain")
        _cfg = check_type(deepcopy(cfg), PowerDepOvernightCfg)
        modules = _cfg["modules"]

        iters = np.arange(num_times, dtype=np.int64)
        gains = sweep2array(
            _cfg["sweep"]["gain"],
            "gain",
            {"soccfg": soccfg, "gen_ch": modules["probe_pulse"]["ch"]},
        )

        def measure_fn(
            ctx: TaskState[NDArray[np.complex128], Any],
            update_hook: Optional[Callable[[int, list[NDArray[np.float64]]], None]],
        ) -> list[NDArray[np.float64]]:
            cfg: PowerDepOvernightCfg = cast(PowerDepOvernightCfg, ctx.cfg)
            modules = cfg["modules"]

            assert update_hook is not None

            gain_sweep = cfg["sweep"]["gain"]
            gain_param = sweep2param("gain", gain_sweep)
            Pulse.set_param(modules["probe_pulse"], "gain", gain_param)

            return ModularProgramV2(
                soccfg,
                cfg,
                modules=[
                    Reset("reset", modules.get("reset")),
                    Pulse("init_pulse", modules.get("init_pulse")),
                    Pulse("probe_pulse", modules["probe_pulse"]),
                    Readout("readout", modules["readout"]),
                ],
                sweep=[("gain", gain_sweep)],
            ).acquire(soc, progress=False, callback=update_hook)

        with LivePlotter2DwithLine(
            "Pulse gain", "Iteration", line_axis=1, title="MIST Overnight"
        ) as viewer:
            signals = run_task(
                task=Task(measure_fn=measure_fn, result_shape=(len(gains),))
                .auto_retry(max_retries=fail_retry)
                .repeat("repeat_over_time", num_times, _cfg["interval"]),
                init_cfg=_cfg,
                on_update=lambda ctx: viewer.update(
                    iters.astype(np.float64),
                    gains,
                    mist_overnight_signal2real(np.asarray(ctx.root_data)),
                ),
            )
            signals = np.asarray(signals)

        # record the last result
        self.last_cfg = _cfg
        self.last_result = (iters, gains, signals)

        return iters, gains, signals

    def analyze(
        self,
        result: Optional[PowerDepOvernightResult] = None,
        *,
        g0=None,
        e0=None,
        ac_coeff=None,
    ) -> Figure:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        _, gains, signals = result

        if g0 is None:
            g0 = np.mean(signals[:, 0])

        abs_diff = np.abs(signals - g0)

        if ac_coeff is None:
            xs = gains
            xlabel = "probe gain (a.u.)"
        else:
            xs = ac_coeff * gains**2
            xlabel = r"$\bar n$"

        fig, ax = plt.subplots(figsize=config.figsize)
        assert isinstance(fig, Figure)

        ax.plot(xs, abs_diff.T)
        ax.set_xscale("log")
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel("Signal difference (a.u.)", fontsize=14)
        ax.grid(True)
        ax.tick_params(axis="both", which="major", labelsize=12)
        if e0 is not None:
            ax.set_ylim(0, 1.1 * np.abs(g0 - e0))

        fig.tight_layout()

        return fig

    def save(
        self,
        filepath: str,
        result: Optional[PowerDepOvernightResult] = None,
        comment: Optional[str] = None,
        tag: str = "mist/gain_overnight",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        iters, gains, overnight_signals = result

        save_data(
            filepath=filepath,
            x_info={"name": "Drive Power (a.u.)", "unit": "a.u.", "values": gains},
            y_info={"name": "Iteration", "unit": "None", "values": iters},
            z_info={"name": "Signal", "unit": "a.u.", "values": overnight_signals},
            comment=comment,
            tag=tag,
            **kwargs,
        )

    def load(self, filepath: str, **kwargs) -> PowerDepOvernightResult:
        overnight_signals, gains, iters = load_data(filepath, **kwargs)
        assert gains is not None and iters is not None
        assert len(gains.shape) == 1 and len(iters.shape) == 1
        assert overnight_signals.shape == (len(iters), len(gains))

        iters = iters.astype(np.int64)
        gains = gains.astype(np.float64)
        overnight_signals = overnight_signals.astype(np.complex128)

        self.last_cfg = None
        self.last_result = (iters, gains, overnight_signals)

        return iters, gains, overnight_signals
