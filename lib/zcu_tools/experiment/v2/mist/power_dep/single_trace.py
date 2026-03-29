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
from zcu_tools.liveplot import LivePlotter1D
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

PowerDepResult: TypeAlias = tuple[NDArray[np.float64], NDArray[np.complex128]]


def mist_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    avg_len = max(int(0.05 * len(signals)), 1)

    mist_signals = signals - np.mean(signals[:avg_len])

    return np.abs(mist_signals)


class PowerDepModuleCfg(TypedDict, closed=True):
    reset: NotRequired[ResetCfg]
    init_pulse: NotRequired[PulseCfg]
    probe_pulse: PulseCfg
    readout: ReadoutCfg


class PowerDepCfg(ModularProgramCfg, TaskCfg):
    modules: PowerDepModuleCfg
    sweep: dict[str, SweepCfg]


class PowerDepExp(AbsExperiment[PowerDepResult, PowerDepCfg]):
    def run(self, soc, soccfg, cfg: dict[str, Any]) -> PowerDepResult:
        cfg["sweep"] = format_sweep1D(cfg["sweep"], "gain")
        _cfg = check_type(deepcopy(cfg), PowerDepCfg)
        modules = _cfg["modules"]

        gains = sweep2array(
            _cfg["sweep"]["gain"],
            "gain",
            {"soccfg": soccfg, "gen_ch": modules["probe_pulse"]["ch"]},
        )

        def measure_fn(
            ctx: TaskState[NDArray[np.complex128], Any],
            update_hook: Optional[Callable[[int, list[NDArray[np.float64]]], None]],
        ) -> list[NDArray[np.float64]]:
            cfg: PowerDepCfg = cast(PowerDepCfg, ctx.cfg)
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

        with LivePlotter1D("Pulse gain", "MIST") as viewer:
            signals = run_task(
                task=Task(measure_fn=measure_fn, result_shape=(len(gains),)),
                init_cfg=_cfg,
                on_update=lambda ctx: viewer.update(
                    gains, mist_signal2real(ctx.root_data)
                ),
            )

        # record the last result
        self.last_cfg = _cfg
        self.last_result = (gains, signals)

        return gains, signals

    def analyze(
        self,
        result: Optional[PowerDepResult] = None,
        *,
        g0=None,
        e0=None,
        ac_coeff=None,
    ) -> Figure:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        gains, signals = result

        if g0 is None:
            g0 = signals[0]

        amp_diff = np.abs(signals - g0)

        if ac_coeff is None:
            xs = gains
            xlabel = "probe gain (a.u.)"
        else:
            xs = ac_coeff * gains**2
            xlabel = r"$\bar n$"

        fig, ax = plt.subplots(figsize=config.figsize)
        ax.plot(xs, amp_diff.T, marker=".")
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel("Signal difference (a.u.)", fontsize=14)
        ax.grid(True)
        ax.tick_params(axis="both", which="major", labelsize=12)
        if e0 is not None:
            ax.set_ylim(0, 1.1 * np.abs(g0 - e0))

        return fig

    def save(
        self,
        filepath: str,
        result: Optional[PowerDepResult] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/mist/gain",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        gains, signals = result

        save_data(
            filepath=filepath,
            x_info={"name": "Drive Power (a.u.)", "unit": "a.u.", "values": gains},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals},
            comment=comment,
            tag=tag,
            **kwargs,
        )

    def load(self, filepath: str, **kwargs) -> PowerDepResult:
        signals, gains, _ = load_data(filepath, **kwargs)
        assert gains is not None
        assert len(gains.shape) == 1 and len(signals.shape) == 1
        assert gains.shape == signals.shape

        gains = gains.astype(np.float64)
        signals = signals.astype(np.complex128)

        self.last_cfg = None
        self.last_result = (gains, signals)

        return gains, signals
