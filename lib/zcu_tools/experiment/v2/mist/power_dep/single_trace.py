from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment import (
    AxesSpec,
    Axis,
    PersistableExperiment,
    ZSpec,
    config,
    record_result,
    retrieve_result,
)
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import setup_devices
from zcu_tools.experiment.v2.runner import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.liveplot import LivePlot1D
from zcu_tools.program.v2 import (
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


@dataclass(frozen=True)
class PowerDepResult:
    gains: NDArray[np.float64]
    signals: NDArray[np.complex128]
    cfg_snapshot: PowerDepCfg | None = None


def mist_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    avg_len = max(int(0.05 * len(signals)), 1)

    mist_signals = signals - np.mean(signals[:avg_len])

    return np.abs(mist_signals)


class PowerDepModuleCfg(ConfigBase):
    reset: ResetCfg | None = None
    init_pulse: PulseCfg | None = None
    probe_pulse: PulseCfg
    readout: ReadoutCfg


class PowerDepSweepCfg(ConfigBase):
    gain: SweepCfg


class PowerDepCfg(ProgramV2Cfg, ExpCfgModel):
    modules: PowerDepModuleCfg
    sweep: PowerDepSweepCfg


class PowerDepExp(PersistableExperiment[PowerDepResult, PowerDepCfg]):
    AXES_SPEC = AxesSpec(
        axes=(Axis("gains", "Drive Power", "a.u."),),
        z=ZSpec("signals", "Signal", "a.u."),
        result_type=PowerDepResult,
        cfg_type=PowerDepCfg,
        tag="mist/gain",
    )

    @record_result
    def run(
        self,
        soc,
        soccfg,
        cfg: PowerDepCfg,
        *,
        acquire_kwargs: dict[str, Any] | None = None,
    ) -> PowerDepResult:
        orig_cfg = deepcopy(cfg)
        setup_devices(cfg, progress=True)
        modules = cfg.modules

        gains = sweep2array(
            cfg.sweep.gain,
            "gain",
            {"soccfg": soccfg, "gen_ch": modules.probe_pulse.ch},
        )

        with LivePlot1D("Pulse gain", "MIST") as viewer:
            signals_buffer = SignalBuffer(
                (len(gains),),
                on_update=lambda data: viewer.update(gains, mist_signal2real(data)),
            )
            with Schedule(cfg, signals_buffer) as sched:
                modules = sched.cfg.modules
                modules.probe_pulse.set_param(
                    "gain", sweep2param("gain", sched.cfg.sweep.gain)
                )
                _ = (
                    sched.prog_builder(soc, soccfg)
                    .add(
                        Reset("reset", modules.reset),
                        Pulse("init_pulse", modules.init_pulse),
                        Pulse("probe_pulse", modules.probe_pulse),
                        Readout("readout", modules.readout),
                    )
                    .declare_sweep("gain", sched.cfg.sweep.gain)
                    .build_and_acquire(
                        **(acquire_kwargs or {}),
                    )
                )
                signals = signals_buffer.array

        return PowerDepResult(gains=gains, signals=signals, cfg_snapshot=orig_cfg)

    @retrieve_result
    def analyze(
        self,
        result: PowerDepResult | None = None,
        *,
        g0=None,
        e0=None,
        ac_coeff=None,
    ) -> Figure:
        assert result is not None, "no result found"

        gains, signals = result.gains, result.signals

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
