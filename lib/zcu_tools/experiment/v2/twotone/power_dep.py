from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment import (
    IDENTITY,
    MHZ_TO_HZ,
    AxesSpec,
    Axis,
    PersistableExperiment,
    ZSpec,
    record_result,
    retrieve_result,
)
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import setup_devices
from zcu_tools.experiment.v2.runner import MeasureSession, TaskState
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.liveplot import LivePlot2DwithLine
from zcu_tools.program.v2 import (
    SweepCfg,
    TwoToneCfg,
    TwoToneProgram,
    sweep2param,
)
from zcu_tools.utils.process import minus_background


@dataclass(frozen=True)
class PowerResult:
    gains: NDArray[np.float64]
    freqs: NDArray[np.float64]
    signals: NDArray[np.complex128]
    cfg_snapshot: PowerCfg | None = None


def gain_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return np.abs(minus_background(signals, axis=1))


class PowerSweepCfg(ConfigBase):
    gain: SweepCfg
    freq: SweepCfg


class PowerCfg(TwoToneCfg, ExpCfgModel):
    sweep: PowerSweepCfg


class PowerExp(PersistableExperiment[PowerResult, PowerCfg]):
    # inner freqs stores MHz on disk (disk Hz) -> scale=MHZ_TO_HZ; outer gains -> IDENTITY
    AXES_SPEC = AxesSpec(
        axes=(
            Axis("freqs", "Frequency", "Hz", scale=MHZ_TO_HZ),
            Axis("gains", "Power", "a.u.", scale=IDENTITY),
        ),
        z=ZSpec("signals", "Signal", "a.u."),
        result_type=PowerResult,
        cfg_type=PowerCfg,
        tag="twotone/power_dep",
    )

    @record_result
    def run(
        self,
        soc,
        soccfg,
        cfg: PowerCfg,
        *,
        acquire_kwargs: dict[str, Any] | None = None,
    ) -> PowerResult:
        orig_cfg = deepcopy(cfg)
        setup_devices(cfg, progress=True)
        modules = cfg.modules

        gain_sweep = cfg.sweep.gain
        freq_sweep = cfg.sweep.freq

        gains = sweep2array(
            gain_sweep,
            "gain",
            {"soccfg": soccfg, "gen_ch": modules.qub_pulse.ch},
            allow_array=True,
        )
        freqs = sweep2array(
            freq_sweep,
            "freq",
            {"soccfg": soccfg, "gen_ch": modules.qub_pulse.ch},
        )

        def measure_fn(
            ctx: TaskState[NDArray[np.complex128], Any, PowerCfg],
            update_hook: Callable | None,
        ) -> list[NDArray[np.float64]]:
            cfg = ctx.cfg
            modules = cfg.modules

            freq_sweep = cfg.sweep.freq
            freq_param = sweep2param("freq", freq_sweep)
            modules.qub_pulse.set_param("freq", freq_param)

            return TwoToneProgram(soccfg, cfg, sweep=[("freq", freq_sweep)]).acquire(
                soc,
                progress=False,
                round_hook=update_hook,
                stop_checkers=[ctx.is_stop],
                **(acquire_kwargs or {}),
            )

        with LivePlot2DwithLine(
            "Pulse Gain (a.u.)", "Frequency (MHz)", line_axis=1, num_lines=2
        ) as viewer:
            with MeasureSession(cfg) as run:
                signals_buffer = run.buffer(
                    (len(gains), len(freqs)),
                    dtype=np.complex128,
                    on_update=lambda data: viewer.update(
                        gains, freqs, gain_signal2real(data)
                    ),
                )
                for step in run.scan("gain", gains.tolist()):
                    step.cfg.modules.qub_pulse.set_param("gain", step.value)
                    signals_buffer[step].measure(measure_fn, pbar_n=step.cfg.rounds)
                signals = signals_buffer.array

        return PowerResult(
            gains=gains, freqs=freqs, signals=signals, cfg_snapshot=orig_cfg
        )

    @retrieve_result
    def analyze(
        self,
        result: PowerResult | None = None,
    ) -> None:
        raise NotImplementedError(
            "Analysis not implemented for two-tone power dependence"
        )
