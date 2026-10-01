from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment import (
    IDENTITY,
    AxesSpec,
    Axis,
    PersistableExperiment,
    ZSpec,
)
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.utils import setup_devices
from zcu_tools.experiment.v2.runtime import Schedule, SignalBuffer
from zcu_tools.program.v2 import (
    LoadValue,
    ProgramV2Cfg,
    Pulse,
    PulseCfg,
    Readout,
    ReadoutCfg,
    Repeat,
    Reset,
    ResetCfg,
)
from zcu_tools.utils.process import rotate2real


@dataclass(frozen=True)
class ZigZagResult:
    times: NDArray[np.int64]
    signals: NDArray[np.complex128]


def zigzag_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return rotate2real(signals).real


class ZigZagModuleCfg(ConfigBase):
    reset: ResetCfg | None = None
    X90_pulse: PulseCfg
    X180_pulse: PulseCfg | None = None
    readout: ReadoutCfg


class ZigZagCfg(ProgramV2Cfg, ExpCfgModel):
    modules: ZigZagModuleCfg
    n_times: int
    repeat_on: Literal["X90_pulse", "X180_pulse"] = "X180_pulse"


class ZigZagExp(PersistableExperiment[ZigZagResult, ZigZagCfg]):
    # times is an int counter (a.u.) -> Axis dtype int64, scale IDENTITY
    AXES_SPEC = AxesSpec(
        axes=(Axis("times", "Times", "a.u.", IDENTITY, np.int64),),
        z=ZSpec("signals", "Signal", "a.u."),
        result_type=ZigZagResult,
        cfg_type=ZigZagCfg,
        tag="twotone/ge/zigzag",
    )

    def run(self, cfg: ZigZagCfg, *, context: RunContext) -> ZigZagResult:
        cfg = deepcopy(cfg)
        soc, soccfg = context.soc, context.soccfg
        repeat_on = cfg.repeat_on

        setup_devices(
            cfg,
            context.devices,
            progress=True,
            cancel_signal=context.cancel_signal.event,
        )

        times = np.arange(0, cfg.n_times + 1)
        # Convert to plain int list: LoadValue.values expects Sequence[int], and
        # numpy 2.x scalar types (int_) are not considered int by pyright.
        loop_n: list[int] = [
            int(x) for x in (2 * times if repeat_on == "X90_pulse" else times)
        ]

        viewer = context.plots.liveplot_1d("measurement", "Times", "Signal")
        signals_buffer = SignalBuffer(
            (len(times),),
            on_update=lambda data: viewer.update(
                times.astype(np.float64),
                zigzag_signal2real(data),
            ),
        )
        with Schedule(cfg, signals_buffer, stop=context.cancel_signal) as sched:
            modules = sched.cfg.modules
            X90_pulse = deepcopy(modules.X90_pulse)
            repeat_pulse = getattr(modules, repeat_on)
            if repeat_pulse is None:
                raise ValueError(f"Repeat on pulse {repeat_on} not found")

            _ = (
                sched.prog_builder(soc, soccfg)
                .add(
                    LoadValue(
                        "load_repeat_count",
                        values=loop_n,
                        idx_reg="times",
                        val_reg="repeat_count",
                    ),
                    Reset("reset", cfg=modules.reset),
                    Pulse("X90_pulse", cfg=X90_pulse),
                    Repeat(
                        "zigzag_loop",
                        n="repeat_count",
                        # int() cast: numpy scalar types are not plain int to pyright.
                        range_hint=(int(min(times)), int(max(times))),
                    ).add_content(Pulse(f"loop_{repeat_on}", cfg=repeat_pulse)),
                    Readout("readout", cfg=modules.readout),
                )
                .declare_sweep("times", len(times))
                .build_and_acquire()
            )

        return ZigZagResult(times=times, signals=signals_buffer.array)
