from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment import (
    IDENTITY,
    AxesSpec,
    Axis,
    PersistableExperiment,
    ZSpec,
    record_result,
    retrieve_result,
)
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import setup_devices
from zcu_tools.experiment.v2.runner import Schedule, SignalBuffer
from zcu_tools.liveplot import LivePlot1D
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
    cfg_snapshot: ZigZagCfg | None = None


def zigzag_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return rotate2real(signals).real  # type: ignore


class ZigZagModuleCfg(ConfigBase):
    reset: ResetCfg | None = None
    X90_pulse: PulseCfg
    X180_pulse: PulseCfg | None = None
    readout: ReadoutCfg


class ZigZagCfg(ProgramV2Cfg, ExpCfgModel):
    modules: ZigZagModuleCfg
    n_times: int


class ZigZagExp(PersistableExperiment[ZigZagResult, ZigZagCfg]):
    # times is an int counter (a.u.) -> Axis dtype int64, scale IDENTITY
    AXES_SPEC = AxesSpec(
        axes=(Axis("times", "Times", "a.u.", IDENTITY, np.int64),),
        z=ZSpec("signals", "Signal", "a.u."),
        result_type=ZigZagResult,
        cfg_type=ZigZagCfg,
        tag="twotone/ge/zigzag",
    )

    @record_result
    def run(
        self,
        soc,
        soccfg,
        cfg: ZigZagCfg,
        *,
        repeat_on: Literal["X90_pulse", "X180_pulse"] = "X180_pulse",
        acquire_kwargs: dict[str, Any] | None = None,
    ) -> ZigZagResult:
        orig_cfg = deepcopy(cfg)

        setup_devices(cfg, progress=True)

        times = np.arange(0, cfg.n_times + 1)
        # Convert to plain int list: LoadValue.values expects Sequence[int], and
        # numpy 2.x scalar types (int_) are not considered int by pyright.
        loop_n: list[int] = [
            int(x) for x in (2 * times if repeat_on == "X90_pulse" else times)
        ]

        with LivePlot1D(
            "Times", "Signal", segment_kwargs=dict(show_grid=True)
        ) as viewer:
            signals_buffer = SignalBuffer(
                (len(times),),
                on_update=lambda data: viewer.update(
                    times.astype(np.float64),
                    zigzag_signal2real(data),
                ),
            )
            with Schedule(cfg, signals_buffer) as sched:
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
                    .build_and_acquire(
                        **(acquire_kwargs or {}),
                    )
                )
                signals = signals_buffer.array

        return ZigZagResult(times=times, signals=signals, cfg_snapshot=orig_cfg)

    @retrieve_result
    def analyze(self, result: ZigZagResult | None = None) -> tuple[float, float]:
        raise NotImplementedError("Not implemented")
