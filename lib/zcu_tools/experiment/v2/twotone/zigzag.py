from __future__ import annotations

from copy import deepcopy

import numpy as np
from numpy.typing import NDArray
from typeguard import check_type
from typing_extensions import (
    Any,
    Dict,
    Literal,
    NotRequired,
    Optional,
    Sequence,
    Tuple,
    TypedDict,
    Union,
)

from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.utils import format_sweep1D, sweep2array
from zcu_tools.experiment.v2.runner import Task, TaskCfg, TaskState, run_task
from zcu_tools.liveplot import LivePlotter1D
from zcu_tools.program import SweepCfg
from zcu_tools.program.v2 import (
    ModularProgramCfg,
    ModularProgramV2,
    Pulse,
    PulseCfg,
    Readout,
    ReadoutCfg,
    Repeat,
    Reset,
    ResetCfg,
)
from zcu_tools.utils.datasaver import load_data, save_data
from zcu_tools.utils.process import rotate2real

# (times, signals)
ZigZagResult = Tuple[NDArray[np.float64], NDArray[np.complex128]]


def zigzag_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return rotate2real(signals).real  # type: ignore


class ZigZagModuleCfg(TypedDict, closed=True):
    reset: NotRequired[ResetCfg]
    X90_pulse: PulseCfg
    X180_pulse: NotRequired[PulseCfg]
    readout: ReadoutCfg


class ZigZagSweepCfg(TypedDict, closed=True):
    times: Union[SweepCfg, NDArray]


class ZigZagCfg(ModularProgramCfg, TaskCfg):
    modules: ZigZagModuleCfg
    sweep: ZigZagSweepCfg


class ZigZagExp(AbsExperiment[ZigZagResult, ZigZagCfg]):
    def run(
        self,
        soc,
        soccfg,
        cfg: Dict[str, Any],
        *,
        repeat_on: Literal["X90_pulse", "X180_pulse"] = "X180_pulse",
    ) -> ZigZagResult:
        cfg["sweep"] = format_sweep1D(cfg["sweep"], "times")
        _cfg = check_type(deepcopy(cfg), ZigZagCfg)
        modules = _cfg["modules"]

        X90_pulse = deepcopy(modules["X90_pulse"])

        times = sweep2array(_cfg["sweep"]["times"], allow_array=True)  # predicted

        del _cfg["sweep"]["times"]  # type: ignore

        with LivePlotter1D(
            "Times", "Signal", segment_kwargs=dict(show_grid=True)
        ) as viewer:

            def measure_fn(ctx: TaskState, update_hook):
                modules = ctx.cfg["modules"]
                zigzag_time = ctx.env["zigzag_time"]
                if repeat_on == "X90_pulse":
                    repeat_time = 2 * zigzag_time
                else:
                    repeat_time = zigzag_time

                return ModularProgramV2(
                    soccfg,
                    ctx.cfg,
                    modules=[
                        Reset("reset", modules.get("reset")),
                        Pulse(name="X90_pulse", cfg=X90_pulse),
                        Repeat(
                            name="zigzag_loop",
                            n=repeat_time,
                            sub_module=Pulse(
                                name=f"loop_{repeat_on}",
                                cfg=modules[repeat_on],
                            ),
                        ),
                        Readout("readout", modules["readout"]),
                    ],
                ).acquire(soc, progress=False, callback=update_hook)

            signals = run_task(
                task=Task(measure_fn=measure_fn).scan(
                    "times",
                    times.tolist(),
                    before_each=lambda _, ctx, time: ctx.env.update(zigzag_time=time),
                ),
                init_cfg=_cfg,
                on_update=lambda ctx: viewer.update(
                    times, zigzag_signal2real(np.asarray(ctx.root_data))
                ),
            )
            signals = np.asarray(signals)

        # record last cfg and result
        self.last_cfg = _cfg
        self.last_result = (times, signals)

        return times, signals

    def analyze(
        self,
        result: Optional[ZigZagResult] = None,
    ) -> Tuple[float, float]:
        raise NotImplementedError("Not implemented")

    def save(
        self,
        filepath: str,
        result: Optional[ZigZagResult] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/ge/zigzag",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        times, signals = result
        save_data(
            filepath=filepath,
            x_info={"name": "Times", "unit": "a.u.", "values": times},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals},
            comment=comment,
            tag=tag,
            **kwargs,
        )

    def load(self, filepath: str, **kwargs) -> ZigZagResult:
        signals, times, _ = load_data(filepath, **kwargs)
        assert times is not None
        assert len(times.shape) == 1 and len(signals.shape) == 1
        assert times.shape == signals.shape

        times = times.astype(np.float64)
        signals = signals.astype(np.complex128)

        self.last_cfg = None
        self.last_result = (times, signals)

        return times, signals
