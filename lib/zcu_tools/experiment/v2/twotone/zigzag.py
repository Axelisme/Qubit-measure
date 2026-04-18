from __future__ import annotations

from copy import deepcopy

import numpy as np
from numpy.typing import NDArray
from typeguard import check_type
from typing_extensions import (
    Any,
    Callable,
    Literal,
    NotRequired,
    Optional,
    TypeAlias,
    TypedDict,
    cast,
)

from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.utils import setup_devices
from zcu_tools.experiment.v2.runner import Task, TaskCfg, TaskState, run_task
from zcu_tools.liveplot import LivePlot1D
from zcu_tools.program.v2 import (
    LoadValue,
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
ZigZagResult: TypeAlias = tuple[NDArray[np.int64], NDArray[np.complex128]]


def zigzag_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return rotate2real(signals).real  # type: ignore


class ZigZagModuleCfg(TypedDict, closed=True):
    reset: NotRequired[ResetCfg]
    X90_pulse: PulseCfg
    X180_pulse: NotRequired[PulseCfg]
    readout: ReadoutCfg


class ZigZagCfg(ModularProgramCfg, TaskCfg):
    modules: ZigZagModuleCfg
    n_times: int


class ZigZagExp(AbsExperiment[ZigZagResult, ZigZagCfg]):
    def run(
        self,
        soc,
        soccfg,
        cfg: dict[str, Any],
        *,
        repeat_on: Literal["X90_pulse", "X180_pulse"] = "X180_pulse",
        acquire_kwargs: Optional[dict[str, Any]] = None,
    ) -> ZigZagResult:
        _cfg = check_type(deepcopy(cfg), ZigZagCfg)
        setup_devices(_cfg, progress=True)

        times = np.arange(_cfg["n_times"])
        loop_n = list(2 * times if repeat_on == "X90_pulse" else times)

        def measure_fn(
            ctx: TaskState, update_hook: Optional[Callable]
        ) -> list[NDArray[np.float64]]:
            cfg = cast(ZigZagCfg, ctx.cfg)
            modules = cfg["modules"]

            X90_pulse = deepcopy(modules["X90_pulse"])
            repeat_pulse = modules.get(repeat_on)
            if repeat_pulse is None:
                raise ValueError(f"Repeat on pulse {repeat_on} not found")

            return ModularProgramV2(
                soccfg,
                cfg,
                modules=[
                    Reset("reset", modules.get("reset")),
                    Pulse("X90_pulse", X90_pulse),
                    LoadValue(
                        "load_repeat_count",
                        values=loop_n,
                        idx_reg="times",
                        val_reg="repeat_count",
                    ),
                    Repeat("zigzag_loop", n="repeat_count").add_content(
                        Pulse(f"loop_{repeat_on}", repeat_pulse)
                    ),
                    Readout("readout", modules["readout"]),
                ],
                sweep=[("times", len(times))],
            ).acquire(
                soc, progress=False, callback=update_hook, **(acquire_kwargs or {})
            )

        with LivePlot1D(
            "Times", "Signal", segment_kwargs=dict(show_grid=True)
        ) as viewer:
            signals = run_task(
                task=Task(
                    measure_fn=measure_fn,
                    result_shape=(len(times),),
                    pbar_n=_cfg["rounds"],
                ),
                init_cfg=_cfg,
                on_update=lambda ctx: viewer.update(
                    times.astype(np.float64),
                    zigzag_signal2real(ctx.root_data),
                ),
            )
            signals = np.asarray(signals, dtype=np.complex128)

        # record last cfg and result
        self.last_cfg = _cfg
        self.last_result = (times, signals)

        return times, signals

    def analyze(self, _result: Optional[ZigZagResult] = None) -> tuple[float, float]:
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

        times = times.astype(np.float64)

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

        times = times.astype(np.int64)
        signals = signals.astype(np.complex128)

        self.last_cfg = None
        self.last_result = (times, signals)

        return times, signals
