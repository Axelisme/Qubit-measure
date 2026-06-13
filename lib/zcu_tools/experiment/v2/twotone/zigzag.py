from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import make_comment, parse_comment, setup_devices
from zcu_tools.experiment.v2.runner import Task, TaskState, run_task
from zcu_tools.liveplot import LivePlot1D
from zcu_tools.program.v2 import (
    LoadValue,
    ModularProgramV2,
    ProgramV2Cfg,
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


class ZigZagExp(AbsExperiment[ZigZagResult, ZigZagCfg]):
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

        def measure_fn(
            ctx: TaskState[NDArray[np.complex128], Any, ZigZagCfg],
            update_hook: Callable | None,
        ) -> list[NDArray[np.float64]]:
            cfg = ctx.cfg
            modules = cfg.modules

            X90_pulse = deepcopy(modules.X90_pulse)
            repeat_pulse = getattr(modules, repeat_on)
            if repeat_pulse is None:
                raise ValueError(f"Repeat on pulse {repeat_on} not found")

            return ModularProgramV2(
                soccfg,
                cfg,
                modules=[
                    LoadValue(
                        "load_repeat_count",
                        values=loop_n,
                        idx_reg="times",
                        val_reg="repeat_count",
                    ),
                    Reset("reset", modules.reset),
                    Pulse("X90_pulse", X90_pulse),
                    Repeat(
                        "zigzag_loop",
                        n="repeat_count",
                        # int() cast: numpy scalar types are not plain int to pyright.
                        range_hint=(int(min(times)), int(max(times))),
                    ).add_content(Pulse(f"loop_{repeat_on}", repeat_pulse)),
                    Readout("readout", modules.readout),
                ],
                sweep=[("times", len(times))],
            ).acquire(
                soc,
                progress=False,
                round_hook=update_hook,
                stop_checkers=[ctx.is_stop],
                **(acquire_kwargs or {}),
            )

        with LivePlot1D(
            "Times", "Signal", segment_kwargs=dict(show_grid=True)
        ) as viewer:
            signals = run_task(
                task=Task(
                    measure_fn=measure_fn,
                    result_shape=(len(times),),
                    pbar_n=cfg.rounds,
                ),
                init_cfg=cfg,
                on_update=lambda ctx: viewer.update(
                    times.astype(np.float64),
                    zigzag_signal2real(ctx.root_data),
                ),
            )
            signals = np.asarray(signals, dtype=np.complex128)

        # record result
        self.last_result = ZigZagResult(
            times=times, signals=signals, cfg_snapshot=orig_cfg
        )

        return self.last_result

    def analyze(self, _result: ZigZagResult | None = None) -> tuple[float, float]:
        raise NotImplementedError("Not implemented")

    def save(
        self,
        filepath: str,
        result: ZigZagResult | None = None,
        comment: str | None = None,
        tag: str = "twotone/ge/zigzag",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        cfg = result.cfg_snapshot
        if cfg is None:
            raise ValueError("cfg_snapshot is None")

        times = result.times.astype(np.float64)
        signals = result.signals
        comment = make_comment(cfg, comment)

        save_data(
            filepath=filepath,
            x_info={"name": "Times", "unit": "a.u.", "values": times},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals},
            comment=comment,
            tag=tag,
            **kwargs,
        )

    def load(self, filepath: str, **kwargs) -> ZigZagResult:
        signals, times, _, comment = load_data(filepath, return_comment=True, **kwargs)
        assert times is not None
        assert len(times.shape) == 1 and len(signals.shape) == 1
        assert times.shape == signals.shape

        times = times.astype(np.int64)
        signals = signals.astype(np.complex128)

        cfg_snapshot = None
        if comment is not None:
            cfg, _, _ = parse_comment(comment)

            if cfg is not None:
                cfg_snapshot = ZigZagCfg.validate_or_warn(cfg, source=filepath)
        self.last_result = ZigZagResult(
            times=times, signals=signals, cfg_snapshot=cfg_snapshot
        )

        return self.last_result
