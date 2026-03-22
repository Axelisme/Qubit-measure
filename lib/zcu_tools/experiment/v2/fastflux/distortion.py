from __future__ import annotations

from copy import deepcopy

import numpy as np
from numpy.typing import NDArray
from typeguard import check_type
from typing_extensions import Any, Callable, NotRequired, Optional, TypeAlias, TypedDict

from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.utils import sweep2array
from zcu_tools.experiment.v2.runner import Task, TaskCfg, TaskState, run_task
from zcu_tools.liveplot import LivePlotter2D
from zcu_tools.program import SweepCfg
from zcu_tools.program.v2 import (
    ModularProgramCfg,
    ModularProgramV2,
    NonBlocking,
    Pulse,
    PulseCfg,
    Readout,
    ReadoutCfg,
    Reset,
    ResetCfg,
    SoftDelay,
    sweep2param,
)
from zcu_tools.utils.datasaver import load_data, save_data
from zcu_tools.utils.process import rotate2real

# (amps, freqs, signals2D)
DistortionResult: TypeAlias = tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.complex128]
]


def distortion_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return rotate2real(signals).real


class DistortionModuleCfg(TypedDict, closed=True):
    reset: NotRequired[ResetCfg]
    flux_pulse: PulseCfg
    pi2_pulse: PulseCfg
    readout: ReadoutCfg


class DistortionCfg(ModularProgramCfg, TaskCfg):
    modules: DistortionModuleCfg
    pi2_interval: float
    sweep: dict[str, SweepCfg]


class DistortionExp(AbsExperiment):
    def run(
        self, soc, soccfg, cfg: dict[str, Any], *, detune: float = 0.0
    ) -> DistortionResult:
        _cfg = check_type(deepcopy(cfg), DistortionCfg)

        # uniform in square space
        gains = sweep2array(_cfg["sweep"]["gain"])
        lengths = sweep2array(_cfg["sweep"]["length"])

        gain_params = sweep2param("gain", _cfg["sweep"]["gain"])
        wait_params = sweep2param("length", _cfg["sweep"]["length"])

        Pulse.set_param(_cfg["modules"]["flux_pulse"], "gain", gain_params)

        with LivePlotter2D("Flux Pulse Gain (a.u.)", "Time (us)") as viewer:

            def measure_fn(ctx: TaskState, update_hook: Optional[Callable]):
                modules = ctx.cfg["modules"]
                return ModularProgramV2(
                    soccfg,
                    ctx.cfg,
                    modules=[
                        Reset("reset", modules.get("reset")),
                        NonBlocking(
                            [
                                SoftDelay("wait_time", delay=wait_params),
                                Pulse("pi2_pulse1", modules["pi2_pulse"]),
                                SoftDelay("pi2_interval", ctx.cfg["pi2_interval"]),
                                Pulse(
                                    name="pi2_pulse2",
                                    cfg={  # type: ignore[dict-item]
                                        **modules["pi2_pulse"],
                                        "phase": modules["pi2_pulse"]["phase"]
                                        + 360 * detune * wait_params,
                                    },
                                ),
                            ]
                        ),
                        Pulse("flux_pulse", modules["flux_pulse"]),
                        Readout("readout", modules["readout"]),
                    ],
                ).acquire(soc, progress=False, callback=update_hook)

            signals = run_task(
                task=Task(
                    measure_fn=measure_fn, result_shape=(len(gains), len(lengths))
                ),
                init_cfg=_cfg,
                on_update=lambda ctx: viewer.update(
                    gains, lengths, distortion_signal2real(ctx.root_data)
                ),
            )

        # Cache results
        self.last_cfg = _cfg
        self.last_result = (gains, lengths, signals)

        return gains, lengths, signals

    def analyze(
        self,
        result: Optional[DistortionResult] = None,
        *,
        detune: float = 0.0,
        cutoff: Optional[float] = None,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "No result found"

        gains, lengths, signals = result

        # apply cutoff if provided
        if cutoff is not None:
            valid_indices = np.where(gains < cutoff)[0]
            gains = gains[valid_indices]
            signals = signals[valid_indices, :]

        raise NotImplementedError

    def save(
        self,
        filepath: str,
        result: Optional[DistortionResult] = None,
        comment: Optional[str] = None,
        tag: str = "fastflux/distortion",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "No result found"

        gains, lengths, signals2D = result

        save_data(
            filepath=filepath,
            x_info={"name": "Flux Pulse Gain", "unit": "a.u.", "values": gains},
            y_info={"name": "Wait Time", "unit": "s", "values": lengths * 1e-6},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals2D.T},
            comment=comment,
            tag=tag,
            **kwargs,
        )

    def load(self, filepath: str, **kwargs) -> DistortionResult:
        signals2D, gains, lengths = load_data(filepath, **kwargs)
        assert gains is not None and lengths is not None
        assert len(gains.shape) == 1 and len(lengths.shape) == 1
        assert signals2D.shape == (len(lengths), len(gains))

        lengths = lengths * 1e6  # s -> us
        signals2D = signals2D.T  # transpose back

        gains = gains.astype(np.float64)
        lengths = lengths.astype(np.float64)
        signals2D = signals2D.astype(np.complex128)

        self.last_cfg = None
        self.last_result = (gains, lengths, signals2D)

        return gains, lengths, signals2D
