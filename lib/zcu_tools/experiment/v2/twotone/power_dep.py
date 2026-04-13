from __future__ import annotations

from copy import deepcopy

import numpy as np
from numpy.typing import NDArray
from typeguard import check_type
from typing_extensions import Any, Optional, TypeAlias, Callable, cast

from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.v2.runner import Task, TaskCfg, run_task, TaskState
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.liveplot import LivePlot2DwithLine
from zcu_tools.program import SweepCfg
from zcu_tools.program.v2 import Pulse, TwoToneCfg, TwoToneProgram, sweep2param
from zcu_tools.utils.datasaver import load_data, save_data
from zcu_tools.utils.process import minus_background

PowerResult: TypeAlias = tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.complex128]
]


def gain_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return np.abs(minus_background(signals, axis=1))


class PowerCfg(TwoToneCfg, TaskCfg):
    sweep: dict[str, SweepCfg]


class PowerExp(AbsExperiment[PowerResult, PowerCfg]):
    def run(
        self,
        soc,
        soccfg,
        cfg: dict[str, Any],
        *,
        acquire_kwargs: Optional[dict[str, Any]] = None,
    ) -> PowerResult:
        _cfg = check_type(deepcopy(cfg), PowerCfg)
        modules = _cfg["modules"]

        gain_sweep = _cfg["sweep"]["gain"]
        freq_sweep = _cfg["sweep"]["freq"]

        gains = sweep2array(
            gain_sweep,
            "gain",
            {"soccfg": soccfg, "gen_ch": modules["qub_pulse"].ch},
            allow_array=True,
        )
        freqs = sweep2array(
            freq_sweep,
            "freq",
            {"soccfg": soccfg, "gen_ch": modules["qub_pulse"].ch},
        )


        def measure_fn(ctx: TaskState, update_hook: Optional[Callable]) -> list[NDArray[np.float64]]:
            cfg = cast(PowerCfg, ctx.cfg)
            modules = cfg["modules"]

            freq_sweep = cfg["sweep"]["freq"]
            freq_param = sweep2param("freq", freq_sweep)
            modules["qub_pulse"].set_param("freq", freq_param)

            return TwoToneProgram(
                        soccfg,
                        cfg,
                        sweep=[
                            ("freq", freq_sweep)
                        ]
                    ).acquire(
                        soc,
                        progress=False,
                        callback=update_hook,
                        **(acquire_kwargs or {}),
                    )

        with LivePlot2DwithLine("Pulse Gain (a.u.)", "Frequency (MHz)", line_axis=1, num_lines=2) as viewer:
            signals = run_task(
                task=Task(
                    measure_fn=measure_fn,
                    result_shape=(len(freqs), ),
                ).scan(
                    "gain", gains.tolist(), before_each=lambda i, ctx, gain: ctx.cfg["modules"]["qub_pulse"].set_param("gain", gain)
                ),
                init_cfg=_cfg,
                on_update=lambda ctx: viewer.update(
                    gains, freqs, gain_signal2real(np.asarray(ctx.root_data))
                ),
            )
            signals = np.asarray(signals)

        # Cache results
        self.last_cfg = _cfg
        self.last_result = (gains, freqs, signals)

        return gains, freqs, signals

    def analyze(
        self,
        result: Optional[PowerResult] = None,
    ) -> None:
        raise NotImplementedError(
            "Analysis not implemented for two-tone power dependence"
        )

    def save(
        self,
        filepath: str,
        result: Optional[PowerResult] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/power_dep",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        gains, freqs, signals2D = result

        save_data(
            filepath=filepath,
            x_info={"name": "Frequency", "unit": "Hz", "values": freqs * 1e6},
            y_info={"name": "Power", "unit": "a.u.", "values": gains},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals2D},
            comment=comment,
            tag=tag,
            **kwargs,
        )

    def load(self, filepath: str, **kwargs) -> PowerResult:
        signals2D, freqs, gains = load_data(filepath, **kwargs)
        assert freqs is not None and gains is not None
        assert len(freqs.shape) == 1 and len(gains.shape) == 1
        assert signals2D.shape == (len(gains), len(freqs))

        freqs = freqs * 1e-6  # Hz -> MHz

        gains = gains.astype(np.float64)
        freqs = freqs.astype(np.float64)
        signals2D = signals2D.astype(np.complex128)

        self.last_cfg = None
        self.last_result = (gains, freqs, signals2D)

        return gains, freqs, signals2D
