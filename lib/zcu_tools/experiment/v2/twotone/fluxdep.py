from __future__ import annotations

from copy import deepcopy

import numpy as np
from numpy.typing import NDArray
from typeguard import check_type
from typing_extensions import Any, Mapping, Optional, TypeAlias, cast

from zcu_tools.device import DeviceInfo
from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.utils import set_flux_in_dev_cfg
from zcu_tools.experiment.v2.runner import Task, TaskCfg, run_task
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.liveplot import LivePlotter2DwithLine
from zcu_tools.notebook.analysis.fluxdep.interactive import (
    InteractiveFindPoints,
    InteractiveLines,
)
from zcu_tools.program import SweepCfg
from zcu_tools.program.v2 import Pulse, TwoToneCfg, TwoToneProgram, sweep2param
from zcu_tools.utils.datasaver import load_data, save_data
from zcu_tools.utils.process import minus_background

FreqFluxResult: TypeAlias = tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.complex128]
]


def freqflux_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return np.abs(minus_background(signals, axis=1))


class FreqFluxCfg(TwoToneCfg, TaskCfg):
    dev: Mapping[str, DeviceInfo]
    sweep: dict[str, SweepCfg]


class FreqFluxExp(AbsExperiment[FreqFluxResult, FreqFluxCfg]):
    def run(
        self, soc, soccfg, cfg: dict[str, Any], fail_retry: int = 0
    ) -> FreqFluxResult:
        _cfg = check_type(deepcopy(cfg), FreqFluxCfg)
        modules = _cfg["modules"]

        value_sweep = _cfg["sweep"]["flux"]
        freq_sweep = _cfg["sweep"]["freq"]

        dev_values = sweep2array(value_sweep, allow_array=True)
        freqs = sweep2array(
            freq_sweep, "freq", {"soccfg": soccfg, "gen_ch": modules["qub_pulse"]["ch"]}
        )

        # Frequency is swept by FPGA (hard sweep)
        freq_param = sweep2param("freq", freq_sweep)
        Pulse.set_param(modules["qub_pulse"], "freq", freq_param)

        with LivePlotter2DwithLine(
            "Flux device value", "Frequency (MHz)", line_axis=1, num_lines=2
        ) as viewer:
            signals = run_task(
                task=Task(
                    measure_fn=lambda ctx, update_hook: TwoToneProgram(
                        soccfg,
                        ctx.cfg,
                        sweep=[("freq", ctx.cfg["sweep"]["freq"])],
                    ).acquire(soc, progress=False, callback=update_hook),
                    result_shape=(len(freqs),),
                )
                .auto_retry(max_retries=fail_retry)
                .scan(
                    "flux",
                    dev_values.tolist(),
                    before_each=lambda _, ctx, flux: set_flux_in_dev_cfg(
                        ctx.cfg["dev"], flux
                    ),
                ),
                init_cfg=_cfg,
                on_update=lambda ctx: viewer.update(
                    dev_values, freqs, freqflux_signal2real(np.asarray(ctx.root_data))
                ),
            )
            signals = np.asarray(signals)

        # Cache results
        self.last_cfg = _cfg
        self.last_result = (dev_values, freqs, signals)

        return dev_values, freqs, signals

    def analyze(
        self,
        result: Optional[FreqFluxResult] = None,
        flux_half: Optional[float] = None,
        flux_int: Optional[float] = None,
    ) -> InteractiveLines:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        values, freqs, signals2D = result

        signals2D = minus_background(signals2D, axis=1)

        actline = InteractiveLines(
            signals2D, values, freqs, flux_half=flux_half, flux_int=flux_int
        )

        return actline

    def extract_points(
        self,
        result: Optional[FreqFluxResult] = None,
    ) -> InteractiveFindPoints:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        values, freqs, signals2D = result

        point_selector = InteractiveFindPoints(signals2D, values, freqs)

        return point_selector

    def save(
        self,
        filepath: str,
        result: Optional[FreqFluxResult] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/flux_dep/freq",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        values, freqs, signals2D = result

        save_data(
            filepath=filepath,
            x_info={"name": "Flux device value", "unit": "a.u.", "values": values},
            y_info={"name": "Frequency", "unit": "Hz", "values": freqs * 1e6},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals2D.T},
            comment=comment,
            tag=tag,
            **kwargs,
        )

    def load(self, filepath: str, **kwargs) -> FreqFluxResult:
        signals2D, values, freqs, cfg = load_data(filepath, return_cfg=True, **kwargs)
        assert values is not None and freqs is not None
        assert len(values.shape) == 1 and len(freqs.shape) == 1
        assert signals2D.shape == (len(values), len(freqs))

        freqs = freqs * 1e-6  # Hz -> MHz

        values = values.astype(np.float64)
        freqs = freqs.astype(np.float64)
        signals2D = signals2D.astype(np.complex128)

        self.last_cfg = cast(FreqFluxCfg, cfg)
        self.last_result = (values, freqs, signals2D)

        return values, freqs, signals2D
