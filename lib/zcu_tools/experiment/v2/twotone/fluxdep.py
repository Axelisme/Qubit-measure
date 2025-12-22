from __future__ import annotations

from copy import deepcopy

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Mapping, Optional, Tuple

from zcu_tools.device import DeviceInfo
from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.utils import set_flux_in_dev_cfg, sweep2array
from zcu_tools.experiment.v2.runner import (
    HardTask,
    ReTryIfFail,
    SoftTask,
    TaskConfig,
    run_task,
)
from zcu_tools.liveplot import LivePlotter2DwithLine
from zcu_tools.notebook.analysis.fluxdep.interactive import (
    InteractiveFindPoints,
    InteractiveLines,
)
from zcu_tools.program.v2 import TwoToneProgram, TwoToneProgramCfg, sweep2param
from zcu_tools.utils.datasaver import load_data, save_data
from zcu_tools.utils.process import minus_background

FreqFluxDepResultType = Tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.complex128]
]


def freq_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return np.abs(minus_background(signals, axis=1))


class FreqFluxDepTaskConfig(TaskConfig, TwoToneProgramCfg):
    dev: Mapping[str, DeviceInfo]


class FreqFluxDepExp(AbsExperiment[FreqFluxDepResultType, FreqFluxDepTaskConfig]):
    def run(
        self, soc, soccfg, cfg: FreqFluxDepTaskConfig, fail_retry: int = 0
    ) -> FreqFluxDepResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        assert "sweep" in cfg
        flx_sweep = cfg["sweep"]["flux"]
        fpt_sweep = cfg["sweep"]["freq"]

        # Remove flux from sweep dict - will be handled by soft loop
        cfg["sweep"] = {"freq": fpt_sweep}

        dev_values = sweep2array(flx_sweep, allow_array=True)
        fpts = sweep2array(fpt_sweep)  # predicted frequency points

        # Frequency is swept by FPGA (hard sweep)
        cfg["qub_pulse"]["freq"] = sweep2param("freq", fpt_sweep)

        with LivePlotter2DwithLine(
            "Flux device value", "Frequency (MHz)", line_axis=1, num_lines=2
        ) as viewer:
            signals = run_task(
                task=SoftTask(
                    sweep_name="flux",
                    sweep_values=dev_values.tolist(),
                    update_cfg_fn=lambda _, ctx, flx: (
                        set_flux_in_dev_cfg(ctx.cfg["dev"], flx)
                    ),
                    sub_task=ReTryIfFail(
                        max_retries=fail_retry,
                        task=HardTask(
                            measure_fn=lambda ctx, update_hook: (
                                TwoToneProgram(soccfg, ctx.cfg).acquire(
                                    soc, progress=False, callback=update_hook
                                )
                            ),
                            result_shape=(len(fpts),),
                        ),
                    ),
                ),
                init_cfg=cfg,
                update_hook=lambda ctx: viewer.update(
                    dev_values, fpts, freq_signal2real(np.asarray(ctx.data))
                ),
            )
            signals = np.asarray(signals)

        # Cache results
        self.last_cfg = cfg
        self.last_result = (dev_values, fpts, signals)

        return dev_values, fpts, signals

    def analyze(
        self,
        result: Optional[FreqFluxDepResultType] = None,
        mA_c: Optional[float] = None,
        mA_e: Optional[float] = None,
    ) -> InteractiveLines:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        values, fpts, signals2D = result

        signals2D = minus_background(signals2D, axis=1)

        actline = InteractiveLines(
            signals2D, mAs=values, fpts=fpts, mA_c=mA_c, mA_e=mA_e
        )

        return actline

    def extract_points(
        self,
        result: Optional[FreqFluxDepResultType] = None,
    ) -> InteractiveFindPoints:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        values, fpts, signals2D = result

        point_selector = InteractiveFindPoints(signals2D.T, mAs=values, fpts=fpts)

        return point_selector

    def save(
        self,
        filepath: str,
        result: Optional[FreqFluxDepResultType] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/flux_dep/freq",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        values, fpts, signals2D = result

        save_data(
            filepath=filepath,
            x_info={"name": "Flux device value", "unit": "a.u.", "values": values},
            y_info={"name": "Frequency", "unit": "Hz", "values": fpts * 1e6},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals2D.T},
            comment=comment,
            tag=tag,
            **kwargs,
        )

    def load(self, filepath: str, **kwargs) -> FreqFluxDepResultType:
        signals2D, values, fpts = load_data(filepath, **kwargs)
        assert values is not None and fpts is not None
        assert len(values.shape) == 1 and len(fpts.shape) == 1
        assert signals2D.shape == (len(fpts), len(values))

        fpts = fpts * 1e-6  # Hz -> MHz
        signals2D = signals2D.T  # transpose back

        values = values.astype(np.float64)
        fpts = fpts.astype(np.float64)
        signals2D = signals2D.astype(np.complex128)

        self.last_cfg = None
        self.last_result = (values, fpts, signals2D)

        return values, fpts, signals2D
