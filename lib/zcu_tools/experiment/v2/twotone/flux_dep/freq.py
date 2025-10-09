from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np

from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.utils import set_flux_in_dev_cfg, sweep2array
from zcu_tools.liveplot import LivePlotter2D, LivePlotter2DwithLine
from zcu_tools.notebook.analysis.fluxdep.interactive import (
    InteractiveFindPoints,
    InteractiveLines,
)
from zcu_tools.program.v2 import TwoToneProgram, sweep2param
from zcu_tools.utils.datasaver import save_data
from zcu_tools.utils.process import minus_background, rotate2real

from ...runner import HardTask, Runner, SoftTask
from .util import check_flux_pulse, wrap_with_flux_pulse

FreqResultType = Tuple[np.ndarray, np.ndarray, np.ndarray]


def freq_signal2real(signals: np.ndarray) -> np.ndarray:
    return np.abs(minus_background(signals, axis=1))


class FreqExperiment(AbsExperiment[FreqResultType]):
    def run_with_yoko(
        self, soc, soccfg, cfg: Dict[str, Any], *, progress: bool = True
    ) -> FreqResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        flx_sweep = cfg["sweep"]["flux"]
        fpt_sweep = cfg["sweep"]["freq"]

        # Remove flux from sweep dict - will be handled by soft loop
        cfg["sweep"] = {"freq": fpt_sweep}

        dev_values = sweep2array(flx_sweep, allow_array=True)
        fpts = sweep2array(fpt_sweep)  # predicted frequency points

        # Frequency is swept by FPGA (hard sweep)
        cfg["qub_pulse"]["freq"] = sweep2param("freq", fpt_sweep)

        with LivePlotter2DwithLine(
            "Flux device value",
            "Frequency (MHz)",
            line_axis=1,
            num_lines=2,
            disable=not progress,
        ) as viewer:
            signals = Runner(
                task=SoftTask(
                    sweep_name="flux",
                    sweep_values=dev_values,
                    update_cfg_fn=lambda _, ctx, flx: (
                        set_flux_in_dev_cfg(ctx.cfg["dev"], flx)
                    ),
                    sub_task=HardTask(
                        measure_fn=lambda ctx, update_hook: (
                            TwoToneProgram(soccfg, ctx.cfg).acquire(
                                soc, progress=False, callback=update_hook
                            )
                        ),
                        result_shape=(len(fpts),),
                    ),
                ),
                update_hook=lambda ctx: viewer.update(
                    dev_values, fpts, freq_signal2real(np.asarray(ctx.get_data()))
                ),
            ).run(cfg)
            signals = np.asarray(signals)

        # Cache results
        self.last_cfg = cfg
        self.last_result = (dev_values, fpts, signals)

        return dev_values, fpts, signals

    def run_fastflux(
        self,
        soc,
        soccfg,
        cfg: Dict[str, Any],
        *,
        flx_margin: float = 0.0,
        progress: bool = True,
    ) -> FreqResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        cfg["sweep"] = {
            "flux": cfg["sweep"]["flux"],
            "freq": cfg["sweep"]["freq"],
        }

        cfg["qub_pulse"], cfg["flx_pulse"] = wrap_with_flux_pulse(
            cfg["qub_pulse"], cfg["flx_pulse"], margin=flx_margin
        )
        check_flux_pulse(cfg["flx_pulse"])

        gains = sweep2array(cfg["sweep"]["flux"])
        fpts = sweep2array(cfg["sweep"]["freq"])

        # Frequency is swept by FPGA (hard sweep)
        cfg["flx_pulse"]["gain"] = sweep2param("flux", cfg["sweep"]["flux"])
        cfg["qub_pulse"]["freq"] = sweep2param("freq", cfg["sweep"]["freq"])

        with LivePlotter2D(
            "Flux device value", "Frequency (MHz)", disable=not progress
        ) as viewer:
            signals = Runner(
                task=HardTask(
                    measure_fn=lambda ctx, update_hook: (
                        TwoToneProgram(soccfg, ctx.cfg).acquire(
                            soc, progress=False, callback=update_hook
                        )
                    ),
                    result_shape=(len(gains), len(fpts)),
                ),
                update_hook=lambda ctx: viewer.update(
                    gains, fpts, rotate2real(np.asarray(ctx.get_data())).real
                ),
            ).run(cfg)
            signals = np.asarray(signals)

        # Cache results
        self.last_cfg = cfg
        self.last_result = (gains, fpts, signals)

        return gains, fpts, signals

    def run(
        self,
        soc,
        soccfg,
        cfg: Dict[str, Any],
        *,
        method: Literal["yoko", "fastflux"] = "yoko",
        progress: bool = True,
        **kwargs,
    ) -> FreqResultType:
        if method == "yoko":
            return self.run_with_yoko(soc, soccfg, cfg, progress=progress, **kwargs)
        elif method == "fastflux":
            return self.run_fastflux(soc, soccfg, cfg, progress=progress, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")

    def analyze(
        self,
        result: Optional[FreqResultType] = None,
        mA_c: Optional[float] = None,
        mA_e: Optional[float] = None,
    ) -> InteractiveLines:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        values, fpts, signals2D = result

        signals2D = minus_background(signals2D, axis=1)

        actline = InteractiveLines(
            signals2D.T, mAs=values, fpts=fpts, mA_c=mA_c, mA_e=mA_e
        )

        return actline

    def extract_points(
        self,
        result: Optional[FreqResultType] = None,
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
        result: Optional[FreqResultType] = None,
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
