from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional, Tuple, Callable

import numpy as np

from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.utils import sweep2array
from zcu_tools.liveplot import LivePlotter2DwithLine
from zcu_tools.program.v2 import (
    ModularProgramV2,
    Pulse,
    make_readout,
    make_reset,
    sweep2param,
)
from zcu_tools.utils.datasaver import save_data
from zcu_tools.utils.process import rotate2real

from ...template import sweep2D_soft_hard_template


def cpmg_signal2real(signals: np.ndarray) -> np.ndarray:
    return rotate2real(signals).real  # type: ignore


CPMGResultType = Tuple[np.ndarray, np.ndarray]  # (times, signals)


class CPMGExperiment(AbsExperiment[CPMGResultType]):
    def run(
        self,
        soc,
        soccfg,
        cfg: Dict[str, Any],
        *,
        progress: bool = True,
    ) -> CPMGResultType:
        cfg = deepcopy(cfg)

        times_sweep = cfg["sweep"]["times"]
        len_sweep = cfg["sweep"]["length"]

        times = sweep2array(times_sweep)
        ts = sweep2array(len_sweep)  # predicted times

        if np.min(times) <= 0:
            raise ValueError("times should be larger than 0")

        def updateCfg(cfg: Dict[str, Any], _: int, value: float) -> None:
            cfg["time"] = value

        updateCfg(cfg, 0, times[0])  # set initial flux

        cpmg_spans = sweep2param("length", len_sweep)

        def measure_fn(
            cfg: Dict[str, Any], cb: Optional[Callable[..., None]]
        ) -> np.ndarray:
            time = cfg["time"]
            interval = cpmg_spans / time
            prog = ModularProgramV2(
                soccfg,
                cfg,
                modules=[
                    make_reset("reset", reset_cfg=cfg.get("reset")),
                    Pulse(
                        name="pi2_pulse1",
                        cfg={
                            **cfg["pi2_pulse"],
                            "post_delay": 0.5 * interval,
                        },
                        pulse_name="pi2_pulse",
                    ),
                    *[
                        Pulse(
                            name=f"pi_pulse_{i}",
                            cfg={
                                **cfg["pi_pulse"],
                                "post_delay": interval,
                            },
                            pulse_name="pi_pulse",
                        )
                        for i in range(time - 1)
                    ],
                    Pulse(
                        name=f"pi_pulse_{time - 1}",
                        cfg={
                            **cfg["pi_pulse"],
                            "post_delay": 0.5 * interval,
                        },
                        pulse_name="pi_pulse",
                    ),
                    Pulse(
                        name="pi2_pulse2", cfg=cfg["pi2_pulse"], pulse_name="pi2_pulse"
                    ),
                    make_readout("readout", readout_cfg=cfg["readout"]),
                ],
            )
            return prog.acquire(soc, progress=False, callback=cb)[0][0].dot([1, 1j])

        signals = sweep2D_soft_hard_template(
            cfg,
            measure_fn,
            LivePlotter2DwithLine(
                "Time (us)",
                "CPMG Time",
                line_axis=1,
                num_lines=2,
                disable=not progress,
            ),
            ticks=(times, ts),
            updateCfg=updateCfg,
            signal2real=cpmg_signal2real,
        )

        # get actual times
        prog = ModularProgramV2(
            soccfg,
            cfg,
            modules=[
                Pulse(
                    name="pi_pulse", cfg={**cfg["pi_pulse"], "post_delay": cpmg_spans}
                )
            ],
        )
        real_ts = prog.get_time_param("pi_pulse_post_delay", "t", as_array=True)
        assert isinstance(real_ts, np.ndarray), "real_ts should be an array"
        real_ts += ts[0] - real_ts[0]  # adjust to start from the first time

        # record last cfg and result
        self.last_cfg = cfg
        self.last_result = (times, real_ts, signals)

        return times, real_ts, signals

    def analyze(
        self,
        result: Optional[CPMGResultType] = None,
        *,
        plot: bool = True,
        max_contrast: bool = True,
        fit_fringe: bool = True,
    ) -> Tuple[float, float, float, float]:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        times, Ts, signals2D = result

        raise NotImplementedError("fit_fringe is not implemented yet")

    def save(
        self,
        filepath: str,
        result: Optional[CPMGResultType] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/ge/cpmg",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        times, Ts, signals2D = result
        save_data(
            filepath=filepath,
            x_info={"name": "Number of pi", "unit": "a.u.", "values": times},
            y_info={"name": "Time", "unit": "s", "values": Ts * 1e-6},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals2D.T},
            comment=comment,
            tag=tag,
            **kwargs,
        )
