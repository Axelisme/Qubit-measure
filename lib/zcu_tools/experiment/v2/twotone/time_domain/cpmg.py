from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional, Tuple, Callable

import numpy as np
import qick.asm_v2 as qasm

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
    real_signals = rotate2real(signals).real
    max_vals = np.max(real_signals, axis=1, keepdims=True)
    min_vals = np.min(real_signals, axis=1, keepdims=True)
    return (real_signals - min_vals) / (max_vals - min_vals)


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

        cfg["sweep"].pop("times")

        times = sweep2array(times_sweep, allow_array=True)
        ts = sweep2array(len_sweep)  # predicted times

        if np.min(times) <= 0:
            raise ValueError("times should be larger than 0")

        def updateCfg(cfg: Dict[str, Any], _: int, value: float) -> None:
            cfg["time"] = int(value)

        updateCfg(cfg, 0, times[0])  # set initial flux

        cpmg_spans = sweep2param("length", len_sweep)

        def make_prog(cfg, time):
            interval = cpmg_spans / time

            if time > 1:  # zero Loop mean infinite loop in qick
                cpmg_pi_loop = [
                    qasm.OpenLoop(name="cpmg_pi_loop", n=time - 1),
                    Pulse(
                        name="pi_pulse",
                        cfg={**cfg["pi_pulse"], "post_delay": interval},
                    ),
                    qasm.CloseLoop(),
                ]
            else:
                cpmg_pi_loop = []

            prog = ModularProgramV2(
                soccfg,
                cfg,
                modules=[
                    make_reset("reset", reset_cfg=cfg.get("reset")),
                    Pulse(
                        name="pi2_pulse1",
                        cfg={**cfg["pi2_pulse"], "post_delay": 0.5 * interval},
                        pulse_name="pi2_pulse",
                    ),
                    *cpmg_pi_loop,
                    Pulse(
                        name="last_pi_pulse",
                        cfg={**cfg["pi_pulse"], "post_delay": 0.5 * interval},
                        pulse_name="pi_pulse",
                    ),
                    Pulse(
                        name="pi2_pulse2", cfg=cfg["pi2_pulse"], pulse_name="pi2_pulse"
                    ),
                    make_readout("readout", readout_cfg=cfg["readout"]),
                ],
            )

            return prog

        def measure_fn(
            cfg: Dict[str, Any], cb: Optional[Callable[..., None]]
        ) -> np.ndarray:
            prog = make_prog(cfg, time=cfg["time"])
            return prog.acquire(soc, progress=False, callback=cb)[0][0].dot([1, 1j])

        make_prog(cfg, time=np.max(times))  # test compile

        signals = sweep2D_soft_hard_template(
            cfg,
            measure_fn,
            LivePlotter2DwithLine(
                "Number of Pi",
                "Time (us)",
                line_axis=1,
                num_lines=2,
                disable=not progress,
            ),
            xs=times,
            ys=ts,
            updateCfg=updateCfg,
            signal2real=cpmg_signal2real,
        )

        # record last cfg and result
        self.last_cfg = cfg
        self.last_result = (times, ts, signals)

        return times, ts, signals

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
