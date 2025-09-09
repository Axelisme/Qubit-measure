from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Tuple

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
from .util import check_flux_pulse

T2RamseyResultType = Tuple[np.ndarray, np.ndarray, np.ndarray]


def t2ramsey_signal2real(signals: np.ndarray) -> np.ndarray:
    return rotate2real(signals).real


class T2RamseyExperiment(AbsExperiment[T2RamseyResultType]):
    def run(
        self,
        soc,
        soccfg,
        cfg: Dict[str, Any],
        *,
        detune: float = 0.0,
        progress: bool = True,
    ) -> T2RamseyResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        flx_pulse = cfg["flx_pulse"]

        check_flux_pulse(flx_pulse)

        flx_sweep = cfg["sweep"]["flux"]
        len_sweep = cfg["sweep"]["length"]

        # Remove flux from sweep dict - will be handled by soft loop
        cfg["sweep"] = {"length": len_sweep}

        gains = sweep2array(flx_sweep, allow_array=True)
        ts = sweep2array(len_sweep)

        # Frequency is swept by FPGA (hard sweep)
        flx_pulse["gain"] = gains[0]  # set initial gain
        t2r_spans = sweep2param("length", len_sweep)

        def updateCfg(cfg: Dict[str, Any], _: int, value: float) -> None:
            cfg["flx_pulse"]["gain"] = value

        updateCfg(cfg, 0, gains[0])  # set initial flux

        def measure_fn(
            cfg: Dict[str, Any], cb: Optional[Callable[..., None]]
        ) -> np.ndarray:
            prog = ModularProgramV2(
                soccfg,
                cfg,
                modules=[
                    make_reset("reset", reset_cfg=cfg.get("reset")),
                    Pulse(name="pi2_pulse1", cfg=cfg["pi2_pulse"]),
                    Pulse(
                        name="flux_pulse",
                        cfg={**cfg["flx_pulse"], "length": t2r_spans},
                    ),
                    Pulse(
                        name="pi2_pulse2",
                        cfg={  # activate detune
                            **cfg["pi2_pulse"],
                            "phase": cfg["pi2_pulse"]["phase"]
                            + 360 * detune * t2r_spans,
                        },
                    ),
                    make_readout("readout", readout_cfg=cfg["readout"]),
                ],
            )
            return prog.acquire(soc, progress=False, callback=cb)[0][0].dot([1, 1j])

        # Run 2D soft-hard sweep (flux soft, length hard)
        signals2D = sweep2D_soft_hard_template(
            cfg,
            measure_fn,
            LivePlotter2DwithLine(
                "Local flux gain (a.u.)",
                "Time (us)",
                line_axis=1,
                num_lines=2,
                disable=not progress,
            ),
            xs=gains,
            ys=ts,
            updateCfg=updateCfg,
            signal2real=t2ramsey_signal2real,
            progress=progress,
        )

        # Get the actual frequency points used by FPGA
        prog = ModularProgramV2(
            soccfg,
            cfg,
            modules=[
                Pulse(name="flux_pulse", cfg={**cfg["flx_pulse"], "length": t2r_spans})
            ],
        )
        real_ts = prog.get_pulse_param("flux_pulse", "length", as_array=True)
        assert isinstance(real_ts, np.ndarray), "fpts should be an array"
        real_ts += ts[0] - real_ts[0]  # correct absolute offset

        # Cache results
        self.last_cfg = cfg
        self.last_result = (gains, real_ts, signals2D)

        return gains, real_ts, signals2D

    def analyze(
        self,
        result: Optional[T2RamseyResultType] = None,
        mA_c: Optional[float] = None,
        mA_e: Optional[float] = None,
    ) -> None:
        raise NotImplementedError("Analysis not implemented yet")

    def save(
        self,
        filepath: str,
        result: Optional[T2RamseyResultType] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/flux_dep/ge/t2ramsey",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        gains, Ts, signals2D = result

        save_data(
            filepath=filepath,
            x_info={"name": "Time", "unit": "s", "values": Ts * 1e-6},
            y_info={"name": "Flux pulse gain", "unit": "a.u.", "values": gains},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals2D},
            comment=comment,
            tag=tag,
            **kwargs,
        )
