from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.utils import sweep2array
from zcu_tools.liveplot import LivePlotter2D
from zcu_tools.program.v2 import (
    ModularProgramV2,
    Pulse,
    make_readout,
    make_reset,
    sweep2param,
    derive_readout_cfg,
    derive_reset_cfg,
)
from zcu_tools.utils.datasaver import save_data
from zcu_tools.utils.process import rotate2real
from zcu_tools.library import ModuleLibrary

from ...template import sweep_hard_template
from .util import check_flux_pulse

FluxLookbackResultType = Tuple[np.ndarray, np.ndarray, np.ndarray]


def fluxlookback_signal2real(signals: np.ndarray) -> np.ndarray:
    return rotate2real(signals).real


class FluxLookbackExperiment(AbsExperiment[FluxLookbackResultType]):
    def derive_cfg(
        self, ml: ModuleLibrary, cfg: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        cfg = deepcopy(cfg)
        cfg.update(kwargs)

        # Ensure the outer loop
        cfg["sweep"] = {
            "length": cfg["sweep"]["length"],
            "freq": cfg["sweep"]["freq"],
        }

        if "reset" in cfg:
            cfg["reset"] = derive_reset_cfg(ml, cfg["reset"])

        flx_pulse = cfg["flx_pulse"]
        flx_pulse.setdefault("nqz", 1)
        flx_pulse.setdefault("freq", 0.0)
        flx_pulse.setdefault("phase", 0.0)
        flx_pulse.setdefault("outsel", "input")
        flx_pulse.setdefault("post_delay", None)

        return cfg

    def run(
        self, soc, soccfg, cfg: Dict[str, Any], *, progress: bool = True
    ) -> FluxLookbackResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        qub_pulse = cfg["qub_pulse"]

        check_flux_pulse(cfg["flx_pulse"])

        # predict point
        lens = sweep2array(cfg["sweep"]["length"])
        fpts = sweep2array(cfg["sweep"]["freq"])

        # swept by FPGA (hard sweep)
        qub_pulse["t"] = sweep2param("length", cfg["sweep"]["length"])
        qub_pulse["freq"] = sweep2param("freq", cfg["sweep"]["freq"])

        prog = ModularProgramV2(
            soccfg,
            cfg,
            modules=[
                make_reset("reset", reset_cfg=cfg.get("reset")),
                Pulse(name="flux_pulse", cfg=cfg["flx_pulse"]),
                Pulse(name="qub_pulse", cfg=cfg["qub_pulse"]),
                make_readout("readout", readout_cfg=cfg["readout"]),
            ],
        )

        def measure_fn(
            _: Dict[str, Any], cb: Optional[Callable[..., None]]
        ) -> np.ndarray:
            return prog.acquire(soc, progress=progress, callback=cb)[0][0].dot([1, 1j])

        # Run 2D soft-hard sweep (flux soft, length hard)
        signals2D = sweep_hard_template(
            cfg,
            measure_fn,
            LivePlotter2D(
                "Time (us)",
                "Frequency (MHz)",
                disable=not progress,
            ),
            ticks=(lens, fpts),
            signal2real=fluxlookback_signal2real,
        )

        # Cache results
        self.last_cfg = cfg
        self.last_result = (lens, fpts, signals2D)

        return lens, fpts, signals2D

    def analyze(self, result: Optional[FluxLookbackResultType] = None) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        # lens, fpts, signals2D = result

        raise NotImplementedError("analysis not implemented yet")

    def save(
        self,
        filepath: str,
        result: Optional[FluxLookbackResultType] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/flux_dep/flx_lookback",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        lens, fpts, signals2D = result

        save_data(
            filepath=filepath,
            x_info={"name": "Time", "unit": "s", "values": lens * 1e-6},
            y_info={"name": "Frequency", "unit": "Hz", "values": fpts * 1e6},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals2D.T},
            comment=comment,
            tag=tag,
            **kwargs,
        )
