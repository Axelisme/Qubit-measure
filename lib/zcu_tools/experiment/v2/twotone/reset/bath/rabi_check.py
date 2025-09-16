from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import numpy as np

from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.utils import format_sweep1D, sweep2array
from zcu_tools.liveplot import LivePlotter1D
from zcu_tools.program.v2 import (
    ModularProgramV2,
    Pulse,
    make_readout,
    make_reset,
    sweep2param,
)
from zcu_tools.utils.datasaver import save_data
from zcu_tools.utils.process import rotate2real

from ....template import sweep_hard_template

# (pdrs, signals_2d)  # signals shape: (2, len(pdrs)) for [w/o reset, w/ reset]
ResetRabiCheckResultType = Tuple[np.ndarray, np.ndarray]


def reset_rabi_signal2real(signals: np.ndarray) -> np.ndarray:
    return rotate2real(signals).real


class RabiCheckExperiment(AbsExperiment[ResetRabiCheckResultType]):
    def run(
        self, soc, soccfg, cfg: Dict[str, Any], *, progress: bool = True
    ) -> ResetRabiCheckResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        # Get reset and init pulse configurations
        reset_cfg = cfg["tested_reset"]
        init_pulse = cfg["init_pulse"]

        # Check that reset pulse is dual pulse type
        if reset_cfg["type"] != "bath":
            raise ValueError("This experiment only supports bath reset")

        # Canonicalise sweep section to single-axis form
        cfg["sweep"] = format_sweep1D(cfg["sweep"], "gain")
        gain_sweep = cfg["sweep"]["gain"]

        # Create 2D sweep: w/o_reset (outer) x gain (inner)
        cfg["sweep"] = {
            "w/o_reset": {"start": 0, "stop": 1.0, "expts": 2},
            "gain": gain_sweep,
        }

        # Attach gain sweep to initialization pulse
        init_pulse["gain"] = sweep2param("gain", gain_sweep)

        # Attach reset factor to control reset on/off
        reset_factor = sweep2param("w/o_reset", cfg["sweep"]["w/o_reset"])
        qub_pulse = reset_cfg["qubit_tone_cfg"]
        res_pulse = reset_cfg["cavity_tone_cfg"]
        pi2_pulse = reset_cfg["pi2_cfg"]

        # Scale qubit and cavity reset pulses by reset factor
        qub_pulse["gain"] = reset_factor * qub_pulse["gain"]
        res_pulse["gain"] = reset_factor * res_pulse["gain"]
        pi2_pulse["gain"] = reset_factor * pi2_pulse["gain"]
        qub_pulse["length"] = reset_factor * qub_pulse["length"] + 0.01
        res_pulse["length"] = reset_factor * res_pulse["length"] + 0.01

        # Handle flat_top pulse style to prevent negative length
        if qub_pulse.get("style") == "flat_top" and "raise_pulse" in qub_pulse:
            qub_pulse["length"] += qub_pulse["raise_pulse"]["length"]
        if res_pulse.get("style") == "flat_top" and "raise_pulse" in res_pulse:
            res_pulse["length"] += res_pulse["raise_pulse"]["length"]
        if pi2_pulse["style"] in ["const", "flat_top"]:
            pi2_pulse["length"] = reset_factor * pi2_pulse["length"] + 0.01
            if pi2_pulse["style"] == "flat_top":
                pi2_pulse["length"] += pi2_pulse["raise_pulse"]["length"]

        prog = ModularProgramV2(
            soccfg,
            cfg,
            modules=[
                make_reset("reset", reset_cfg=cfg.get("reset")),
                Pulse("init_pulse", cfg=cfg.get("init_pulse")),
                make_reset("tested_reset", reset_cfg=reset_cfg),
                make_readout("readout", readout_cfg=cfg["readout"]),
            ],
        )

        pdrs = sweep2array(gain_sweep)  # predicted amplitudes

        signals = sweep_hard_template(
            cfg,
            lambda _, cb: prog.acquire(soc, progress=progress, callback=cb)[0][0].dot(
                [1, 1j]
            ),
            LivePlotter1D("Pulse gain", "Amplitude", num_lines=2, disable=not progress),
            ticks=(pdrs,),
            signal2real=reset_rabi_signal2real,
        )

        # Get actual parameters used by the FPGA
        pdrs = prog.get_pulse_param("init_pulse", "gain", as_array=True)
        assert isinstance(pdrs, np.ndarray), "pdrs should be an array"

        # Cache results
        self.last_cfg = cfg
        self.last_result = (pdrs, signals)

        return pdrs, signals

    def analyze(
        self,
        result: Optional[ResetRabiCheckResultType] = None,
        *,
        plot: bool = True,
    ) -> None:
        """Analyze reset rabi check results. (No specific analysis implemented)"""
        # No specific analysis needed for this experiment
        raise NotImplementedError("No specific analysis implemented")

    def save(
        self,
        filepath: str,
        result: Optional[ResetRabiCheckResultType] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/reset/bath/rabi_check",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        pdrs, signals = result

        save_data(
            filepath=filepath,
            x_info={"name": "Amplitude", "unit": "a.u.", "values": pdrs},
            y_info={"name": "Reset", "unit": "None", "values": np.array([0, 1])},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals},
            comment=comment,
            tag=tag,
            **kwargs,
        )
