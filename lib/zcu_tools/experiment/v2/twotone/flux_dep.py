from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.utils import sweep2array
from zcu_tools.liveplot import LivePlotter2DwithLine
from zcu_tools.program.v2 import TwoToneProgram, sweep2param
from zcu_tools.utils.datasaver import save_data
from zcu_tools.utils.process import minus_background

from ..template import sweep2D_soft_hard_template

FluxDepResultType = Tuple[np.ndarray, np.ndarray, np.ndarray]


def fluxdep_signal2real(signals: np.ndarray) -> np.ndarray:
    return np.abs(minus_background(signals, axis=1))


class FluxDepExperiment(AbsExperiment[FluxDepResultType]):
    """Two-tone flux dependence experiment.

    Sweeps flux bias and qubit frequency to map out the
    qubit transition as a function of magnetic flux.
    """

    def run(
        self,
        soc,
        soccfg,
        cfg: Dict[str, Any],
        *,
        progress: bool = True,
    ) -> FluxDepResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        qub_pulse = cfg["qub_pulse"]
        flx_sweep = cfg["sweep"]["flux"]
        fpt_sweep = cfg["sweep"]["freq"]

        # Remove flux from sweep dict - will be handled by soft loop
        cfg["sweep"] = {"freq": fpt_sweep}

        As = sweep2array(flx_sweep, allow_array=True)
        fpts = sweep2array(fpt_sweep)  # predicted frequency points

        # Check flux device is configured
        if cfg["dev"]["flux_dev"] == "none":
            raise ValueError("Flux sweep requires flux_dev != 'none'")

        # Frequency is swept by FPGA (hard sweep)
        qub_pulse["freq"] = sweep2param("freq", fpt_sweep)

        # Set initial flux
        cfg["dev"]["flux"] = As[0]

        def updateCfg(cfg: Dict[str, Any], _: int, mA: float) -> None:
            """Update configuration for each flux point."""
            cfg["dev"]["flux"] = mA * 1e-3  # convert mA to A

        def measure_fn(
            cfg: Dict[str, Any], cb: Optional[Callable[..., None]]
        ) -> np.ndarray:
            """Measurement function for each flux point."""
            prog = TwoToneProgram(soccfg, cfg)
            return prog.acquire(soc, progress=False, callback=cb)[0][0].dot([1, 1j])

        # Run 2D soft-hard sweep (flux soft, frequency hard)
        signals2D = sweep2D_soft_hard_template(
            cfg,
            measure_fn,
            LivePlotter2DwithLine(
                "Flux (mA)",
                "Frequency (MHz)",
                line_axis=1,
                num_lines=2,
                disable=not progress,
            ),
            xs=1e3 * As,  # convert to mA for display
            ys=fpts,
            updateCfg=updateCfg,
            signal2real=fluxdep_signal2real,
            progress=progress,
        )

        # Get the actual frequency points used by FPGA
        prog = TwoToneProgram(soccfg, cfg)
        fpts_real = prog.get_pulse_param("qubit_pulse", "freq", as_array=True)
        assert isinstance(fpts_real, np.ndarray), "fpts should be an array"

        # Cache results
        self.last_cfg = cfg
        self.last_result = (As, fpts_real, signals2D)

        return As, fpts_real, signals2D

    def analyze(
        self,
        result: Optional[FluxDepResultType] = None,
    ) -> None:
        """Analysis not yet implemented for two-tone flux dependence."""
        raise NotImplementedError(
            "Analysis not implemented for two-tone flux dependence"
        )

    def save(
        self,
        filepath: str,
        result: Optional[FluxDepResultType] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/flux_dep",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        As, fpts, signals2D = result

        save_data(
            filepath=filepath,
            x_info={"name": "Frequency", "unit": "Hz", "values": fpts * 1e6},
            y_info={"name": "Current", "unit": "A", "values": As},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals2D},
            comment=comment,
            tag=tag,
            **kwargs,
        )
