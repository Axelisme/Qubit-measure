from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.utils import sweep2array
from zcu_tools.liveplot import LivePlotter2DwithLine
from zcu_tools.notebook.analysis.fluxdep.interactive import InteractiveLines
from zcu_tools.program.v2 import TwoToneProgram, sweep2param
from zcu_tools.utils.datasaver import save_data
from zcu_tools.utils.process import minus_background

from ..template import sweep2D_soft_hard_template, sweep2D_soft_template

FluxDepResultType = Tuple[np.ndarray, np.ndarray, np.ndarray]


def fluxdep_signal2real(signals: np.ndarray) -> np.ndarray:
    return np.abs(minus_background(signals, axis=1))


class FluxDepExperiment(AbsExperiment[FluxDepResultType]):
    """Two-tone flux dependence experiment.

    Sweeps flux bias and qubit frequency to map out the
    qubit transition as a function of magnetic flux.
    """

    def run_pure_zcu(
        self, soc, soccfg, cfg: Dict[str, Any], *, progress: bool = True
    ) -> FluxDepResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        flux_cfg = cfg["dev"]["flux_dev"]

        qub_pulse = cfg["qub_pulse"]
        flx_sweep = cfg["sweep"]["flux"]
        fpt_sweep = cfg["sweep"]["freq"]

        # Remove flux from sweep dict - will be handled by soft loop
        cfg["sweep"] = {"freq": fpt_sweep}

        flux_values = sweep2array(flx_sweep, allow_array=True)
        fpts = sweep2array(fpt_sweep)  # predicted frequency points

        # Frequency is swept by FPGA (hard sweep)
        qub_pulse["freq"] = sweep2param("freq", fpt_sweep)

        # Set initial flux
        flux_cfg["value"] = flux_values[0]

        def updateCfg(cfg: Dict[str, Any], _: int, value: float) -> None:
            if flux_cfg["mode"] == "current":
                value *= 1e-3  # convert mA to A

            cfg["dev"]["flux_dev"]["value"] = value

        def measure_fn(
            cfg: Dict[str, Any], cb: Optional[Callable[..., None]]
        ) -> np.ndarray:
            prog = TwoToneProgram(soccfg, cfg)
            return prog.acquire(soc, progress=False, callback=cb)[0][0].dot([1, 1j])

        # Run 2D soft-hard sweep (flux soft, frequency hard)
        signals2D = sweep2D_soft_hard_template(
            cfg,
            measure_fn,
            LivePlotter2DwithLine(
                f"Flux ({'mA' if flux_cfg['mode'] == 'current' else 'V'})",
                "Frequency (MHz)",
                line_axis=1,
                num_lines=2,
                disable=not progress,
            ),
            xs=flux_values * (1e3 if flux_cfg["mode"] == "current" else 1),  # mA / V
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
        self.last_result = (flux_values, fpts_real, signals2D)

        return flux_values, fpts_real, signals2D

    def run_with_rf_source(
        self,
        soc,
        soccfg,
        cfg: Dict[str, Any],
        *,
        progress: bool = True,
    ) -> FluxDepResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        flux_cfg = cfg["dev"]["flux_dev"]

        qub_pulse = cfg["qub_pulse"]
        flx_sweep = cfg["sweep"]["flux"]
        fpt_sweep = cfg["sweep"]["freq"]

        # Both sweep will be handled by soft loop
        del cfg["sweep"]

        dev_values = sweep2array(flx_sweep, allow_array=True)
        fpts = sweep2array(fpt_sweep, allow_array=True)

        # Frequency is swept by RF source, zcu only controls the waveform
        qub_pulse["freq"] = 0.0

        # Set initial flux
        flux_cfg["value"] = dev_values[0]

        def updateCfg_x(cfg: Dict[str, Any], _: int, value: float) -> None:
            if flux_cfg["mode"] == "current":
                value *= 1e-3  # convert mA to A

            cfg["dev"]["flux_dev"]["value"] = value

        def updateCfg_y(cfg: Dict[str, Any], _: int, fpt: float) -> None:
            cfg["dev"]["rf_dev"]["freq"] = fpt * 1e6  # convert MHz to Hz

        def measure_fn(
            cfg: Dict[str, Any], cb: Optional[Callable[..., None]]
        ) -> np.ndarray:
            prog = TwoToneProgram(soccfg, cfg)
            return prog.acquire(soc, progress=False, callback=cb)[0][0].dot([1, 1j])

        # Run 2D soft-hard sweep (flux soft, frequency hard)
        signals2D = sweep2D_soft_template(
            cfg,
            measure_fn,
            LivePlotter2DwithLine(
                f"Flux ({'mA' if flux_cfg['mode'] == 'current' else 'V'})",
                "Frequency (MHz)",
                line_axis=1,
                num_lines=2,
                disable=not progress,
            ),
            xs=dev_values * (1e3 if flux_cfg["mode"] == "current" else 1),  # mA / V
            ys=fpts,
            updateCfg_x=updateCfg_x,
            updateCfg_y=updateCfg_y,
            signal2real=fluxdep_signal2real,
            progress=progress,
        )

        # Cache results
        self.last_cfg = cfg
        self.last_result = (dev_values, fpts, signals2D)

        return dev_values, fpts, signals2D

    def run(
        self,
        soc,
        soccfg,
        cfg: Dict[str, Any],
        *,
        with_rf_source: bool = False,
        progress: bool = True,
    ) -> FluxDepResultType:
        if with_rf_source:
            return self.run_with_rf_source(soc, soccfg, cfg, progress=progress)
        else:
            return self.run_pure_zcu(soc, soccfg, cfg, progress=progress)

    def analyze(
        self,
        result: Optional[FluxDepResultType] = None,
        mA_c: Optional[float] = None,
        mA_e: Optional[float] = None,
    ) -> InteractiveLines:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        values, fpts, signals2D = result

        actline = InteractiveLines(
            signals2D,
            mAs=values,
            fpts=fpts,
            mA_c=mA_c,
            mA_e=mA_e,
            use_phase=True,
        )

        return actline

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

        values, fpts, signals2D = result

        save_data(
            filepath=filepath,
            x_info={"name": "Frequency", "unit": "Hz", "values": fpts * 1e6},
            y_info={"name": "Flux device value", "unit": "a.u.", "values": values},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals2D},
            comment=comment,
            tag=tag,
            **kwargs,
        )
