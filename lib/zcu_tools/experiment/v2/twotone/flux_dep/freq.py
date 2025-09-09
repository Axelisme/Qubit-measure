from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Tuple, Literal

import numpy as np

from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.utils import (
    sweep2array,
    set_flux_in_dev_cfg,
    set_freq_in_dev_cfg,
)
from zcu_tools.liveplot import LivePlotter2DwithLine
from zcu_tools.notebook.analysis.fluxdep.interactive import InteractiveLines
from zcu_tools.program.v2 import (
    TwoToneProgram,
    sweep2param,
    ModularProgramV2,
    make_readout,
    make_reset,
    Pulse,
)
from zcu_tools.utils.datasaver import save_data
from zcu_tools.utils.process import minus_background

from ...template import sweep2D_soft_hard_template, sweep2D_soft_template
from .util import check_flux_pulse, wrap_with_flux_pulse

FreqResultType = Tuple[np.ndarray, np.ndarray, np.ndarray]


def freq_signal2real(signals: np.ndarray) -> np.ndarray:
    return np.abs(minus_background(signals, axis=1))


class FreqExperiment(AbsExperiment[FreqResultType]):
    def run_with_yoko(
        self, soc, soccfg, cfg: Dict[str, Any], *, progress: bool = True
    ) -> FreqResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        qub_pulse = cfg["qub_pulse"]
        flx_sweep = cfg["sweep"]["flux"]
        fpt_sweep = cfg["sweep"]["freq"]

        # Remove flux from sweep dict - will be handled by soft loop
        cfg["sweep"] = {"freq": fpt_sweep}

        dev_values = sweep2array(flx_sweep, allow_array=True)
        fpts = sweep2array(fpt_sweep)  # predicted frequency points

        # Frequency is swept by FPGA (hard sweep)
        qub_pulse["freq"] = sweep2param("freq", fpt_sweep)

        def updateCfg(cfg: Dict[str, Any], _: int, value: float) -> None:
            set_flux_in_dev_cfg(cfg["dev"], value)

        updateCfg(cfg, 0, dev_values[0])  # set initial flux

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
                "Flux device value",
                "Frequency (MHz)",
                line_axis=1,
                num_lines=2,
                disable=not progress,
            ),
            xs=dev_values,
            ys=fpts,
            updateCfg=updateCfg,
            signal2real=freq_signal2real,
            progress=progress,
        )

        # Get the actual frequency points used by FPGA
        prog = TwoToneProgram(soccfg, cfg)
        fpts_real = prog.get_pulse_param("qubit_pulse", "freq", as_array=True)
        assert isinstance(fpts_real, np.ndarray), "fpts should be an array"

        # Cache results
        self.last_cfg = cfg
        self.last_result = (dev_values, fpts_real, signals2D)

        return dev_values, fpts_real, signals2D

    def run_with_rf_yoko(
        self,
        soc,
        soccfg,
        cfg: Dict[str, Any],
        *,
        progress: bool = True,
    ) -> FreqResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        if "rf_dev" not in cfg["dev"]:
            raise ValueError("RF source is not configured")

        qub_pulse = cfg["qub_pulse"]
        flx_sweep = cfg["sweep"]["flux"]
        fpt_sweep = cfg["sweep"]["freq"]

        # Both sweep will be handled by soft loop
        del cfg["sweep"]

        dev_values = sweep2array(flx_sweep, allow_array=True)
        fpts = sweep2array(fpt_sweep, allow_array=True)

        # Frequency is swept by RF source, zcu only controls the waveform
        qub_pulse["freq"] = 0.0

        def updateCfg_x(cfg: Dict[str, Any], _: int, value: float) -> None:
            set_flux_in_dev_cfg(cfg["dev"], value)

        def updateCfg_y(cfg: Dict[str, Any], _: int, fpt: float) -> None:
            set_freq_in_dev_cfg((cfg["dev"], fpt * 1e6))  # convert MHz to Hz

        updateCfg_x(cfg, 0, dev_values[0])  # set initial flux
        updateCfg_y(cfg, 0, fpts[0])  # set initial frequency

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
                "Flux device value",
                "Frequency (MHz)",
                line_axis=1,
                num_lines=2,
                disable=not progress,
            ),
            xs=dev_values,
            ys=fpts,
            updateCfg_x=updateCfg_x,
            updateCfg_y=updateCfg_y,
            signal2real=freq_signal2real,
            progress=progress,
        )

        # Cache results
        self.last_cfg = cfg
        self.last_result = (dev_values, fpts, signals2D)

        return dev_values, fpts, signals2D

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

        qub_pulse = cfg["qub_pulse"]
        flx_pulse = cfg["flx_pulse"]

        qub_pulse, flx_pulse = wrap_with_flux_pulse(
            qub_pulse, flx_pulse, margin=flx_margin
        )
        check_flux_pulse(flx_pulse)

        flx_sweep = cfg["sweep"]["flux"]
        fpt_sweep = cfg["sweep"]["freq"]

        # Remove flux from sweep dict - will be handled by soft loop
        cfg["sweep"] = {"freq": fpt_sweep}

        gains = sweep2array(flx_sweep, allow_array=True)
        fpts = sweep2array(fpt_sweep)  # predicted frequency points

        # Frequency is swept by FPGA (hard sweep)
        qub_pulse["freq"] = sweep2param("freq", fpt_sweep)
        flx_pulse["gain"] = gains[0]  # set initial gain

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
                    Pulse(name="flux_pulse", cfg=cfg["flx_pulse"]),
                    Pulse(name="qubit_pulse", cfg=cfg["qub_pulse"]),
                    make_readout("readout", readout_cfg=cfg["readout"]),
                ],
            )
            return prog.acquire(soc, progress=False, callback=cb)[0][0].dot([1, 1j])

        # Run 2D soft-hard sweep (flux soft, frequency hard)
        signals2D = sweep2D_soft_hard_template(
            cfg,
            measure_fn,
            LivePlotter2DwithLine(
                "Local flux gain (a.u.)",
                "Frequency (MHz)",
                line_axis=1,
                num_lines=2,
                disable=not progress,
            ),
            xs=gains,
            ys=fpts,
            updateCfg=updateCfg,
            signal2real=freq_signal2real,
            progress=progress,
        )

        # Get the actual frequency points used by FPGA
        prog = ModularProgramV2(
            soccfg, cfg, modules=[Pulse(name="qubit_pulse", cfg=cfg["qub_pulse"])]
        )
        fpts_real = prog.get_pulse_param("qubit_pulse", "freq", as_array=True)
        assert isinstance(fpts_real, np.ndarray), "fpts should be an array"

        # Cache results
        self.last_cfg = cfg
        self.last_result = (gains, fpts_real, signals2D)

        return gains, fpts_real, signals2D

    def run(
        self,
        soc,
        soccfg,
        cfg: Dict[str, Any],
        *,
        method: Literal["yoko", "rf_yoko", "fastflux"] = "yoko",
        progress: bool = True,
        **kwargs,
    ) -> FreqResultType:
        if method == "yoko":
            return self.run_with_yoko(soc, soccfg, cfg, progress=progress, **kwargs)
        elif method == "rf_yoko":
            return self.run_with_rf_yoko(soc, soccfg, cfg, progress=progress, **kwargs)
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
            x_info={"name": "Frequency", "unit": "Hz", "values": fpts * 1e6},
            y_info={"name": "Flux device value", "unit": "a.u.", "values": values},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals2D},
            comment=comment,
            tag=tag,
            **kwargs,
        )
