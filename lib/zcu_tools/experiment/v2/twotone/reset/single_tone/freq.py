from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from zcu_tools.experiment import AbsExperiment, config
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
from zcu_tools.utils.fitting import fit_resonence_freq
from zcu_tools.utils.process import rotate2real

from ....template import sweep_hard_template

# (fpts, signals)
SingleToneResetFreqResultType = Tuple[np.ndarray, np.ndarray]


class ResetFreqExperiment(AbsExperiment[SingleToneResetFreqResultType]):
    """Single-tone reset frequency measurement experiment.

    Measures the optimal frequency for a single reset pulse by sweeping the reset
    pulse frequency and observing the qubit state after initialization and reset.

    The experiment performs:
    1. Initial reset (optional)
    2. Qubit initialization pulse (to prepare a state to reset from)
    3. Reset probe pulse with variable frequency
    4. Readout to measure reset effectiveness
    """

    def run(
        self,
        soc,
        soccfg,
        cfg: Dict[str, Any],
        *,
        progress: bool = True,
        remove_bg: bool = False,
    ) -> SingleToneResetFreqResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        # Canonicalise sweep section to single-axis form
        cfg["sweep"] = format_sweep1D(cfg["sweep"], "freq")
        sweep_cfg = cfg["sweep"]["freq"]

        # Check that reset pulse is single pulse type
        tested_reset = cfg["tested_reset"]
        if tested_reset["type"] != "pulse":
            raise ValueError("This experiment only supports single pulse reset")

        prog = ModularProgramV2(
            soccfg,
            soc,
            modules=[
                make_reset("reset", reset_cfg=cfg.get("reset")),
                Pulse("init_pulse", cfg=cfg.get("init_pulse")),
                Pulse(
                    "reset_probe_pulse",
                    cfg={
                        **tested_reset["pulse_cfg"],
                        "freq": sweep2param("freq", sweep_cfg),
                    },
                ),
                make_readout("readout", readout_cfg=cfg["readout"]),
            ],
        )

        fpts = sweep2array(sweep_cfg)  # predicted frequency points

        def reset_signal2real(signals: np.ndarray) -> np.ndarray:
            # Remove background if requested
            if remove_bg:
                signals = signals - np.mean(signals)
            return rotate2real(signals).real

        signals = sweep_hard_template(
            cfg,
            lambda _, cb: prog.acquire(soc, progress=progress, callback=cb)[0][0].dot(
                [1, 1j]
            ),
            LivePlotter1D("Frequency (MHz)", "Amplitude", disable=not progress),
            ticks=(fpts,),
            signal2real=reset_signal2real,
        )

        # Get the actual frequency points used by FPGA
        fpts = prog.get_pulse_param("reset_probe_pulse", "freq", as_array=True)
        assert isinstance(fpts, np.ndarray), "fpts should be an array"

        # Cache results
        self.last_cfg = cfg
        self.last_result = (fpts, signals)

        return fpts, signals

    def analyze(
        self,
        result: Optional[SingleToneResetFreqResultType] = None,
        *,
        type: Literal["lor", "sinc"] = "lor",
        asym: bool = False,
        plot: bool = True,
        max_contrast: bool = True,
    ) -> Tuple[float, float]:
        """Analyze reset frequency measurement results.

        Parameters
        ----------
        result : Optional[SingleToneResetFreqResultType]
            Measurement result. If None, uses last result.
        type : str, default="lor"
            Fitting function type ("lor" or "sinc").
        asym : bool, default=False
            Whether to use asymmetric fitting.
        plot : bool, default=True
            Whether to show analysis plot.
        max_contrast : bool, default=True
            Whether to use maximum contrast for analysis.

        Returns
        -------
        Tuple[float, float]
            Reset frequency (MHz) and linewidth (MHz).
        """
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        fpts, signals = result

        # Discard NaNs (possible early abort)
        val_mask = ~np.isnan(signals)
        fpts = fpts[val_mask]
        signals = signals[val_mask]

        y = rotate2real(signals).real if max_contrast else np.abs(signals)

        freq, freq_err, kappa, _, y_fit, _ = fit_resonence_freq(
            fpts, y, type=type, asym=asym
        )

        if plot:
            plt.figure(figsize=config.figsize)
            plt.tight_layout()
            plt.plot(fpts, y, label="signal", marker="o", markersize=3)
            plt.plot(fpts, y_fit, label=f"fit, κ = {kappa:.1g} MHz")
            label = f"f_reset = {freq:.5g} ± {freq_err:.1g} MHz"
            plt.axvline(freq, color="r", ls="--", label=label)
            plt.xlabel("Frequency (MHz)")
            plt.ylabel("Signal Real (a.u.)" if max_contrast else "Magnitude (a.u.)")
            plt.title("Reset frequency optimization")
            plt.legend()
            plt.grid(True)
            plt.show()

        return freq, kappa

    def save(
        self,
        filepath: str,
        result: Optional[SingleToneResetFreqResultType] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/reset/single_tone/freq",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        fpts, signals = result

        save_data(
            filepath=filepath,
            x_info={"name": "Frequency", "unit": "Hz", "values": fpts * 1e6},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals},
            comment=comment,
            tag=tag,
            **kwargs,
        )
