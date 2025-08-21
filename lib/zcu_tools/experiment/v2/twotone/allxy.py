from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.liveplot import LivePlotter1D
from zcu_tools.program.v2 import ModularProgramV2, Pulse, make_readout, make_reset
from zcu_tools.utils.datasaver import save_data
from zcu_tools.utils.process import rotate2real

from ..template import sweep1D_soft_template

# (sequence, signals)
AllXYResultType = Tuple[List[Tuple[str, str]], np.ndarray]

# Standard AllXY sequence of 21 gate pairs
ALLXY_SEQUENCE = [
    ("I", "I"),
    ("X180", "X180"),
    ("Y180", "Y180"),
    ("X180", "Y180"),
    ("Y180", "X180"),
    ("X90", "I"),
    ("Y90", "I"),
    ("X90", "Y90"),
    ("Y90", "X90"),
    ("X90", "Y180"),
    ("Y90", "X180"),
    ("X180", "Y90"),
    ("Y180", "X90"),
    ("X90", "X180"),
    ("X180", "X90"),
    ("Y90", "Y180"),
    ("Y180", "Y90"),
    ("X180", "I"),
    ("Y180", "I"),
    ("X90", "X90"),
    ("Y90", "Y90"),
]

# ------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------


def predict_state_with_error(
    gates: Tuple[str, str], power_err: float, detune_err: float
) -> float:
    ep = power_err
    ed = detune_err

    # reference: https://rsl.yale.edu/sites/default/files/2024-08/2013-RSL-Thesis-Matthew-Reed.pdf
    # page 154

    if gates == ("I", "I"):
        return 1
    elif gates in [("X180", "X180"), ("Y180", "Y180")]:
        return 1 - 8 * ep**2 - (np.pi**2 / 32) * ed**4
    elif gates in [("X180", "Y180"), ("Y180", "X180")]:
        return 1 - 4 * ep**2 - ed**2
    elif gates in [("X90", "I"), ("Y90", "I"), ("I", "X90"), ("I", "Y90")]:
        return -ep + (1 - np.pi / 2) * ed**2
    elif gates == ("X90", "Y90"):
        return ep**2 - 2 * ed
    elif gates == ("Y90", "X90"):
        return ep**2 + 2 * ed
    elif gates in [("X90", "Y180"), ("X180", "Y90")]:
        return ep - ed
    elif gates in [("Y90", "X180"), ("Y180", "X90")]:
        return ep + ed
    elif gates in [("X90", "X180"), ("X180", "X90"), ("Y90", "Y180"), ("Y180", "Y90")]:
        return 3 * ep + (3 * np.pi / 8) * ed**2
    elif gates in [("X180", "I"), ("Y180", "I"), ("I", "X180"), ("I", "Y180")]:
        return -1 + 2 * ep**2 + 0.5 * ed**2
    elif gates in [("X90", "X90"), ("Y90", "Y90")]:
        return -1 + 2 * ep**2 + 2 * ed**2
    else:
        raise ValueError(f"Invalid gate pair: {gates}")


def allxy_signal2real(signals: np.ndarray) -> np.ndarray:
    """Convert complex signals to real values for AllXY analysis."""
    return rotate2real(signals).real  # type: ignore


# ------------------------------------------------------------------------------
# AllXYExperiment
# ------------------------------------------------------------------------------


class AllXYExperiment(AbsExperiment[AllXYResultType]):
    """AllXY gate characterization experiment.

    Performs a comprehensive test of single-qubit gates using the AllXY sequence,
    which consists of 21 specific gate pairs designed to reveal various types of
    gate errors including calibration errors, cross-talk, and decoherence.

    The experiment performs:
    1. Initial reset (optional)
    2. First gate from the gate pair
    3. Second gate from the gate pair
    4. Readout to measure final state

    Each gate pair is executed in sequence using a soft sweep approach.
    """

    def run(
        self, soc, soccfg, cfg: Dict[str, Any], *, progress: bool = True
    ) -> AllXYResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        assert cfg.get("sweep", dict()) == {}, (
            "AllXY experiment does not support sweep configurations. "
            "Please remove 'sweep' key from the configuration."
        )

        # Create gate-to-pulse mapping from configuration
        gate2pulse_map = {
            "I": cfg.get("I_pulse"),
            "X180": cfg["X180_pulse"],
            "Y180": cfg["Y180_pulse"],
            "X90": cfg["X90_pulse"],
            "Y90": cfg["Y90_pulse"],
        }

        # Validate that all required gates are defined
        for gate_name, pulse_cfg in gate2pulse_map.items():
            if gate_name != "I" and pulse_cfg is None:
                raise ValueError(f"Gate '{gate_name}' pulse configuration is missing")

        sequence = ALLXY_SEQUENCE

        def updateCfg(cfg: Dict[str, Any], i: int, _: Any) -> None:
            """Update configuration for each gate pair in the sequence."""
            cfg["current_gates"] = sequence[i]

        def measure_fn(cfg: Dict[str, Any], callback) -> np.ndarray:
            """Measure function for each gate pair."""
            gate1, gate2 = cfg["current_gates"]

            prog = ModularProgramV2(
                soccfg,
                cfg,
                modules=[
                    make_reset("reset", reset_cfg=cfg.get("reset")),
                    Pulse(name="first_pulse", cfg=gate2pulse_map[gate1]),
                    Pulse(name="second_pulse", cfg=gate2pulse_map[gate2]),
                    make_readout("readout", readout_cfg=cfg["readout"]),
                ],
            )

            result = prog.acquire(soc, progress=False, callback=callback)
            return result[0][0].dot([1, 1j])

        # Set up live plotter with gate labels
        liveplotter = LivePlotter1D(
            xlabel="Gate",
            ylabel="Signal",
            disable=not progress,
            line_kwargs=[dict(marker="o", linestyle="None", markersize=5)],
        )

        # Configure x-axis labels if plotter is available
        if not liveplotter.disable and liveplotter.axs:
            ax = liveplotter.axs[0]
            if isinstance(ax, plt.Axes):
                ax.set_xticks(np.arange(len(sequence)))
                ax.set_xticklabels(
                    [f"({gate1}, {gate2})" for gate1, gate2 in sequence],
                    rotation=45,
                    ha="right",
                )
                ax.grid(True)

        # Execute soft sweep over all gate pairs
        signals = sweep1D_soft_template(
            cfg,
            measure_fn,
            liveplotter,
            xs=np.arange(len(sequence)),
            updateCfg=updateCfg,
            signal2real=allxy_signal2real,
            progress=progress,
        )

        # Cache results
        self.last_cfg = cfg
        self.last_result = (sequence, signals)

        return sequence, signals

    def analyze(
        self, result: Optional[AllXYResultType] = None, fit_contrast: bool = False
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, (
            "No measurement data available. Run experiment first."
        )

        sequence, signals = result

        # Rotate IQ data so that the contrast lies on the real axis and take only
        # the real part for further analysis.
        signals = allxy_signal2real(signals)

        # ------------------------------------------------------------------
        # fitting the signal with error
        # ------------------------------------------------------------------

        g_signal = signals[sequence.index(("I", "I"))]
        init_contrast = (
            np.ptp(signals) if g_signal < np.mean(signals) else -np.ptp(signals)
        )

        if fit_contrast:
            params, _ = curve_fit(
                lambda idxs, contrast, ep, ed: [
                    g_signal
                    + 0.5
                    * contrast
                    * (1 - predict_state_with_error(sequence[int(i)], ep, ed))
                    for i in idxs
                ],
                np.arange(len(sequence)),
                signals,
                p0=(init_contrast, 0.0, 0.0),
            )

            contrast, ep, ed = params
        else:
            contrast = init_contrast

            params, _ = curve_fit(
                lambda idxs, ep, ed: [
                    g_signal
                    + 0.5
                    * contrast
                    * (1 - predict_state_with_error(sequence[int(i)], ep, ed))
                    for i in idxs
                ],
                np.arange(len(sequence)),
                signals,
                p0=(0.0, 0.0),
            )

            ep, ed = params

        predict_signals = [
            g_signal + 0.5 * contrast * (1 - predict_state_with_error(seq, ep, ed))
            for seq in sequence
        ]

        # ------------------------------------------------------------------
        # calculate the error
        # ------------------------------------------------------------------
        perfect_states = [predict_state_with_error(seq, 0.0, 0.0) for seq in sequence]
        power_err = np.mean(
            [
                np.abs(predict_state_with_error(seq, ep, 0.0) - perf_state)
                for seq, perf_state in zip(sequence, perfect_states)
            ]
        )
        detune_err = np.mean(
            [
                np.abs(predict_state_with_error(seq, 0.0, ed) - perf_state)
                for seq, perf_state in zip(sequence, perfect_states)
            ]
        )

        # ------------------------------------------------------------------
        # 3. Plotting
        # ------------------------------------------------------------------

        _, ax = plt.subplots(figsize=config.figsize)
        ax.plot(signals, marker="o", linestyle="None", label="Measured Signals")
        ax.plot(
            predict_signals,
            marker="x",
            linestyle="-",
            color="red",
            label="Predicted Signals",
        )

        ax.set_xlabel("Gate")
        ax.set_xticks(np.arange(len(sequence)))
        ax.set_xticklabels([f"{g1}-{g2}" for g1, g2 in sequence], rotation=45)

        ax.set_ylabel("Signal")
        ax.legend()
        ax.grid(True)

        ax.set_title(f"power dep: {power_err:.1%}, detune dep: {detune_err:.1%}")

        plt.tight_layout()
        plt.show()

    def save(
        self,
        filepath: str,
        result: Optional[AllXYResultType] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/ge/allxy",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, (
            "No measurement data available. Run experiment first."
        )

        sequence, signals = result

        # Create gate indices and labels
        gate_indices = np.arange(len(sequence))
        gate_labels = [f"{gate1}-{gate2}" for gate1, gate2 in sequence]

        save_data(
            filepath=filepath,
            x_info={
                "name": "Gate Pair Index",
                "unit": "",
                "values": gate_indices,
                "labels": gate_labels,  # Include gate labels for reference
            },
            z_info={"name": "Signal", "unit": "a.u.", "values": signals},
            comment=comment,
            tag=tag,
            **kwargs,
        )
