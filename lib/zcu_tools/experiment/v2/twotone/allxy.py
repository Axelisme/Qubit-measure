from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.liveplot import LivePlotter1D
from zcu_tools.program.v2 import ModularProgramV2, Pulse, make_readout, make_reset
from zcu_tools.utils.datasaver import save_data
from zcu_tools.utils.process import rotate2real

from ..template import sweep1D_soft_template

# (sequence, signals)
AllXYResultType = Tuple[List[Tuple[str, str]], np.ndarray]


def allxy_signal2real(signals: np.ndarray) -> np.ndarray:
    """Convert complex signals to real values for AllXY analysis."""
    return rotate2real(signals).real  # type: ignore


def calc_predicted_state(gate1: str, gate2: str) -> np.ndarray:
    """Calculate the expected ⟨σ_z⟩ after gate pair (gate1, gate2).

    The calculation is performed by treating the qubit as a Bloch vector that
    starts at (0, 0, 1) (ground state). Each gate is mapped to a classical
    rotation on the Bloch sphere and applied in the order they appear in the
    gate pair.

    Parameters
    ----------
    gate1 : str
        First gate in the pair.
    gate2 : str
        Second gate in the pair.

    Returns
    -------
    np.ndarray
        Array of predicted ⟨σ_z⟩ values, taking values in {+1, 0, –1} for ideal gates.
    """

    def _rotation_x(theta: float) -> np.ndarray:
        """Rotation matrix around x-axis for Bloch vector representation."""

        return np.array(
            [
                [1, 0, 0],
                [0, np.cos(theta), -np.sin(theta)],
                [0, np.sin(theta), np.cos(theta)],
            ]
        )

    def _rotation_y(theta: float) -> np.ndarray:
        """Rotation matrix around y-axis for Bloch vector representation."""

        return np.array(
            [
                [np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)],
            ]
        )

    # Map each gate to its corresponding rotation matrix
    gate2rot = {
        "I": np.identity(3),
        "X180": _rotation_x(np.pi),
        "Y180": _rotation_y(np.pi),
        "X90": _rotation_x(np.pi / 2),
        "Y90": _rotation_y(np.pi / 2),
    }

    vec = np.array([0.0, 0.0, 1.0])
    vec = gate2rot[gate1] @ vec
    vec = gate2rot[gate2] @ vec

    return vec[2]


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

    def run(
        self, soc, soccfg, cfg: Dict[str, Any], *, progress: bool = True
    ) -> AllXYResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

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

        sequence = self.ALLXY_SEQUENCE

        def updateCfg(cfg: Dict[str, Any], i: int, _: Any) -> None:
            """Update configuration for each gate pair in the sequence."""
            cfg["current_gates"] = sequence[i]

        def measure_fn(cfg: Dict[str, Any], callback) -> np.ndarray:
            """Measure function for each gate pair."""
            gate1, gate2 = cfg["current_gates"]

            prog = ModularProgramV2(
                soccfg,
                soc,
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
            line_kwargs=[dict(marker="o", linestyle="None", markersize=5)],
            disable=not progress,
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

        # Execute soft sweep over all gate pairs
        _, signals = sweep1D_soft_template(
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

    def analyze(self, result: Optional[AllXYResultType] = None) -> Tuple[float, float]:
        if result is None:
            result = self.last_result
        assert result is not None, (
            "No measurement data available. Run experiment first."
        )

        sequence, signals = result

        # Rotate IQ data so that the contrast lies on the real axis and take only
        # the real part for further analysis.
        signals = rotate2real(signals).real

        # ------------------------------------------------------------------
        # 1. Calculate the ideal expectation values (+1, 0, –1)
        # ------------------------------------------------------------------
        predicted_state = np.array([calc_predicted_state(*gates) for gates in sequence])

        # ------------------------------------------------------------------
        # 2. Use non-linear least squares to fit average signal and contrast
        #    such that:  s_pred = avg + 0.5 * contrast * predicted_state
        # ------------------------------------------------------------------

        def calc_predicted(params: np.ndarray) -> np.ndarray:
            avg, contrast = params
            return avg + 0.5 * contrast * predicted_state

        # Initial guess: use simple statistics of the measured signals
        avg_guess = float(np.mean(signals))
        contrast_guess = float(np.ptp(signals))

        res = least_squares(
            lambda p: calc_predicted(p) - signals, x0=[avg_guess, contrast_guess]
        )

        avg_signal, contrast = res.x  # fitted parameters

        # ------------------------------------------------------------------
        # 3. Plotting
        # ------------------------------------------------------------------

        _, ax = plt.subplots(figsize=config.figsize)
        ax.plot(signals, marker="o", linestyle="None", label="Measured Signals")
        ax.plot(
            calc_predicted([avg_signal, contrast]),
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
