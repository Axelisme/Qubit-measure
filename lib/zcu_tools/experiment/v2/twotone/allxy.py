from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.liveplot import LivePlotter1D
from zcu_tools.program.v2 import ModularProgramV2, Pulse, Readout, Reset
from zcu_tools.utils.datasaver import save_data
from zcu_tools.utils.process import rotate2real

from ..runner import BatchTask, HardTask, Runner

# (sequence, signals)
AllXYResultType = Dict[Tuple[str, str], np.ndarray]

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


def allxy_signal2real(signals_dict: Dict[Tuple[str, str], np.ndarray]) -> np.ndarray:
    all_signals = np.array(list(signals_dict.values()))
    return rotate2real(all_signals).real  # type: ignore


# ------------------------------------------------------------------------------
# AllXYExperiment
# ------------------------------------------------------------------------------


class AllXYExperiment(AbsExperiment[AllXYResultType]):
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

        liveplotter = LivePlotter1D(
            xlabel="Gate",
            ylabel="Signal",
            disable=not progress,
            segment_kwargs=dict(
                show_grid=True,
                line_kwargs=[dict(marker="o", linestyle=None, markersize=5)],
            ),
        )

        # Configure x-axis labels if plotter is available
        if not liveplotter.disable:
            ax = liveplotter.get_ax()
            ax.set_xticks(np.arange(len(ALLXY_SEQUENCE)))
            ax.set_xticklabels(
                [f"({gate1}, {gate2})" for gate1, gate2 in ALLXY_SEQUENCE],
                rotation=45,
                ha="right",
            )

        with liveplotter as viewer:
            signals_dict = Runner(
                task=BatchTask(
                    tasks={
                        (gate1, gate2): HardTask(
                            measure_fn=lambda ctx, update_hook: (
                                ModularProgramV2(
                                    soccfg,
                                    ctx.cfg,
                                    modules=[
                                        Reset(
                                            "reset",
                                            ctx.cfg.get("reset", {"type": "none"}),
                                        ),
                                        Pulse("first_pulse", gate2pulse_map[gate1]),
                                        Pulse("second_pulse", gate2pulse_map[gate2]),
                                        Readout("readout", ctx.cfg["readout"]),
                                    ],
                                ).acquire(soc, progress=False, callback=update_hook)
                            )
                        )
                        for gate1, gate2 in ALLXY_SEQUENCE
                    }
                ),
                update_hook=lambda ctx: viewer.update(
                    np.arange(len(ALLXY_SEQUENCE)),
                    allxy_signal2real(ctx.get_data()),
                ),
            ).run(cfg)

        # Cache results
        self.last_cfg = cfg
        self.last_result = signals_dict

        return signals_dict

    def analyze(
        self, result: Optional[AllXYResultType] = None, fit_ge: bool = False
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, (
            "No measurement data available. Run experiment first."
        )

        signals_dict = result

        # Rotate IQ data so that the contrast lies on the real axis and take only
        # the real part for further analysis.
        sequence = list(signals_dict.keys())
        real_signals = allxy_signal2real(signals_dict)

        # ------------------------------------------------------------------
        # fitting the signal with error
        # ------------------------------------------------------------------

        g_signal = real_signals[signals_dict[("I", "I")]]
        init_center = 0.5 * (np.max(real_signals) + np.min(real_signals))
        init_contrast = np.ptp(real_signals)
        if g_signal < init_center:
            init_contrast = -init_contrast

        def calc_sim_signal(seq, center, contrast, ep, ed) -> float:
            return center + 0.5 * contrast * predict_state_with_error(seq, ep, ed)

        if fit_ge:
            params, *_ = curve_fit(
                lambda _, *args: [calc_sim_signal(seq, *args) for seq in sequence],
                np.arange(len(sequence)),
                real_signals,
                p0=(init_center, init_contrast, 0.0, 0.0),
                bounds=(
                    (np.min(real_signals), -np.abs(init_contrast), -0.2, -0.2),
                    (np.max(real_signals), np.abs(init_contrast), 0.2, 0.2),
                ),
            )

            center, contrast, ep, ed = params
        else:
            center = init_center
            contrast = init_contrast

            params, *_ = curve_fit(
                lambda _, *args: [
                    calc_sim_signal(seq, center, contrast, *args) for seq in sequence
                ],
                np.arange(len(sequence)),
                real_signals,
                p0=(0.0, 0.0),
                bounds=((-0.2, -0.2), (0.2, 0.2)),
            )

            ep, ed = params

        predict_signals = [
            calc_sim_signal(seq, center, contrast, ep, ed) for seq in sequence
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
        ax.plot(real_signals, marker="o", linestyle="None", label="Measured Signals")
        ax.plot(
            predict_signals,
            marker="x",
            linestyle="-",
            color="red",
            label="Predicted Signals",
        )
        ax.axhline(y=center, color="green", linestyle="--", alpha=0.2)
        ax.axhline(y=center + 0.5 * contrast, color="blue", linestyle="--", alpha=0.2)
        ax.axhline(y=center - 0.5 * contrast, color="blue", linestyle="--", alpha=0.2)

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

        signals_dict = result
        sequence = list(signals_dict.keys())
        signals = np.concatenate(list(signals_dict.values()))

        # Create gate indices and labels
        gate_indices = np.arange(len(sequence))

        save_data(
            filepath=filepath,
            x_info={"name": "Gate Pair Index", "unit": "", "values": gate_indices},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals},
            comment=comment,
            tag=tag,
            **kwargs,
        )
