from __future__ import annotations

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy.optimize import curve_fit
from typing_extensions import Any, Callable, Optional, TypeAlias

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import make_comment, parse_comment, setup_devices
from zcu_tools.experiment.v2.runner import Task, TaskState, run_task
from zcu_tools.liveplot import LivePlot1D
from zcu_tools.program.v2 import (
    ComputedPulse,
    LoadValue,
    ModularProgramV2,
    ProgramV2Cfg,
    PulseCfg,
    Readout,
    ReadoutCfg,
    Reset,
    ResetCfg,
)
from zcu_tools.utils.datasaver import load_data, save_data
from zcu_tools.utils.process import rotate2real

# (signals in ALLXY_SEQUENCE order)
AllXY_Result: TypeAlias = NDArray[np.complex128]

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

GATE_LIST = ["I", "X90", "Y90", "X180", "Y180"]
ALLXY_GATE1_IDX = [GATE_LIST.index(g1) for g1, _ in ALLXY_SEQUENCE]
ALLXY_GATE2_IDX = [GATE_LIST.index(g2) for _, g2 in ALLXY_SEQUENCE]

# ------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------


def predict_state_with_error(
    gates: tuple[str, str], power_err: float, detune_err: float
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


def allxy_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return rotate2real(signals).real


# ------------------------------------------------------------------------------
# AllXYExperiment
# ------------------------------------------------------------------------------


class AllXYModuleCfg(ConfigBase):
    reset: Optional[ResetCfg] = None
    I_pulse: Optional[PulseCfg] = None
    X180_pulse: PulseCfg
    X90_pulse: PulseCfg
    readout: ReadoutCfg


class AllXYCfg(ProgramV2Cfg, ExpCfgModel):
    modules: AllXYModuleCfg


class AllXY_Exp(AbsExperiment[AllXY_Result, AllXYCfg]):
    def run(
        self,
        soc,
        soccfg,
        cfg: AllXYCfg,
        *,
        acquire_kwargs: Optional[dict[str, Any]] = None,
    ) -> AllXY_Result:
        setup_devices(cfg, progress=True)

        def measure_fn(
            ctx: TaskState[NDArray[np.complex128], Any, AllXYCfg],
            update_hook: Optional[Callable[[int, list[NDArray[np.float64]]], None]],
        ) -> list[NDArray[np.float64]]:
            cfg = ctx.cfg
            modules = cfg.modules

            I_pulse = modules.I_pulse
            X180_pulse = modules.X180_pulse
            X90_pulse = modules.X90_pulse
            Y180_pulse = X180_pulse.with_updates(phase=X180_pulse.phase + 90)
            Y90_pulse = X90_pulse.with_updates(phase=X90_pulse.phase + 90)

            if I_pulse is None:
                I_pulse = X90_pulse.with_updates(gain=0.0)

            # Order must match GATE_LIST = ["I", "X90", "Y90", "X180", "Y180"]
            gate_pulses = [I_pulse, X90_pulse, Y90_pulse, X180_pulse, Y180_pulse]

            return ModularProgramV2(
                soccfg,
                cfg,
                modules=[
                    LoadValue(
                        "load_gate1_idx",
                        values=ALLXY_GATE1_IDX,
                        idx_reg="allxy_idx",
                        val_reg="gate_idx1",
                    ),
                    LoadValue(
                        "load_gate2_idx",
                        values=ALLXY_GATE2_IDX,
                        idx_reg="allxy_idx",
                        val_reg="gate_idx2",
                    ),
                    Reset("reset", modules.reset),
                    ComputedPulse("gate1", val_reg="gate_idx1", pulses=gate_pulses),
                    ComputedPulse("gate2", val_reg="gate_idx2", pulses=gate_pulses),
                    Readout("readout", modules.readout),
                ],
                sweep=[("allxy_idx", len(ALLXY_SEQUENCE))],
            ).acquire(
                soc, progress=False, round_hook=update_hook, **(acquire_kwargs or {})
            )

        with LivePlot1D(
            xlabel="Gate",
            ylabel="Signal",
            segment_kwargs=dict(
                show_grid=True,
                line_kwargs=[dict(marker=".", linestyle=None, markersize=5)],
            ),
        ) as viewer:
            # Configure x-axis labels
            name_map = {
                "I": "$I$",
                "X90": "$X_{90}$",
                "Y90": "$Y_{90}$",
                "X180": "$X_{180}$",
                "Y180": "$Y_{180}$",
            }
            gate_labels = [
                f"{name_map[gate1]}-{name_map[gate2]}"
                for gate1, gate2 in ALLXY_SEQUENCE
            ]
            ax = viewer.get_ax()
            ax.set_xticks(np.arange(len(ALLXY_SEQUENCE)))
            ax.set_xticklabels(gate_labels, rotation=30, ha="right", fontsize=8)

            signals = run_task(
                task=Task(
                    measure_fn=measure_fn,
                    result_shape=(len(ALLXY_SEQUENCE),),
                    pbar_n=cfg.rounds,
                ),
                init_cfg=cfg,
                on_update=lambda ctx: viewer.update(
                    np.arange(len(ALLXY_SEQUENCE), dtype=np.float64),
                    allxy_signal2real(ctx.root_data),
                ),
            )

        # Cache results
        self.last_cfg = deepcopy(cfg)
        self.last_result = signals

        return signals

    def analyze(
        self, result: Optional[AllXY_Result] = None, fit_ge: bool = False
    ) -> Figure:
        if result is None:
            result = self.last_result
        assert result is not None, (
            "No measurement data available. Run experiment first."
        )

        signals = result

        # Rotate IQ data so that the contrast lies on the real axis and take only
        # the real part for further analysis.
        sequence = ALLXY_SEQUENCE
        real_signals = allxy_signal2real(signals)

        # ------------------------------------------------------------------
        # fitting the signal with error
        # ------------------------------------------------------------------

        g_signal = real_signals[sequence.index(("I", "I"))]
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

        fig, ax = plt.subplots(figsize=config.figsize)
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

        return fig

    def save(
        self,
        filepath: str,
        result: Optional[AllXY_Result] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/ge/allxy",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, (
            "No measurement data available. Run experiment first."
        )

        gate_idxs = np.arange(len(ALLXY_SEQUENCE))
        signals = result

        cfg = self.last_cfg
        assert cfg is not None
        comment = make_comment(cfg, comment)

        save_data(
            filepath=filepath,
            x_info={"name": "Gate Pair Index", "unit": "", "values": gate_idxs},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals},
            comment=comment,
            tag=tag,
            **kwargs,
        )

    def load(self, filepath: str, **kwargs) -> AllXY_Result:
        signals, gate_idxs, _, comment = load_data(
            filepath, return_comment=True, **kwargs
        )
        assert len(gate_idxs.shape) == 1 and len(signals.shape) == 1
        assert len(gate_idxs) == len(ALLXY_SEQUENCE)
        assert signals.shape == gate_idxs.shape

        signals = signals.astype(np.complex128)

        if comment is not None:
            cfg, _, _ = parse_comment(comment)

            if cfg is not None:
                self.last_cfg = AllXYCfg.validate_or_warn(cfg, source=filepath)
        self.last_result = signals

        return signals
