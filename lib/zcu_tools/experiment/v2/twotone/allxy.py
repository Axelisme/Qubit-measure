from __future__ import annotations

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit
from typeguard import check_type
from typing_extensions import (
    Any,
    Callable,
    Mapping,
    NotRequired,
    Optional,
    TypeAlias,
    TypedDict,
    cast,
)

from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.v2.runner import BatchTask, Task, TaskCfg, TaskState, run_task
from zcu_tools.liveplot import LivePlot1D
from zcu_tools.program.v2 import (
    ModularProgramCfg,
    ModularProgramV2,
    Pulse,
    PulseCfg,
    Readout,
    ReadoutCfg,
    Reset,
    ResetCfg,
)
from zcu_tools.utils.datasaver import load_data, save_data
from zcu_tools.utils.process import rotate2real

# (sequence, signals)
AllXY_Result: TypeAlias = dict[tuple[str, str], NDArray[np.complex128]]

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


def allxy_signal2real(
    signals_dict: Mapping[tuple[str, str], NDArray[np.complex128]],
) -> NDArray[np.float64]:
    all_signals = np.array(list(signals_dict.values()))
    return rotate2real(all_signals).real


# ------------------------------------------------------------------------------
# AllXYExperiment
# ------------------------------------------------------------------------------


class AllXY_ModuleCfg(TypedDict, closed=True):
    reset: NotRequired[ResetCfg]
    I_pulse: NotRequired[PulseCfg]
    X180_pulse: PulseCfg
    Y180_pulse: PulseCfg
    X90_pulse: PulseCfg
    Y90_pulse: PulseCfg
    readout: ReadoutCfg


class AllXY_Cfg(ModularProgramCfg, TaskCfg):
    modules: AllXY_ModuleCfg


class AllXY_Exp(AbsExperiment[AllXY_Result, AllXY_Cfg]):
    def run(
        self,
        soc,
        soccfg,
        cfg: dict[str, Any],
        *,
        acquire_kwargs: Optional[dict[str, Any]] = None,
    ) -> AllXY_Result:
        _cfg = check_type(deepcopy(cfg), AllXY_Cfg)

        rounds = _cfg["rounds"]
        _cfg["rounds"] = 1  # We'll handle the rounds in the task loop

        prog_cache: dict[tuple[str, str], ModularProgramV2] = {}
        def make_task(gate1: str, gate2: str) -> Task:

            def measure_fn(
                ctx: TaskState,
                update_hook: Optional[Callable[[int, list[NDArray[np.float64]]], None]],
            ) -> list[NDArray[np.float64]]:
                cfg = cast(AllXY_Cfg, ctx.cfg)
                modules = cfg["modules"]

                gate2pulse_map = {
                    "I": modules.get("I_pulse"),
                    "X180": modules["X180_pulse"],
                    "Y180": modules["Y180_pulse"],
                    "X90": modules["X90_pulse"],
                    "Y90": modules["Y90_pulse"],
                }

                if (gate1, gate2) not in prog_cache:
                    prog_cache[(gate1, gate2)] =  ModularProgramV2(
                        soccfg,
                        cfg,
                        modules=[
                            Reset("reset", modules.get("reset")),
                            Pulse("first_pulse", gate2pulse_map[gate1]),
                            Pulse("second_pulse", gate2pulse_map[gate2]),
                            Readout("readout", modules["readout"]),
                        ],
                    )


                return prog_cache[(gate1, gate2)].acquire(
                    soc,
                    progress=False,
                    callback=update_hook,
                    **(acquire_kwargs or {}),
                )

            return Task(measure_fn=measure_fn)

        def average_round(signals: list[dict[tuple[str, str], NDArray[np.complex128]]]) -> dict[tuple[str, str], NDArray[np.complex128]]:
            avg_signals: dict[tuple[str, str], NDArray[np.complex128]] = {}
            for gate_pair in ALLXY_SEQUENCE:
                gate_signals = [sig[gate_pair] for sig in signals]
                if np.all(np.isnan(gate_signals)):
                    avg_signals[gate_pair] = np.full_like(gate_signals[0], np.nan)
                else:
                    avg_signals[gate_pair] = np.nanmean(gate_signals, axis=0)
            return avg_signals

        with LivePlot1D(
            xlabel="Gate",
            ylabel="Signal",
            segment_kwargs=dict(
                show_grid=True,
                line_kwargs=[dict(marker=".", linestyle=None, markersize=5)],
            ),
        ) as viewer:
            # Configure x-axis labels
            ax = viewer.get_ax()
            ax.set_xticks(np.arange(len(ALLXY_SEQUENCE)))
            ax.set_xticklabels(
                [f"({gate1}, {gate2})" for gate1, gate2 in ALLXY_SEQUENCE],
                rotation=45,
                ha="right",
            )

            signals_dict = run_task(
                task=BatchTask(
                    tasks={
                        (gate1, gate2): make_task(gate1, gate2)
                        for gate1, gate2 in ALLXY_SEQUENCE
                    }
                ).scan(
                    "round",
                    list(range(rounds)),
                    before_each=lambda *_: None,
                ),
                init_cfg=_cfg,
                on_update=lambda ctx: viewer.update(
                    np.arange(len(ALLXY_SEQUENCE), dtype=np.float64),
                    allxy_signal2real(average_round(ctx.root_data)),
                ),
            )
            signals_dict = average_round(signals_dict)

        # Cache results
        self.last_cfg = check_type(deepcopy(cfg), AllXY_Cfg)
        self.last_result = signals_dict

        return signals_dict

    def analyze(
        self, result: Optional[AllXY_Result] = None, fit_ge: bool = False
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

    def load(self, filepath: str, **kwargs) -> AllXY_Result:
        signals, gate_indices, _ = load_data(filepath, **kwargs)
        assert gate_indices is not None
        assert len(gate_indices.shape) == 1 and len(signals.shape) == 1
        assert len(gate_indices) == len(ALLXY_SEQUENCE)
        assert signals.shape == gate_indices.shape

        signals = signals.astype(np.complex128)

        # Reconstruct signals_dict from flat signals array
        signals_dict: AllXY_Result = {}
        for i, seq in enumerate(ALLXY_SEQUENCE):
            signals_dict[seq] = signals[i : i + 1]

        self.last_cfg = None
        self.last_result = signals_dict

        return signals_dict
