from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from matplotlib import rcParams
from matplotlib.axes import Axes
from numpy.typing import NDArray
from scipy.optimize import curve_fit
from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment import (
    IDENTITY,
    AxesSpec,
    Axis,
    PersistableExperiment,
    ZSpec,
    config,
)
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.utils import setup_devices
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    ComputedPulse,
    LoadValue,
    ProgramV2Cfg,
    PulseCfg,
    Readout,
    ReadoutCfg,
    Reset,
    ResetCfg,
)
from zcu_tools.utils.process import rotate2real


@dataclass(frozen=True)
class AllXY_Result:
    gate_idxs: NDArray[np.int64]
    signals: NDArray[np.complex128]


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


def _configure_allxy_axes(ax: Axes) -> None:
    # Configure x-axis labels
    name_map = {
        "I": "$I$",
        "X90": "$X_{90}$",
        "Y90": "$Y_{90}$",
        "X180": "$X_{180}$",
        "Y180": "$Y_{180}$",
    }
    gate_labels = [
        f"{name_map[gate1]}-{name_map[gate2]}" for gate1, gate2 in ALLXY_SEQUENCE
    ]
    ax.set_xticks(np.arange(len(ALLXY_SEQUENCE)))
    ax.set_xticklabels(gate_labels, rotation=30, ha="right", fontsize=8)
    ax.grid(True)
    line = ax.lines[0]
    line.set_marker(".")
    line.set_linestyle(rcParams["lines.linestyle"])
    line.set_markersize(5)


class AllXYModuleCfg(ConfigBase):
    reset: ResetCfg | None = None
    I_pulse: PulseCfg | None = None
    X180_pulse: PulseCfg
    X90_pulse: PulseCfg
    readout: ReadoutCfg


class AllXYCfg(ProgramV2Cfg, ExpCfgModel):
    modules: AllXYModuleCfg


@dataclass(frozen=True)
class AllXYAnalyzeOptions:
    fit_ge: bool = False


@dataclass(frozen=True)
class AllXYAnalysis:
    """Fitted gate errors.

    ``amplitude_error`` is the relative drive amplitude error: +0.02 means the
    pulses rotate 2% too far. ``detune_param`` is the dimensionless detuning
    parameter of the Reed thesis model.

    ``power_err``, ``detune_err`` and ``residual_rms`` are in Bloch-z units,
    where ground is +1 and excited is -1. ``power_err`` and ``detune_err`` are
    the mean absolute deviation each fitted error alone causes over the 21
    pairs; they are not gate infidelities. ``residual_rms`` is the RMS
    distance between the data and the fitted model. A residual comparable to
    the fitted deviations means the low-order model does not describe the
    data, for example because T1/T2 decay during the sequence matters.
    """

    amplitude_error: float
    detune_param: float
    power_err: float
    detune_err: float
    residual_rms: float


class AllXY_Exp(PersistableExperiment[AllXY_Result, AllXYCfg]):
    Options: ClassVar[type[AllXYAnalyzeOptions]] = AllXYAnalyzeOptions

    AXES_SPEC = AxesSpec(
        axes=(Axis("gate_idxs", "Gate Pair Index", "", IDENTITY, np.int64),),
        z=ZSpec("signals", "Signal", "a.u."),
        result_type=AllXY_Result,
        cfg_type=AllXYCfg,
        tag="twotone/ge/allxy",
    )

    def run(self, cfg: AllXYCfg, *, context: RunContext) -> AllXY_Result:
        cfg = deepcopy(cfg)
        soc, soccfg = context.soc, context.soccfg

        setup_devices(
            cfg,
            context.devices,
            progress=True,
            cancel_signal=context.cancel_signal,
        )

        viewer = context.plots.liveplot_1d(
            "measurement", "Gate", "Signal", configure_axes=_configure_allxy_axes
        )
        signals_buffer = SignalBuffer(
            (len(ALLXY_SEQUENCE),),
            on_update=lambda data: viewer.update(
                np.arange(len(ALLXY_SEQUENCE), dtype=np.float64),
                allxy_signal2real(data),
            ),
        )
        with Schedule(cfg, signals_buffer, stop=context.cancel_signal) as sched:
            modules = sched.cfg.modules
            I_pulse = modules.I_pulse
            X180_pulse = modules.X180_pulse
            X90_pulse = modules.X90_pulse
            Y180_pulse = X180_pulse.with_updates(phase=X180_pulse.phase + 90)
            Y90_pulse = X90_pulse.with_updates(phase=X90_pulse.phase + 90)

            if I_pulse is None:
                I_pulse = X90_pulse.with_updates(gain=0.0)

            # Order must match GATE_LIST = ["I", "X90", "Y90", "X180", "Y180"]
            gate_pulses = [I_pulse, X90_pulse, Y90_pulse, X180_pulse, Y180_pulse]

            _ = (
                sched.prog_builder(soc, soccfg)
                .add(
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
                    Reset("reset", cfg=modules.reset),
                    ComputedPulse("gate1", val_reg="gate_idx1", pulses=gate_pulses),
                    ComputedPulse("gate2", val_reg="gate_idx2", pulses=gate_pulses),
                    Readout("readout", cfg=modules.readout),
                )
                .declare_sweep("allxy_idx", len(ALLXY_SEQUENCE))
                .build_and_acquire()
            )

        return AllXY_Result(
            gate_idxs=np.arange(len(ALLXY_SEQUENCE), dtype=np.int64),
            signals=signals_buffer.array,
        )

    def analyze(
        self,
        source: RunRecord[AllXYCfg, AllXY_Result],
        options: AllXYAnalyzeOptions,
        *,
        plots: Plots,
    ) -> AllXYAnalysis:
        result = source.result
        fit_ge = options.fit_ge

        signals = result.signals

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
        # The model's ep is the pi/2 rotation-angle error, (pi/2) * amplitude error.
        amplitude_error = ep / (np.pi / 2)
        residual_rms = np.sqrt(
            np.mean((real_signals - np.asarray(predict_signals)) ** 2)
        ) / (0.5 * np.abs(contrast))

        # ------------------------------------------------------------------
        # calculate the error
        # ------------------------------------------------------------------
        perfect_states = [predict_state_with_error(seq, 0.0, 0.0) for seq in sequence]
        power_err = np.mean(
            [
                np.abs(predict_state_with_error(seq, ep, 0.0) - perf_state)
                for seq, perf_state in zip(sequence, perfect_states, strict=False)
            ]
        )
        detune_err = np.mean(
            [
                np.abs(predict_state_with_error(seq, 0.0, ed) - perf_state)
                for seq, perf_state in zip(sequence, perfect_states, strict=False)
            ]
        )

        # ------------------------------------------------------------------
        # 3. Plotting
        # ------------------------------------------------------------------

        fig, ax = plots.subplots("fit", figsize=config.figsize)
        ax.plot(real_signals, marker="o", linestyle="None", label="Measured Signals")
        ax.plot(
            predict_signals,
            marker="x",
            linestyle="-",
            color="red",
            label="Predicted Signals",
        )
        ax.axhline(y=float(center), color="green", linestyle="--", alpha=0.2)
        ax.axhline(
            y=float(center) + 0.5 * float(contrast),
            color="blue",
            linestyle="--",
            alpha=0.2,
        )
        ax.axhline(
            y=float(center) - 0.5 * float(contrast),
            color="blue",
            linestyle="--",
            alpha=0.2,
        )

        ax.set_xlabel("Gate")
        ax.set_xticks(np.arange(len(sequence)))
        ax.set_xticklabels([f"{g1}-{g2}" for g1, g2 in sequence], rotation=45)

        ax.set_ylabel("Signal")
        ax.legend()
        ax.grid(True)

        ax.set_title(
            f"amp err: {amplitude_error:+.2%}, power dep: {power_err:.1%}, "
            f"detune dep: {detune_err:.1%}, residual: {residual_rms:.1%}"
        )

        fig.tight_layout()

        return AllXYAnalysis(
            amplitude_error=float(amplitude_error),
            detune_param=float(ed),
            power_err=float(power_err),
            detune_err=float(detune_err),
            residual_rms=float(residual_rms),
        )
