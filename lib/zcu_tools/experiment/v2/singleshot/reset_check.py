from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray
from pydantic import field_serializer

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment import (
    AxesSpec,
    Axis,
    PersistableExperiment,
    ZSpec,
)
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.utils import setup_devices
from zcu_tools.experiment.v2.runtime import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    Branch,
    Module,
    ProgramV2Cfg,
    Pulse,
    PulseCfg,
    Readout,
    ReadoutCfg,
    Reset,
    ResetCfg,
    SweepCfg,
    sweep2param,
)

from .util import (
    calc_populations,
    classify_result,
    correct_populations,
    raw_population_signal,
)

BRANCH_LABELS = ("Before reset", "Reset only", "Reset + Rabi")
BRANCH_LINESTYLES = ("-", "--", ":")


class ResetCheckModuleCfg(ConfigBase):
    reset: ResetCfg | None = None
    rabi_pulse: PulseCfg
    tested_reset: ResetCfg
    readout: ReadoutCfg


class ResetCheckSweepCfg(ConfigBase):
    gain: SweepCfg


class ResetCheckCfg(ProgramV2Cfg, ExpCfgModel):
    modules: ResetCheckModuleCfg
    sweep: ResetCheckSweepCfg
    g_center: complex
    e_center: complex
    radius: float

    @field_serializer("g_center", "e_center")
    def serialize_center(self, value: complex) -> str:
        """Keep experiment comments JSON-safe with a lossless complex literal."""
        return str(value)


@dataclass(frozen=True)
class ResetCheckResult:
    """Classified G/E fractions in (gain, reset_state, population_state) order.

    Branches are independent preparations, not paired shot trajectories.
    """

    gains: NDArray[np.float64]
    reset_states: NDArray[np.int64]
    signals: NDArray[np.float64]
    population_states: NDArray[np.int64] = field(
        default_factory=lambda: np.array([0, 1], dtype=np.int64)
    )


@dataclass(frozen=True)
class ResetCheckAnalysis:
    populations: NDArray[np.float64]
    reset_mean_excited_population: float
    reset_max_excited_population: float
    reset_max_other_population: float
    worst_sample_gain: float
    analyzed_reset_points: int


def _reset_check_sequence(
    modules: ResetCheckModuleCfg, gain_sweep: SweepCfg
) -> tuple[Module, ...]:
    modules.rabi_pulse.set_param("gain", sweep2param("gain", gain_sweep))
    return (
        Reset("reset", modules.reset),
        Pulse("rabi_pulse", modules.rabi_pulse),
        Branch(
            "reset_sel",
            [],
            Reset("tested_reset_1", modules.tested_reset),
            [
                Reset("tested_reset_2", modules.tested_reset),
                Pulse("rabi_pulse_after_reset", modules.rabi_pulse),
            ],
        ),
        Readout("readout", modules.readout),
    )


def _population_line_kwargs() -> list[dict[str, str]]:
    return [
        {"label": f"{label}: {state}", "color": color, "linestyle": style}
        for label, style in zip(BRANCH_LABELS, BRANCH_LINESTYLES, strict=True)
        for state, color in (("Ground", "blue"), ("Excited", "red"), ("Other", "green"))
    ]


@dataclass(frozen=True)
class ResetCheckAnalyzeOptions:
    confusion_matrix: NDArray[np.float64] | None = None


class ResetCheckExp(PersistableExperiment[ResetCheckResult, ResetCheckCfg]):
    Options: ClassVar[type[ResetCheckAnalyzeOptions]] = ResetCheckAnalyzeOptions

    AXES_SPEC = AxesSpec(
        axes=(
            Axis("population_states", "GE Population", "None", dtype=np.int64),
            Axis("reset_states", "Reset State", "None", dtype=np.int64),
            Axis("gains", "Gain", "a.u.", dtype=np.float64),
        ),
        z=ZSpec("signals", "Population", "a.u.", dtype=np.float64),
        result_type=ResetCheckResult,
        cfg_type=ResetCheckCfg,
        tag="singleshot/reset_check",
    )

    def run(
        self,
        cfg: ResetCheckCfg,
        *,
        context: RunContext,
    ) -> ResetCheckResult:
        soc, soccfg = context.soc, context.soccfg
        g_center, e_center, radius = cfg.g_center, cfg.e_center, cfg.radius
        classify_result(np.empty(0, dtype=np.complex128), g_center, e_center, radius)
        cfg = deepcopy(cfg)
        setup_devices(
            cfg,
            context.devices,
            progress=True,
            cancel_signal=context.cancel_signal,
        )
        gains = sweep2array(
            cfg.sweep.gain,
            "gain",
            {"soccfg": soccfg, "gen_ch": cfg.modules.rabi_pulse.ch},
        )

        def configure_axes(ax: Axes) -> None:
            ax.set_ylim(-0.02, 1.02)
            for line, kwargs in zip(ax.lines, _population_line_kwargs(), strict=True):
                line.set(**kwargs)
            ax.legend()

        viewer = context.plots.liveplot_1d(
            "measurement",
            "Pulse gain (a.u.)",
            "Classified population",
            num_lines=9,
            configure_axes=configure_axes,
        )
        buffer = SignalBuffer(
            (gains.size, 3, 2),
            dtype=np.float64,
            on_update=lambda data: viewer.update(
                gains, calc_populations(data).reshape(gains.size, 9).T
            ),
        )
        with Schedule(cfg, buffer, stop=context.cancel_signal) as sched:
            (
                sched.prog_builder(soc, soccfg)
                .add(*_reset_check_sequence(sched.cfg.modules, sched.cfg.sweep.gain))
                .declare_sweep("gain", sched.cfg.sweep.gain)
                .declare_sweep("reset_sel", 3)
                .build_and_acquire(
                    raw2signal_fn=raw_population_signal,
                    g_center=g_center,
                    e_center=e_center,
                    ge_radius=radius,
                )
            )
            sched.trigger_update(flush=True)
        return ResetCheckResult(
            gains=gains,
            reset_states=np.arange(3, dtype=np.int64),
            signals=buffer.array,
        )

    def analyze(
        self,
        source: RunRecord[ResetCheckCfg, ResetCheckResult],
        options: ResetCheckAnalyzeOptions,
        *,
        plots: Plots,
    ) -> ResetCheckAnalysis:
        result = source.result
        confusion_matrix = options.confusion_matrix
        if (
            result.signals.shape != (result.gains.size, 3, 2)
            or not np.array_equal(result.reset_states, [0, 1, 2])
            or not np.array_equal(result.population_states, [0, 1])
        ):
            raise ValueError("Reset check requires (gain, branch, G/E) populations")
        populations = correct_populations(
            calc_populations(result.signals), confusion_matrix
        )
        reset_e = populations[:, 1, 1]
        valid = np.isfinite(result.gains) & np.isfinite(populations[:, 1]).all(axis=-1)
        if not valid.any():
            raise ValueError("No finite reset-only populations to analyze")
        worst = np.flatnonzero(valid)[np.argmax(reset_e[valid])]
        analysis = ResetCheckAnalysis(
            populations,
            float(np.mean(reset_e[valid])),
            float(reset_e[worst]),
            float(np.max(populations[valid, 1, 2])),
            float(result.gains[worst]),
            int(valid.sum()),
        )
        _, ax = plots.subplots("populations", figsize=(10, 6))
        order = np.argsort(result.gains)
        for values, kwargs in zip(
            populations[order].reshape(result.gains.size, 9).T,
            _population_line_kwargs(),
            strict=True,
        ):
            ax.plot(
                result.gains[order],
                values,
                marker=".",
                label=kwargs["label"],
                color=kwargs["color"],
                linestyle=kwargs["linestyle"],
            )
        ax.set(
            xlabel="Pulse gain (a.u.)",
            ylabel="Population"
            if confusion_matrix is not None
            else "Classified population",
            ylim=(-0.02, 1.02),
            title=f"Reset excited: mean {analysis.reset_mean_excited_population:.2%}, worst sampled {analysis.reset_max_excited_population:.2%}\nOther is not calibrated leakage; populations are not reset-channel fidelity",
        )
        ax.legend(ncol=3, fontsize=8)
        ax.grid(True)
        return analysis
