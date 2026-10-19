from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Annotated, ClassVar

import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray
from pydantic import field_serializer
from zcu_tools.analysis.singleshot import (
    calc_populations,
    correct_populations,
    raw_population_signal,
)
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import (
    IDENTITY,
    MHZ_TO_HZ,
    AxesSpec,
    Axis,
    PersistableExperiment,
    ZSpec,
)
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    ProgramV2Cfg,
    PulseCfg,
    ReadoutCfg,
    ResetCfg,
    SweepCfg,
    sweep2param,
)


def _default_population_states() -> NDArray[np.int64]:
    return np.array([0, 1], dtype=np.int64)


@dataclass(frozen=True)
class PreFreqResult:
    freqs: NDArray[np.float64]
    signals: NDArray[np.float64]
    population_states: NDArray[np.int64] = field(
        default_factory=_default_population_states
    )


class PreFreqModuleCfg(ConfigBase):
    reset: Annotated[ResetCfg | None, FieldMeta(label="Reset")] = None
    init_pulse: Annotated[PulseCfg, FieldMeta(label="Preparation pulse")]
    pi_pulse: Annotated[PulseCfg | None, FieldMeta(label="Pi pulse")] = None
    probe_pulse: Annotated[PulseCfg, FieldMeta(label="MIST probe pulse")]
    readout: Annotated[ReadoutCfg, FieldMeta(label="Readout")]


class PreFreqSweepCfg(ConfigBase):
    freq: Annotated[SweepCfg, FieldMeta(label="Preparation freq", unit="MHz")]


class PreFreqCfg(ProgramV2Cfg, ExpCfgModel):
    modules: PreFreqModuleCfg
    sweep: PreFreqSweepCfg
    g_center: Annotated[complex, FieldMeta(label="Ground center")]
    e_center: Annotated[complex, FieldMeta(label="Excited center")]
    radius: Annotated[float, FieldMeta(label="Classification radius")]

    @field_serializer("g_center", "e_center")
    def serialize_center(self, value: complex) -> str:
        return str(value)


@dataclass(frozen=True)
class PreFreqAnalyzeOptions:
    confusion_matrix: NDArray[np.float64] | None = None


class PreFreqExp(PersistableExperiment[PreFreqResult, PreFreqCfg]):
    Options: ClassVar[type[PreFreqAnalyzeOptions]] = PreFreqAnalyzeOptions

    AXES_SPEC = AxesSpec(
        axes=(
            Axis(
                "population_states",
                "GE Population",
                "None",
                scale=IDENTITY,
                dtype=np.int64,
            ),
            Axis(
                "freqs",
                "PrePulse frequency",
                "Hz",
                scale=MHZ_TO_HZ,
                dtype=np.float64,
            ),
        ),
        z=ZSpec("signals", "Population", "a.u.", dtype=np.float64),
        result_type=PreFreqResult,
        cfg_type=PreFreqCfg,
        cfg_schema_version="2.0",
        tag="singleshot/mist/pre_freq",
    )

    def run(self, cfg: PreFreqCfg, *, context: RunContext) -> PreFreqResult:
        """Measure MIST populations versus the preparation-pulse frequency.

        After optional reset, sweep init_pulse frequency on the board, apply
        the optional pi pulse, then the fixed probe pulse and readout.
        Return rounded preparation frequencies (MHz) and classified G/E
        fractions shaped (Nfreq, 2), averaged over shots and rounds.

        cfg supplies distinct finite IQ centers and a positive finite radius.
        Other is 1-G-E, not calibrated leakage; optional confusion correction
        belongs to subsequent analysis.
        """

        # Sweep the preparation tone, not the later MIST probe.
        soc, soccfg = context.soc, context.soccfg
        cfg = deepcopy(cfg)
        modules = cfg.modules

        freqs = sweep2array(
            cfg.sweep.freq,
            "freq",
            {"soccfg": soccfg, "gen_ch": modules.init_pulse.ch},
        )

        # Show G/E and the unclassified remainder without altering saved fractions.
        def configure_axes(ax: Axes) -> None:
            ax.set_ylim(0.0, 1.0)
            for line, label in zip(
                ax.lines, ("Ground", "Excited", "Other"), strict=True
            ):
                line.set_label(label)
            ax.legend()

        viewer = context.plots.liveplot_1d(
            "measurement",
            "Pre Pulse Frequency",
            "MIST",
            num_lines=3,
            configure_axes=configure_axes,
        )
        buffer = SignalBuffer(
            (len(freqs), 2),
            dtype=np.float64,
            on_update=lambda data: viewer.update(freqs, calc_populations(data).T),
        )
        with Schedule(cfg, buffer, stop=context.cancel_signal) as sched:
            run_cfg = sched.cfg
            modules = run_cfg.modules
            freq_sweep = run_cfg.sweep.freq
            modules.init_pulse.set_param("freq", sweep2param("freq", freq_sweep))

            # Vary initialization before the optional pi pulse and fixed probe.
            _ = (
                sched.prog_builder(soc, soccfg)
                .add_reset("reset", modules.reset)
                .add_pulse("init_pulse", modules.init_pulse)
                .add_pulse("pi_pulse", modules.pi_pulse)
                .add_pulse("probe_pulse", modules.probe_pulse)
                .add_readout("readout", modules.readout)
                .declare_sweep("freq", freq_sweep)
                .build_and_acquire(
                    # Classify individual shots, then average their G/E fractions.
                    raw2signal_fn=raw_population_signal,
                    g_center=cfg.g_center,
                    e_center=cfg.e_center,
                    ge_radius=cfg.radius,
                )
            )

        # Store G/E only; the Other curve is derived as 1-G-E.
        signals = buffer.array

        return PreFreqResult(freqs=freqs, signals=signals)

    def analyze(
        self,
        source: RunRecord[PreFreqCfg, PreFreqResult],
        options: PreFreqAnalyzeOptions,
        *,
        plots: Plots,
    ) -> None:
        """Plot G/E/Other populations versus preparation frequency in MHz.

        Use the explicit run's G/E fractions, derive Other as 1-G-E, and apply
        the optional confusion matrix. Publish a named fit figure without
        estimating scalars. Historical data can be analyzed without cfg.
        """
        result = source.result
        freqs, populations = result.freqs, result.signals

        populations = calc_populations(populations)

        populations = correct_populations(populations, options.confusion_matrix)

        _, ax = plots.subplots("fit", figsize=(6, 6))

        ax.plot(
            freqs,
            populations[:, 0],
            color="blue",
            label="Ground",
            ls="-",
            marker="o",
            markersize=1,
        )
        ax.plot(
            freqs,
            populations[:, 1],
            color="red",
            label="Excited",
            ls="-",
            marker="o",
            markersize=1,
        )
        ax.plot(
            freqs,
            populations[:, 2],
            color="green",
            label="Other",
            ls="-",
            marker="o",
            markersize=1,
        )
        ax.set_xlabel("Frequency (MHz)", fontsize=14)
        ax.set_ylabel("Population", fontsize=14)
        ax.grid(True)
        ax.tick_params(axis="both", which="major", labelsize=12)
        ax.set_ylim(0, 1)
