"""Acquisition, analysis and persistence for zipped mux spectroscopy."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from pydantic import model_validator
from qick import QickConfig
from zcu_tools.experiment import PersistableExperiment
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.utils.round_zcu import round_zcu_freq, sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import ProgramV2Cfg

from zcu_lab.v2._support.mux import MuxFitFailure, MuxPulseScan, MuxSweepResult
from zcu_lab.v2._support.mux_acquisition import acquire_mux_scan
from zcu_lab.v2._support.mux_persistence import mux_sweep_spec
from zcu_lab.v2._support.mux_sweep_cfg import (
    MuxDriveModuleCfg,
    MuxToneCfg,
    participant_indices,
    validate_mux_scan,
)
from zcu_lab.v2.twotone.freq.core import FreqAnalyzeOptions

from .analysis import MuxFreqFit, analyze_frequency


class MuxFreqCfg(ProgramV2Cfg, ExpCfgModel):
    """Complete named tones, ordered participants and shared acquisition controls.

    modules holds optional reset and complete mux readout. tones contains one
    MuxToneCfg per original readout position, with unique nonempty names.
    participants is a nonempty, duplicate-free list of those names; its order
    determines execution/data rows (Q3/Q1 maps to original indices 2/0).
    Every drive is required/validated, including nonparticipants, on independent
    generators distinct from readout. Independent finite MHz sweeps have equal point counts across all tones.
    Inherited reps/rounds are positive; initial_delay/relax_delay are finite
    nonnegative us. No capture, bindings or writeback destinations are stored.
    """

    modules: MuxDriveModuleCfg
    tones: list[MuxToneCfg]
    participants: list[str]

    @model_validator(mode="after")
    def validate_scan(self) -> MuxFreqCfg:
        """Reject invalid full length, drive channels and zip counts at admission."""
        validate_mux_scan(self, self.modules, self.tones, self.participants)
        return self


def prepare_frequency_scan(cfg: MuxFreqCfg, *, soccfg: QickConfig) -> MuxPulseScan:
    """Lower full tones once and prepare fixed per-point pulses without acquisition.

    cfg is validated again from a copy, never mutated. soccfg is the target
    board's public QickConfig; coordinates use each active drive's own MHz
    grid and match program-native conversion. Returned point_pulses keeps nominal
    values so the program remains the sole native-conversion owner. Equal point
    counts are zipped by index, never expanded to a product. Invalid cfg raises
    ValueError; hardware conversion errors propagate. No devices, GUI, files or
    bindings are accessed. Program construction is left to acquire_mux_scan.
    """
    cfg = MuxFreqCfg.model_validate(cfg.model_dump())
    indices = participant_indices([tone.name for tone in cfg.tones], cfg.participants)
    coordinates = []
    points = [[] for _ in range(cfg.tones[indices[0]].sweep.expts)]
    for index in indices:
        tone = cfg.tones[index]
        drive = tone.drive
        nominal = sweep2array(tone.sweep, "none")
        actual = round_zcu_freq(nominal, soccfg, drive.ch, mixer_freq=drive.mixer_freq)
        coordinates.append(actual)
        for point, value in enumerate(nominal):
            points[point].append(drive.with_updates(freq=float(value)))
    return MuxPulseScan(indices, np.asarray(coordinates, dtype=np.float64), points)


class MuxFreqExp(PersistableExperiment[MuxSweepResult, MuxFreqCfg]):
    """Stateless zipped mux spectroscopy with per-original-index results.

    Run uses a common-start Join and host zip scan. Analysis and native
    persistence retain independent quantized coordinates, complex IQ and full cfg.
    Core neither registers a GUI nor produces binding-dependent proposals.
    """

    AXES_SPEC = mux_sweep_spec(
        MuxFreqCfg,
        tag="mux/freq",
        coordinate_label="Drive frequency",
        unit="Hz",
        scale=1e6,
    )

    Options: ClassVar[type[FreqAnalyzeOptions]] = FreqAnalyzeOptions

    def run(self, config: MuxFreqCfg, *, context: RunContext) -> MuxSweepResult:
        """Acquire independent zipped drives with common-start mux readout.

        config is copied and validated without mutation. Coordinates and complex
        average IQ are (K,P), in active original tone order. Frequencies are MHz.
        context owns compatible hardware, plots and a fresh cancel signal.
        Publish measurement_i for each original index; caller finishes/releases
        plots. Cancellation preserves completed points and the current point's
        completed-round average, leaving unwritten IQ NaN. Acquisition errors
        remain on context.cancel_signal: call raise_if_error() before success.
        Validation/build errors propagate. No saving or GUI access occurs.
        """
        cfg = MuxFreqCfg.model_validate(config.model_dump())
        scan = prepare_frequency_scan(cfg, soccfg=context.soccfg)
        return acquire_mux_scan(
            cfg,
            cfg.modules,
            scan,
            context=context,
            coordinate_label="Drive frequency (MHz)",
        )

    def analyze(
        self,
        source: RunRecord[MuxFreqCfg, MuxSweepResult],
        options: FreqAnalyzeOptions,
        *,
        plots: Plots,
    ) -> tuple[MuxFreqFit | MuxFitFailure, ...]:
        """Fit explicit source data, allowing cfg=None; publish distinct fit_i plots.

        Coordinates and IQ alone determine analysis. Numeric failures retain each
        original index and reason. No current form, device or binding is read.
        """
        return analyze_frequency(source.result, options, plots=plots)
