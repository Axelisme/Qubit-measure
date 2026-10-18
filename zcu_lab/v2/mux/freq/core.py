"""Numeric cfg, analysis and persistence for zipped mux qubit spectroscopy."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from pydantic import model_validator
from qick import QickConfig
from zcu_tools.experiment import PersistableExperiment
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.utils.round_zcu import round_zcu_freq, sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import ProgramV2Cfg

from zcu_lab.v2._support.mux import MuxFitFailure, MuxPulseScan, MuxSweepResult
from zcu_lab.v2._support.mux_persistence import mux_sweep_spec
from zcu_lab.v2._support.mux_sweep_cfg import (
    MuxDriveModuleCfg,
    MuxToneCfg,
    validate_mux_scan,
)
from zcu_lab.v2.twotone.freq.core import FreqAnalyzeOptions

from .analysis import MuxFreqFit, analyze_frequency


class MuxFreqCfg(ProgramV2Cfg, ExpCfgModel):
    """Full ordered tones and one shared acquisition/reset/readout configuration.

    modules contains optional reset and full mux readout. tones has exactly one
    active/drive/sweep element per full readout tone. At least one is active;
    only active drives are required, on independent generators distinct from
    readout. Active sweeps are finite MHz ranges with equal point counts.
    reps/rounds are positive shared counts; initial_delay/relax_delay are finite
    nonnegative us, with defaults inherited from ProgramV2Cfg. No identity,
    component capture or writeback destination is stored in numeric cfg.
    """

    modules: MuxDriveModuleCfg
    tones: list[MuxToneCfg]

    @model_validator(mode="after")
    def validate_scan(self) -> MuxFreqCfg:
        """Reject invalid full length, drive channels and zip counts at admission."""
        validate_mux_scan(self, self.modules, self.tones)
        return self


def prepare_frequency_scan(cfg: MuxFreqCfg, *, soccfg: QickConfig) -> MuxPulseScan:
    """Lower full tones once and prepare fixed per-point pulses without acquisition.

    cfg is validated again from a copy, never mutated. soccfg is the target
    board's public QickConfig; coordinates use each active drive's own MHz
    grid and match program-native conversion. Returned point_pulses keeps nominal
    values so the program remains the sole native-conversion owner. Equal point
    counts are zipped by index, never expanded to a product. Invalid cfg raises
    ValueError; hardware conversion errors propagate. No devices, GUI, files or
    bindings are accessed. End-aligned program construction is not done here.
    """
    cfg = MuxFreqCfg.model_validate(cfg.model_dump())
    indices = [i for i, tone in enumerate(cfg.tones) if tone.active]
    coordinates = []
    points = [[] for _ in range(cfg.tones[indices[0]].sweep.expts)]
    for index in indices:
        tone = cfg.tones[index]
        drive = tone.drive
        if drive is None:
            raise ValueError(f"tones[{index}].drive is required")
        nominal = sweep2array(tone.sweep, "none")
        actual = round_zcu_freq(nominal, soccfg, drive.ch, mixer_freq=drive.mixer_freq)
        coordinates.append(actual)
        for point, value in enumerate(nominal):
            points[point].append(drive.with_updates(freq=float(value)))
    return MuxPulseScan(indices, np.asarray(coordinates, dtype=np.float64), points)


class MuxFreqExp(PersistableExperiment[MuxSweepResult, MuxFreqCfg]):
    """Stateless per-original-index analysis and persistence, without acquisition.

    Native persistence retains full cfg, independent MHz coordinates and complex
    IQ. Coordinates export as Hz. No GUI or catalog registration is performed.
    Acquisition awaits the program owner's public end-alignment contract.
    """

    AXES_SPEC = mux_sweep_spec(
        MuxFreqCfg,
        tag="mux/freq",
        coordinate_label="Drive frequency",
        unit="Hz",
        scale=1e6,
    )

    Options: ClassVar[type[FreqAnalyzeOptions]] = FreqAnalyzeOptions

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
