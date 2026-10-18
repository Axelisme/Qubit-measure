"""Numeric cfg, analysis and persistence for zipped mux amplitude Rabi."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from pydantic import model_validator
from qick import QickConfig
from zcu_tools.experiment import PersistableExperiment
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.utils.round_zcu import round_zcu_gain, sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import ProgramV2Cfg

from zcu_lab.v2._support.mux import MuxFitFailure, MuxPulseScan, MuxSweepResult
from zcu_lab.v2._support.mux_persistence import mux_sweep_spec
from zcu_lab.v2._support.mux_sweep_cfg import (
    MuxDriveModuleCfg,
    MuxToneCfg,
    validate_mux_scan,
)
from zcu_lab.v2.twotone.rabi.amp_rabi.core import AmpRabiAnalyzeOptions

from .analysis import MuxAmpRabiFit, analyze_amplitude


class MuxAmpRabiCfg(ProgramV2Cfg, ExpCfgModel):
    """Full ordered tones and shared reps/rounds/reset/readout/delays.

    modules holds optional reset and complete mux readout. tones has exactly one
    active/drive/sweep element per full tone; at least one active. Active drives
    use independent generators distinct from readout. Active sweeps are finite
    normalized gains in [-1,1], with equal point counts. Inactive drives may be
    None and their sweeps do not constrain the active zip count. Counts must be
    positive and inherited initial_delay/relax_delay finite nonnegative us.
    Numeric cfg contains no capture, component names or writeback destinations.
    """

    modules: MuxDriveModuleCfg
    tones: list[MuxToneCfg]

    @model_validator(mode="after")
    def validate_scan(self) -> MuxAmpRabiCfg:
        """Reject invalid full length, drive channels, counts or gain ranges."""
        validate_mux_scan(self, self.modules, self.tones)
        for index, tone in enumerate(self.tones):
            if tone.active and max(abs(tone.sweep.start), abs(tone.sweep.stop)) > 1:
                raise ValueError(f"tones[{index}].sweep gains must be in [-1,1]")
        return self


def prepare_amplitude_scan(cfg: MuxAmpRabiCfg, *, soccfg: QickConfig) -> MuxPulseScan:
    """Lower full tones once and prepare fixed per-point pulses without acquisition.

    cfg is validated again from a copy, never mutated. soccfg is the target
    board's public QickConfig; coordinates use each active drive's own normalized
    gain grid and match program-native conversion. Returned point_pulses keeps nominal
    values so the program remains the sole native-conversion owner. Equal point
    counts are zipped by index, never expanded to a product. Invalid cfg raises
    ValueError; hardware conversion errors propagate. No devices, GUI, files or
    bindings are accessed. End-aligned program construction is not done here.
    """
    cfg = MuxAmpRabiCfg.model_validate(cfg.model_dump())
    indices = [i for i, tone in enumerate(cfg.tones) if tone.active]
    coordinates = []
    points = [[] for _ in range(cfg.tones[indices[0]].sweep.expts)]
    for index in indices:
        tone = cfg.tones[index]
        drive = tone.drive
        if drive is None:
            raise ValueError(f"tones[{index}].drive is required")
        nominal = sweep2array(tone.sweep, "none")
        actual = round_zcu_gain(nominal, soccfg, drive.ch)
        coordinates.append(actual)
        for point, value in enumerate(nominal):
            points[point].append(drive.with_updates(gain=float(value)))
    return MuxPulseScan(indices, np.asarray(coordinates, dtype=np.float64), points)


class MuxAmpRabiExp(PersistableExperiment[MuxSweepResult, MuxAmpRabiCfg]):
    """Stateless existing undamped Rabi analysis and persistence, without acquisition.

    Persistence retains full cfg, original indices, independent actual gains and
    complex averaged IQ. This core does not register a GUI or produce proposals.
    Acquisition awaits the program owner's public end-alignment contract.
    """

    AXES_SPEC = mux_sweep_spec(
        MuxAmpRabiCfg,
        tag="mux/rabi/amp",
        coordinate_label="Pulse gain",
        unit="",
        scale=1.0,
    )

    Options: ClassVar[type[AmpRabiAnalyzeOptions]] = AmpRabiAnalyzeOptions

    def analyze(
        self,
        source: RunRecord[MuxAmpRabiCfg, MuxSweepResult],
        options: AmpRabiAnalyzeOptions,
        *,
        plots: Plots,
    ) -> tuple[MuxAmpRabiFit | MuxFitFailure, ...]:
        """Fit explicit source using each row's gains and existing Rabi policy.

        cfg=None is supported. Return pi/pi2 gain values/errors/quality or an
        indexed failure reason. Plots owns fit_i; no identity or live state read.
        """
        return analyze_amplitude(source.result, options, plots=plots)
