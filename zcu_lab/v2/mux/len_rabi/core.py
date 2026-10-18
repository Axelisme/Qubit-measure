"""Zipped mux length Rabi acquisition with explicit-source native persistence."""

from __future__ import annotations

from copy import deepcopy
from typing import ClassVar

import numpy as np
from pydantic import model_validator
from qick import QickConfig
from zcu_tools.experiment import US_TO_S, PersistableExperiment
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import ProgramV2Cfg
from zcu_tools.program.v2.modules import ConstWaveformCfg, FlatTopWaveformCfg

from zcu_lab.v2._support.mux import MuxFitFailure, MuxPulseScan, MuxSweepResult
from zcu_lab.v2._support.mux_acquisition import acquire_mux_scan
from zcu_lab.v2._support.mux_persistence import mux_sweep_spec
from zcu_lab.v2._support.mux_sweep_cfg import (
    MuxDriveModuleCfg,
    MuxToneCfg,
    participant_indices,
    validate_mux_scan,
)
from zcu_lab.v2.twotone.rabi.len_rabi.core import LenRabiAnalyzeOptions

from .analysis import MuxLenRabiFit, analyze_length


class MuxLenRabiCfg(ProgramV2Cfg, ExpCfgModel):
    """Complete named tones, ordered participants and shared acquisition controls.

    modules holds optional reset and complete mux readout. tones contains one
    MuxToneCfg per original readout position, with unique nonempty names.
    participants is a nonempty, duplicate-free list of those names; its order
    determines execution/data rows (Q3/Q1 maps to original indices 2/0).
    Every drive is required/validated, including nonparticipants, on independent
    generators distinct from readout. Independent positive us length sweeps have equal point counts. Every waveform
    must be const or flat_top; arb duration belongs to its asset, not this sweep.
    Inherited reps/rounds are positive; initial_delay/relax_delay are finite
    nonnegative us. No capture, bindings or writeback destinations are stored.
    """

    modules: MuxDriveModuleCfg
    tones: list[MuxToneCfg]
    participants: list[str]

    @model_validator(mode="after")
    def validate_scan(self) -> MuxLenRabiCfg:
        """Reject invalid full tones, active generators, counts or length policy."""
        validate_mux_scan(self, self.modules, self.tones, self.participants)
        for index, tone in enumerate(self.tones):
            drive = tone.drive
            if not isinstance(drive.waveform, (ConstWaveformCfg, FlatTopWaveformCfg)):
                raise ValueError(
                    f"tones[{index}].drive.waveform: length sweep requires const or "
                    "flat_top; arb length is asset-derived with no public length sweep"
                )
            if min(tone.sweep.start, tone.sweep.stop) <= 0:
                raise ValueError(f"tones[{index}].sweep lengths must be positive us")
        return self


def prepare_length_scan(cfg: MuxLenRabiCfg, *, soccfg: QickConfig) -> MuxPulseScan:
    """Copy cfg and lower active tones once into independent zipped lengths.

    soccfg is the target board's public QickConfig. Return original indices,
    float64 coordinates (K,P) in us and copied point_pulses (P,K). Each row uses
    sweep2array's time grid for its drive generator. Pulse.set_param receives
    these quantized requested lengths, including ramps for flat_top, not plateau
    lengths. Waveform owners handle native conversion. Coordinates are not a
    compiled hardware trace; total playback can differ by native clock ticks.
    No product, padding, sorting or point removal occurs.

    Reject invalid cfg, a nonpositive quantized length or collapsed sweep before
    acquisition. Hardware conversion and waveform errors propagate. Inputs stay
    unchanged. This function acquires no devices, files or plotting resources.
    """
    cfg = MuxLenRabiCfg.model_validate(cfg.model_dump())
    indices = participant_indices([tone.name for tone in cfg.tones], cfg.participants)
    coordinates = []
    points = [[] for _ in range(cfg.tones[indices[0]].sweep.expts)]
    for index in indices:
        tone = cfg.tones[index]
        drive = tone.drive
        lengths = sweep2array(
            tone.sweep, "time", {"soccfg": soccfg, "gen_ch": drive.ch}
        )
        if np.any(lengths <= 0):
            raise ValueError(
                f"tones[{index}].sweep quantized lengths must be positive us"
            )
        coordinates.append(lengths)
        for point, length in enumerate(lengths):
            pulse = deepcopy(drive)
            pulse.set_param("length", float(length))
            points[point].append(pulse)
    return MuxPulseScan(indices, np.asarray(coordinates, dtype=np.float64), points)


class MuxLenRabiExp(PersistableExperiment[MuxSweepResult, MuxLenRabiCfg]):
    """Stateless mux length Rabi using the shared host zip acquisition.

    Result is MuxSweepResult with length coordinates in us. Analysis and native
    persistence retain original tone indices, independent axes and full cfg.
    No GUI registration, proposal or cross-run state is created.
    """

    AXES_SPEC = mux_sweep_spec(
        MuxLenRabiCfg,
        tag="mux/rabi/len",
        coordinate_label="Pulse length",
        unit="s",
        scale=US_TO_S,
    )

    Options: ClassVar[type[LenRabiAnalyzeOptions]] = LenRabiAnalyzeOptions

    def run(self, config: MuxLenRabiCfg, *, context: RunContext) -> MuxSweepResult:
        """Acquire zipped drive lengths with common-start Join and mux readout.

        config is copied and validated. Return original tone indices (K,),
        quantized requested total lengths (K,P) in us and complex128 average IQ
        (K,P), ADC-length-normalized a.u. Each point compiles once; readout
        follows every drive end, without sweep-wide padding or end alignment.
        Shared reset/reps/rounds/initial/relax delays follow the numeric cfg.

        context owns compatible soc/soccfg, caller-owned Plots and a fresh
        StopSignal. Publish measurement_i for each original index. Cancellation
        retains completed points and completed-round averages at the current
        point; unwritten points stay NaN. Acquisition errors retain their cause
        on context.cancel_signal: call raise_if_error() before reporting success.
        Validation/build errors propagate. Caller finishes/releases plots and
        owns hardware cleanup. Run does not save or read GUI/component state.
        """
        cfg = MuxLenRabiCfg.model_validate(config.model_dump())
        scan = prepare_length_scan(cfg, soccfg=context.soccfg)
        return acquire_mux_scan(
            cfg,
            cfg.modules,
            scan,
            context=context,
            coordinate_label="Pulse length (us)",
        )

    def analyze(
        self,
        source: RunRecord[MuxLenRabiCfg, MuxSweepResult],
        options: LenRabiAnalyzeOptions,
        *,
        plots: Plots,
    ) -> tuple[MuxLenRabiFit | MuxFitFailure, ...]:
        """Fit explicit source, accepting cfg=None, with single-qubit Len options.

        options.decay selects the damped cosine, default True. fit_phase defaults
        False, fixing phase to zero with signed amplitude; True fits it freely.
        Each original tone returns pi/pi2 lengths and errors in us, Rabi frequency
        and error in MHz, and fit quality, or a numerical failure with its reason.
        plots owns fit_i; use fresh Plots for each analysis and finish/release it.
        Invalid source contracts raise instead of becoming fit failures. No live cfg or hardware is read.
        """
        return analyze_length(source.result, options, plots=plots)
