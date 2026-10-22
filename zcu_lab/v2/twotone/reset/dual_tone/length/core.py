from __future__ import annotations

import warnings
from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated

import numpy as np
from numpy.typing import NDArray
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import (
    US_TO_S,
    AxesSpec,
    Axis,
    PersistableExperiment,
    ZSpec,
    config,
)
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    ProgramV2Cfg,
    Pulse,
    PulseCfg,
    Readout,
    ReadoutCfg,
    Reset,
    ResetCfg,
    SweepCfg,
    TwoPulseReset,
    sweep2param,
)
from zcu_tools.program.v2.modules import TwoPulseResetCfg
from zcu_tools.utils.process import rotate2real


@dataclass(frozen=True)
class LengthResult:
    """Pulse1 lengths in us and mean complex IQ (length,)."""

    lengths: NDArray[np.float64]
    signals: NDArray[np.complex128]


def reset_length_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return rotate2real(signals).real


class LengthModuleCfg(ConfigBase):
    reset: Annotated[ResetCfg | None, FieldMeta(label="Reset")] = None
    init_pulse: Annotated[PulseCfg | None, FieldMeta(label="Init pulse")] = None
    tested_reset: Annotated[TwoPulseResetCfg, FieldMeta(label="Tested reset")]
    readout: Annotated[ReadoutCfg, FieldMeta(label="Readout")]


class LengthSweepCfg(ConfigBase):
    length: Annotated[SweepCfg, FieldMeta(label="Length", unit="us")]


class LengthCfg(ProgramV2Cfg, ExpCfgModel):
    modules: LengthModuleCfg
    sweep: LengthSweepCfg


class LengthExp(PersistableExperiment[LengthResult, LengthCfg]):
    # Length stores seconds on disk; Result holds us -> scale=US_TO_S
    AXES_SPEC = AxesSpec(
        axes=(Axis("lengths", "Length", "s", US_TO_S),),
        z=ZSpec("signals", "Signal", "a.u."),
        result_type=LengthResult,
        cfg_type=LengthCfg,
        cfg_schema_version="2.0",
        tag="twotone/reset/dual_tone/length",
    )

    def run(self, cfg: LengthCfg, *, context: RunContext) -> LengthResult:
        """Measure the dual-tone reset response versus the first pulse length.

        Apply optional reset and initialization, start both tested reset
        pulses together, then read out after the longer pulse finishes.
        Sweep pulse1 length while preserving pulse2-minus-pulse1 length.
        Return pulse1 lengths in us and mean complex IQ shaped (Nlength,).
        Warn if channel time grids differ; raise ValueError if the derived
        pulse2 length is negative. The IQ response is not reset fidelity.
        """
        # Check both channel grids and keep the configured pulse-duration difference.
        cfg = deepcopy(cfg)
        soc, soccfg = context.soc, context.soccfg
        modules = cfg.modules

        length_sweep = cfg.sweep.length

        reset_cfg = modules.tested_reset
        pulse1_cfg = reset_cfg.pulse1_cfg
        pulse2_cfg = reset_cfg.pulse2_cfg
        length_diff = pulse2_cfg.waveform.length - pulse1_cfg.waveform.length

        pulse1_lengths = sweep2array(
            length_sweep, "time", dict(soccfg=soccfg, gen_ch=pulse1_cfg.ch)
        )
        pulse2_lengths = sweep2array(
            length_sweep, "time", dict(soccfg=soccfg, gen_ch=pulse2_cfg.ch)
        )
        if not np.allclose(pulse1_lengths, pulse2_lengths, atol=1e-2):
            warnings.warn(
                "Sweep lengths for pulse1 and pulse2 are different. This may lead to unexpected results.",
                stacklevel=2,
            )
        if np.any(pulse2_lengths + length_diff < 0):
            raise ValueError(
                "Find negative length in pulse2 while sweeping pulse1 length. Please check the sweep configuration."
            )
        lengths = pulse1_lengths  # Use pulse1 lengths as the x-axis values

        # Retain complex IQ while displaying its common-axis projection.
        viewer = context.plots.liveplot_1d("measurement", "Length (us)", "Amplitude")
        signals_buffer = SignalBuffer(
            (len(lengths),),
            on_update=lambda data: viewer.update(
                lengths, reset_length_signal2real(data)
            ),
        )

        # Sweep both durations together before reading out the reset response.
        with Schedule(cfg, signals_buffer, scope=context.scope) as sched:
            modules = sched.cfg.modules
            tested_reset_cfg = modules.tested_reset
            pulse1_cfg = tested_reset_cfg.pulse1_cfg
            pulse2_cfg = tested_reset_cfg.pulse2_cfg

            length_diff = pulse2_cfg.waveform.length - pulse1_cfg.waveform.length
            length1_param = sweep2param("length", sched.cfg.sweep.length)
            pulse1_cfg.set_param("length", length1_param)
            pulse2_cfg.set_param("length", length1_param + length_diff)

            _ = (
                sched.prog_builder(soc, soccfg)
                .add(
                    Reset("reset", modules.reset),
                    Pulse("init_pulse", modules.init_pulse),
                    TwoPulseReset("tested_reset", tested_reset_cfg),
                    Readout("readout", modules.readout),
                )
                .declare_sweep("length", sched.cfg.sweep.length)
                .build_and_acquire()
            )

        return LengthResult(lengths, signals_buffer.array)

    def analyze(
        self,
        source: RunRecord[LengthCfg, LengthResult],
        options: None,
        *,
        plots: Plots,
    ) -> None:
        """Plot the projected IQ response versus pulse1 length in us.

        Omit unmeasured NaN points and publish fit without fitting or returning
        an optimal duration. source.cfg may be None.
        """
        del options
        result = source.result

        lens, signals = result.lengths, result.signals

        # Discard NaNs (possible early abort)
        val_mask = ~np.isnan(signals)
        lens = lens[val_mask]
        signals = signals[val_mask]

        real_signals = reset_length_signal2real(signals)

        fig, ax = plots.subplots("fit", figsize=config.figsize)

        ax.plot(lens, real_signals, marker=".")
        ax.set_xlabel("ProbeTime (us)", fontsize=14)
        ax.set_ylabel("Signal (a.u.)", fontsize=14)
        ax.grid(True)
        ax.tick_params(axis="both", which="major", labelsize=12)

        fig.tight_layout()
