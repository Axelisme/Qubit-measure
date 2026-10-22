from __future__ import annotations

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
    PulseReset,
    Readout,
    ReadoutCfg,
    Reset,
    ResetCfg,
    SweepCfg,
    sweep2param,
)
from zcu_tools.program.v2.modules import PulseResetCfg
from zcu_tools.utils.process import rotate2real


@dataclass(frozen=True)
class LengthResult:
    """Reset pulse lengths in us and mean complex IQ (length,)."""

    lengths: NDArray[np.float64]
    signals: NDArray[np.complex128]


def reset_length_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return rotate2real(signals).real


class LengthModuleCfg(ConfigBase):
    reset: Annotated[ResetCfg | None, FieldMeta(label="Reset")] = None
    init_pulse: Annotated[PulseCfg | None, FieldMeta(label="Init pulse")] = None
    tested_reset: Annotated[PulseResetCfg, FieldMeta(label="Tested reset")]
    readout: Annotated[ReadoutCfg, FieldMeta(label="Readout")]


class LengthSweepCfg(ConfigBase):
    length: Annotated[SweepCfg, FieldMeta(label="Length", unit="us")]


class LengthCfg(ProgramV2Cfg, ExpCfgModel):
    modules: LengthModuleCfg
    sweep: LengthSweepCfg


class LengthExp(PersistableExperiment[LengthResult, LengthCfg]):
    # length stores us in-memory, seconds on disk -> scale=US_TO_S (1e-6)
    AXES_SPEC = AxesSpec(
        axes=(Axis("lengths", "Length", "s", scale=US_TO_S),),
        z=ZSpec("signals", "Signal", "a.u."),
        result_type=LengthResult,
        cfg_type=LengthCfg,
        cfg_schema_version="2.0",
        tag="twotone/reset/single_tone/length",
    )

    def run(self, cfg: LengthCfg, *, context: RunContext) -> LengthResult:
        """Measure the single-tone reset response versus pulse duration.

        Apply optional reset and initialization, the length-swept tested
        reset pulse, then read out. Return reset lengths in us and mean
        complex IQ shaped (Nlength,). The curve shows duration dependence,
        not a calibrated reset fidelity; analysis only plots the response.
        """
        # Quantize the tested reset duration on its generator channel.
        cfg = deepcopy(cfg)
        soc, soccfg = context.soc, context.soccfg
        modules = cfg.modules

        length_sweep = cfg.sweep.length

        pulse_cfg = modules.tested_reset.pulse_cfg

        lengths = sweep2array(
            length_sweep, "time", dict(soccfg=soccfg, gen_ch=pulse_cfg.ch)
        )

        # Retain complex IQ while displaying the duration-dependent projection.
        viewer = context.plots.liveplot_1d("measurement", "Length (us)", "Amplitude")
        signals_buffer = SignalBuffer(
            (len(lengths),),
            on_update=lambda data: viewer.update(
                lengths, reset_length_signal2real(data)
            ),
        )

        # Sweep only the tested reset length after the optional preparation.
        with Schedule(cfg, signals_buffer, scope=context.scope) as sched:
            modules = sched.cfg.modules
            modules.tested_reset.set_param(
                "length", sweep2param("length", sched.cfg.sweep.length)
            )
            _ = (
                sched.prog_builder(soc, soccfg)
                .add(
                    Reset("reset", modules.reset),
                    Pulse("init_pulse", modules.init_pulse),
                    PulseReset("tested_reset", modules.tested_reset),
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
        """Plot the projected reset IQ response versus pulse duration in us.

        Omit unmeasured NaN points and publish fit without fitting or returning
        an optimal length. source.cfg may be None.
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
