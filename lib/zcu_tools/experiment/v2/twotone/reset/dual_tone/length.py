from __future__ import annotations

import warnings
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from zcu_tools.cfg_model import ConfigBase
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
from zcu_tools.experiment.utils import setup_devices
from zcu_tools.experiment.v2.runtime import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils import sweep2array
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
    lengths: NDArray[np.float64]
    signals: NDArray[np.complex128]


def reset_length_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return rotate2real(signals).real


class LengthModuleCfg(ConfigBase):
    reset: ResetCfg | None = None
    init_pulse: PulseCfg | None = None
    tested_reset: TwoPulseResetCfg
    readout: ReadoutCfg


class LengthSweepCfg(ConfigBase):
    length: SweepCfg


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
        tag="twotone/reset/dual_tone/length",
    )

    def run(self, cfg: LengthCfg, *, context: RunContext) -> LengthResult:
        cfg = deepcopy(cfg)
        soc, soccfg = context.soc, context.soccfg
        setup_devices(
            cfg,
            context.devices,
            progress=True,
            cancel_signal=context.cancel_signal.event,
        )
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

        viewer = context.plots.liveplot_1d("measurement", "Length (us)", "Amplitude")
        signals_buffer = SignalBuffer(
            (len(lengths),),
            on_update=lambda data: viewer.update(
                lengths, reset_length_signal2real(data)
            ),
        )
        with Schedule(cfg, signals_buffer, stop=context.cancel_signal) as sched:
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
