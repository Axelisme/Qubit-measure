from __future__ import annotations

import warnings
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment import (
    US_TO_S,
    AxesSpec,
    Axis,
    PersistableExperiment,
    ZSpec,
    config,
    record_result,
    retrieve_result,
)
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import setup_devices
from zcu_tools.experiment.v2.runner import MeasureSession, TaskState
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.liveplot import LivePlot1D
from zcu_tools.program.v2 import (
    ModularProgramV2,
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
    cfg_snapshot: LengthCfg | None = None


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

    @record_result
    def run(
        self,
        soc,
        soccfg,
        cfg: LengthCfg,
        *,
        acquire_kwargs: dict[str, Any] | None = None,
    ) -> LengthResult:
        orig_cfg = deepcopy(cfg)
        setup_devices(cfg, progress=True)
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
                "Sweep lengths for pulse1 and pulse2 are different. This may lead to unexpected results."
            )
        if np.any(pulse2_lengths + length_diff < 0):
            raise ValueError(
                "Find negative length in pulse2 while sweeping pulse1 length. Please check the sweep configuration."
            )
        lengths = pulse1_lengths  # Use pulse1 lengths as the x-axis values

        def measure_fn(
            ctx: TaskState[NDArray[np.complex128], Any, LengthCfg],
            update_hook: Callable | None,
        ) -> list[NDArray[np.float64]]:
            cfg = ctx.cfg
            modules = cfg.modules

            tested_reset_cfg = modules.tested_reset
            pulse1_cfg = tested_reset_cfg.pulse1_cfg
            pulse2_cfg = tested_reset_cfg.pulse2_cfg

            length_diff = pulse2_cfg.waveform.length - pulse1_cfg.waveform.length
            length1_param = sweep2param("length", length_sweep)

            pulse1_cfg.set_param("length", length1_param)
            pulse2_cfg.set_param("length", length1_param + length_diff)

            return ModularProgramV2(
                soccfg,
                cfg,
                sweep=[("length", length_sweep)],
                modules=[
                    Reset("reset", modules.reset),
                    Pulse("init_pulse", modules.init_pulse),
                    TwoPulseReset("tested_reset", tested_reset_cfg),
                    Readout("readout", modules.readout),
                ],
            ).acquire(
                soc,
                progress=False,
                round_hook=update_hook,
                stop_checkers=[ctx.is_stop],
                **(acquire_kwargs or {}),
            )

        with LivePlot1D("Length (us)", "Amplitude") as viewer:
            with MeasureSession(cfg) as run:
                signals_buffer = run.buffer(
                    (len(lengths),),
                    dtype=np.complex128,
                    on_update=lambda data: viewer.update(
                        lengths, reset_length_signal2real(data)
                    ),
                )
                signals_buffer.measure(measure_fn, pbar_n=run.cfg.rounds)
                signals = signals_buffer.array

        return LengthResult(lengths, signals, cfg_snapshot=orig_cfg)

    @retrieve_result
    def analyze(self, result: LengthResult | None = None) -> Figure:
        assert result is not None, "no result found"

        lens, signals = result.lengths, result.signals

        # Discard NaNs (possible early abort)
        val_mask = ~np.isnan(signals)
        lens = lens[val_mask]
        signals = signals[val_mask]

        real_signals = reset_length_signal2real(signals)

        fig, ax = plt.subplots(figsize=config.figsize)
        assert isinstance(fig, Figure)

        ax.plot(lens, real_signals, marker=".")
        ax.set_xlabel("ProbeTime (us)", fontsize=14)
        ax.set_ylabel("Signal (a.u.)", fontsize=14)
        ax.grid(True)
        ax.tick_params(axis="both", which="major", labelsize=12)
        plt.show()

        fig.tight_layout()

        return fig
