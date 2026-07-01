from __future__ import annotations

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
    record_result,
    retrieve_result,
)
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import setup_devices
from zcu_tools.experiment.v2.runner import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.liveplot import LivePlot2D
from zcu_tools.program.v2 import (
    Join,
    ProgramV2Cfg,
    Pulse,
    PulseCfg,
    Readout,
    ReadoutCfg,
    Reset,
    ResetCfg,
    SoftDelay,
    SweepCfg,
    sweep2param,
)
from zcu_tools.utils.process import rotate2real


@dataclass(frozen=True)
class PhaseResult:
    lengths: NDArray[np.float64]
    phases: NDArray[np.float64]
    signals: NDArray[np.complex128]
    cfg_snapshot: PhaseCfg | None = None


def phase_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return rotate2real(signals).real


class PhaseModuleCfg(ConfigBase):
    reset: ResetCfg | None = None
    flux_pulse: PulseCfg
    pi2_pulse: PulseCfg
    readout: ReadoutCfg


class PhaseSweepCfg(ConfigBase):
    length: SweepCfg
    phase: SweepCfg


class PhaseCfg(ProgramV2Cfg, ExpCfgModel):
    modules: PhaseModuleCfg
    readout_t: float
    sweep: PhaseSweepCfg


class PhaseExp(PersistableExperiment[PhaseResult, PhaseCfg]):
    # inner phases raw deg (IDENTITY); outer lengths memory-us -> disk-s via US_TO_S
    AXES_SPEC = AxesSpec(
        axes=(
            Axis("phases", "Phase", "deg"),
            Axis("lengths", "Wait Time", "s", scale=US_TO_S),
        ),
        z=ZSpec("signals", "Signal", "a.u."),
        result_type=PhaseResult,
        cfg_type=PhaseCfg,
        tag="fastflux/distortion/phase",
    )

    @record_result
    def run(
        self,
        soc,
        soccfg,
        cfg: PhaseCfg,
        *,
        acquire_kwargs: dict[str, Any] | None = None,
    ) -> PhaseResult:
        orig_cfg = deepcopy(cfg)
        setup_devices(cfg, progress=True)
        modules = cfg.modules

        length_sweep = cfg.sweep.length
        phase_sweep = cfg.sweep.phase

        pi2_pulse = modules.pi2_pulse

        lengths = sweep2array(length_sweep, "time", {"soccfg": soccfg})
        phases = sweep2array(
            phase_sweep, "phase", {"soccfg": soccfg, "gen_ch": pi2_pulse.ch}
        )

        with LivePlot2D("Time (us)", "Phase (deg)") as viewer:
            signals_buffer = SignalBuffer(
                (len(lengths), len(phases)),
                on_update=lambda data: viewer.update(
                    lengths, phases, phase_signal2real(data)
                ),
            )
            with Schedule(cfg, signals_buffer) as sched:
                modules = sched.cfg.modules
                length_param = sweep2param("length", sched.cfg.sweep.length)
                phase_param = sweep2param("phase", sched.cfg.sweep.phase)
                _ = (
                    sched.prog_builder(soc, soccfg)
                    .add(
                        Reset("reset", modules.reset),
                        Join(
                            Pulse("flux_pulse", modules.flux_pulse),
                            [
                                SoftDelay("wait_time", delay=length_param),
                                Pulse("pi2_pulse1", modules.pi2_pulse),
                                Pulse(
                                    name="pi2_pulse2",
                                    cfg=modules.pi2_pulse.with_updates(
                                        phase=phase_param
                                    ),
                                ),
                            ],
                            SoftDelay("readout_t", sched.cfg.readout_t),
                        ),
                        Readout("readout", modules.readout),
                    )
                    .declare_sweep("length", sched.cfg.sweep.length)
                    .declare_sweep("phase", sched.cfg.sweep.phase)
                    .build_and_acquire(
                        **(acquire_kwargs or {}),
                    )
                )
                signals = signals_buffer.array

        return PhaseResult(lengths, phases, signals, cfg_snapshot=orig_cfg)

    @retrieve_result
    def analyze(
        self,
        result: PhaseResult | None = None,
    ) -> Figure:
        assert result is not None, "No result found"

        cfg = result.cfg_snapshot
        if cfg is None:
            raise ValueError("cfg_snapshot is None")
        modules = cfg.modules

        flux_pulse = modules.flux_pulse
        pi2_len = float(modules.pi2_pulse.waveform.length)

        lengths, phases, signals2D = result.lengths, result.phases, result.signals

        # align to middle of two pi/2 pulses
        lengths = lengths + pi2_len

        real_signals = phase_signal2real(signals2D)

        thetas = np.deg2rad(phases)
        X = np.column_stack([np.ones_like(thetas), np.cos(thetas), np.sin(thetas)])
        coeffs = np.linalg.pinv(X) @ real_signals.T
        init_phases = np.rad2deg(np.unwrap(np.arctan2(coeffs[2], coeffs[1])))

        sort_idxs = np.argsort(np.abs(init_phases))
        mean_background = np.median(
            init_phases[sort_idxs[: int(len(init_phases) * 0.2)]]
        )
        mean_topdetune = np.median(
            init_phases[sort_idxs[int(len(init_phases) * 0.8) :]]
        )

        start_t = float(flux_pulse.pre_delay)
        end_t = start_t + float(flux_pulse.waveform.length)

        ideal_lengths = np.linspace(lengths[0], lengths[-1], 1000)
        ideal_curve = np.full_like(ideal_lengths, mean_background)
        ideal_curve[(ideal_lengths >= start_t) & (ideal_lengths <= end_t)] = (
            mean_topdetune
        )

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4), sharex=True)
        ax1.imshow(
            real_signals.T,
            extent=(lengths[0], lengths[-1], phases[0], phases[-1]),
            aspect="auto",
            interpolation="none",
            cmap="RdBu_r",
        )
        ax1.set_ylabel("Phase (deg)")

        ax2.plot(ideal_lengths, ideal_curve, "g-", label="Ideal")
        ax2.plot(lengths, init_phases, ".-")
        ax2.set_ylabel("Phase (deg)")
        ax2.set_xlabel("Wait Time (us)")

        plot_kwargs = dict(color="gray", alpha=0.3)
        ax2.axvspan(start_t - pi2_len, start_t + pi2_len, **plot_kwargs)
        ax1.axvline(start_t, color="black", linestyle="--")
        ax2.axvspan(end_t - pi2_len, end_t + pi2_len, **plot_kwargs)
        ax1.axvline(end_t, color="black", linestyle="--")

        ax2.legend()

        fig.tight_layout()

        return fig
