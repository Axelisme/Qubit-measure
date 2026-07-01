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
    IDENTITY,
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
class AccPhaseResult:
    lengths: NDArray[np.float64]
    phases: NDArray[np.float64]
    signals: NDArray[np.complex128]
    cfg_snapshot: AccPhaseCfg | None = None


def acc_phase_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return rotate2real(signals).real


class AccPhaseModuleCfg(ConfigBase):
    reset: ResetCfg | None = None
    flux_pulse: PulseCfg
    pi2_pulse: PulseCfg
    readout: ReadoutCfg


class AccPhaseSweepCfg(ConfigBase):
    length: SweepCfg
    phase: SweepCfg


class AccPhaseCfg(ProgramV2Cfg, ExpCfgModel):
    modules: AccPhaseModuleCfg
    sweep: AccPhaseSweepCfg
    readout_t: float


class AccPhaseExp(PersistableExperiment[AccPhaseResult, AccPhaseCfg]):
    # inner phases stored as-is (deg) -> IDENTITY; outer lengths mem us, disk s -> US_TO_S
    AXES_SPEC = AxesSpec(
        axes=(
            Axis("phases", "Phase", "deg", scale=IDENTITY),
            Axis("lengths", "Wait Time", "s", scale=US_TO_S),
        ),
        z=ZSpec("signals", "Signal", "a.u."),
        result_type=AccPhaseResult,
        cfg_type=AccPhaseCfg,
        tag="fastflux/distortion/acc_phase",
    )

    @record_result
    def run(
        self,
        soc,
        soccfg,
        cfg: AccPhaseCfg,
        *,
        acquire_kwargs: dict[str, Any] | None = None,
    ) -> AccPhaseResult:
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
                    lengths, phases, acc_phase_signal2real(data)
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
                            ],
                            SoftDelay("readout_t", sched.cfg.readout_t),
                        ),
                        Pulse(
                            "pi2_pulse2",
                            modules.pi2_pulse.with_updates(phase=phase_param),
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

        return AccPhaseResult(lengths, phases, signals, cfg_snapshot=orig_cfg)

    @retrieve_result
    def analyze(
        self,
        result: AccPhaseResult | None = None,
    ) -> Figure:
        assert result is not None, "No result found"

        cfg = result.cfg_snapshot
        if cfg is None:
            raise ValueError("cfg_snapshot is None")
        modules = cfg.modules

        flux_pulse = modules.flux_pulse
        pi2_len = float(modules.pi2_pulse.waveform.length)

        lengths, phases, signals2D = result.lengths, result.phases, result.signals

        # align to center of pi/2 pulse
        lengths = lengths + pi2_len / 2

        real_signals = acc_phase_signal2real(signals2D)

        thetas = np.deg2rad(phases)
        X = np.column_stack([np.ones_like(thetas), np.cos(thetas), np.sin(thetas)])
        X_inv = np.linalg.pinv(X)
        coeffs = X_inv @ real_signals.T
        init_phases = np.unwrap(np.arctan2(coeffs[2], coeffs[1]))
        detunes = -np.gradient(init_phases, lengths) / (2 * np.pi)

        sort_idxs = np.argsort(np.abs(detunes))
        mean_background = np.mean(detunes[sort_idxs[: int(len(detunes) * 0.1)]])
        mean_topdetune = np.mean(detunes[sort_idxs[int(len(detunes) * 0.9) :]])

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
        ax2.plot(lengths, detunes, ".-")
        ax2.set_ylabel("Detune (MHz)")
        ax2.set_xlabel("Wait Time (us)")

        plot_kwargs = dict(color="gray", alpha=0.3)
        ax2.axvspan(start_t - pi2_len / 2, start_t + pi2_len / 2, **plot_kwargs)
        ax2.axvspan(end_t - pi2_len / 2, end_t + pi2_len / 2, **plot_kwargs)
        ax1.axvline(start_t, color="black", linestyle="--")
        ax1.axvline(end_t, color="black", linestyle="--")

        ax2.legend()

        fig.tight_layout()

        return fig
