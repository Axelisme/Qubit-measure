from __future__ import annotations

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from typing_extensions import Any, Callable, Optional, TypeAlias

from zcu_tools.config import ConfigBase
from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import make_comment, parse_comment, setup_devices
from zcu_tools.experiment.v2.runner import Task, TaskState, run_task
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.liveplot import LivePlot2D
from zcu_tools.program import SweepCfg
from zcu_tools.program.v2 import (
    Join,
    ModularProgramV2,
    ProgramV2Cfg,
    Pulse,
    PulseCfg,
    Readout,
    ReadoutCfg,
    Reset,
    ResetCfg,
    SoftDelay,
    sweep2param,
)
from zcu_tools.utils.datasaver import load_data, save_data
from zcu_tools.utils.process import rotate2real

# (lengths, phases, signals2D)
AccPhaseResult: TypeAlias = tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.complex128]
]


def acc_phase_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return rotate2real(signals).real


class AccPhaseModuleCfg(ConfigBase):
    reset: Optional[ResetCfg] = None
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


class AccPhaseExp(AbsExperiment[AccPhaseResult, AccPhaseCfg]):
    def run(
        self,
        soc,
        soccfg,
        cfg: AccPhaseCfg,
        *,
        acquire_kwargs: Optional[dict[str, Any]] = None,
    ) -> AccPhaseResult:
        setup_devices(cfg, progress=True)
        modules = cfg.modules

        length_sweep = cfg.sweep.length
        phase_sweep = cfg.sweep.phase

        pi2_pulse = modules.pi2_pulse

        lengths = sweep2array(length_sweep, "time", {"soccfg": soccfg})
        phases = sweep2array(
            phase_sweep, "phase", {"soccfg": soccfg, "gen_ch": pi2_pulse.ch}
        )

        def measure_fn(
            ctx: TaskState[NDArray[np.complex128], Any, AccPhaseCfg],
            update_hook: Optional[Callable],
        ) -> list[NDArray[np.float64]]:
            modules = ctx.cfg.modules

            length_sweep = ctx.cfg.sweep.length
            phase_sweep = ctx.cfg.sweep.phase
            length_param = sweep2param("length", length_sweep)
            phase_param = sweep2param("phase", phase_sweep)

            return ModularProgramV2(
                soccfg,
                cfg,
                modules=[
                    Reset("reset", modules.reset),
                    Join(
                        Pulse("flux_pulse", modules.flux_pulse),
                        [
                            SoftDelay("wait_time", delay=length_param),
                            Pulse("pi2_pulse1", modules.pi2_pulse, tag="pi2_pulse1"),
                        ],
                        SoftDelay("readout_t", ctx.cfg.readout_t),
                    ),
                    Pulse(
                        name="pi2_pulse2",
                        cfg=modules.pi2_pulse.with_updates(phase=phase_param),
                    ),
                    Readout("readout", modules.readout),
                ],
                sweep=[
                    ("length", length_sweep),
                    ("phase", phase_sweep),
                ],
            ).acquire(
                soc, progress=False, round_hook=update_hook, **(acquire_kwargs or {})
            )

        with LivePlot2D("Time (us)", "Phase (deg)") as viewer:
            signals = run_task(
                task=Task(
                    measure_fn=measure_fn,
                    result_shape=(len(lengths), len(phases)),
                    pbar_n=cfg.rounds,
                ),
                init_cfg=cfg,
                on_update=lambda ctx: viewer.update(
                    lengths, phases, acc_phase_signal2real(ctx.root_data)
                ),
            )

        # Cache results
        self.last_cfg = deepcopy(cfg)
        self.last_result = (lengths, phases, signals)

        return lengths, phases, signals

    def analyze(
        self,
        cfg: Optional[AccPhaseCfg] = None,
        result: Optional[AccPhaseResult] = None,
    ) -> Figure:
        if cfg is None:
            cfg = self.last_cfg
        assert cfg is not None, "No config found"
        modules = cfg.modules

        flux_pulse = modules.flux_pulse
        pi2_len = float(modules.pi2_pulse.waveform.length)

        if result is None:
            result = self.last_result
        assert result is not None, "No result found"

        lengths, phases, signals2D = result

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

    def save(
        self,
        filepath: str,
        result: Optional[AccPhaseResult] = None,
        comment: Optional[str] = None,
        tag: str = "fastflux/distortion/acc_phase",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "No result found"

        lengths, phases, signals2D = result

        cfg = self.last_cfg
        assert cfg is not None
        comment = make_comment(cfg, comment)

        save_data(
            filepath=filepath,
            x_info={"name": "Wait Time", "unit": "s", "values": lengths * 1e-6},
            y_info={"name": "Phase", "unit": "deg", "values": phases},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals2D.T},
            comment=comment,
            tag=tag,
            **kwargs,
        )

    def load(self, filepath: str, **kwargs) -> AccPhaseResult:
        signals2D, lengths, phases, comment = load_data(filepath, return_comment=True, **kwargs)
        assert phases is not None and lengths is not None
        assert len(phases.shape) == 1 and len(lengths.shape) == 1
        assert signals2D.shape == (len(phases), len(lengths))

        lengths = lengths * 1e6  # s -> us
        signals2D = signals2D.T  # transpose back

        phases = phases.astype(np.float64)
        lengths = lengths.astype(np.float64)
        signals2D = signals2D.T.astype(np.complex128)

        if comment is not None:
            cfg, _, _ = parse_comment(comment)
            if cfg is not None:
                self.last_cfg = AccPhaseCfg.validate_or_warn(cfg, source=filepath)
        self.last_result = (lengths, phases, signals2D)

        return lengths, phases, signals2D
