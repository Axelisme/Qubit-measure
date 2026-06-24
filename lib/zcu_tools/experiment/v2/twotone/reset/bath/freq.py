from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from qick.asm_v2 import QickSweep1D

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment import (
    IDENTITY,
    MHZ_TO_HZ,
    AxesSpec,
    Axis,
    PersistableExperiment,
    ZSpec,
    record_result,
    retrieve_result,
)
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import setup_devices
from zcu_tools.experiment.v2.runner import Task, TaskState, run_task
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.liveplot import LivePlot2D
from zcu_tools.program.v2 import (
    BathReset,
    BathResetCfg,
    ModularProgramV2,
    ProgramV2Cfg,
    Pulse,
    PulseCfg,
    Readout,
    ReadoutCfg,
    Reset,
    ResetCfg,
    SweepCfg,
    sweep2param,
)
from zcu_tools.utils.process import SmoothMethod, smooth_signal_nd


def _default_phase_values() -> NDArray[np.float64]:
    return np.array([0.0, 90.0, 180.0, 270.0], dtype=np.float64)


@dataclass(frozen=True)
class FreqGainResult:
    gains: NDArray[np.float64]
    freqs: NDArray[np.float64]
    signals: NDArray[np.complex128]
    phases: NDArray[np.float64] = field(default_factory=_default_phase_values)
    cfg_snapshot: FreqGainCfg | None = None


class FreqGainModuleCfg(ConfigBase):
    reset: ResetCfg | None = None
    init_pulse: PulseCfg | None = None
    tested_reset: BathResetCfg
    readout: ReadoutCfg


class FreqGainSweepCfg(ConfigBase):
    gain: SweepCfg
    freq: SweepCfg


class FreqGainCfg(ProgramV2Cfg, ExpCfgModel):
    modules: FreqGainModuleCfg
    sweep: FreqGainSweepCfg


def bathreset_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return (
        np.abs(signals[2] - signals[0]) ** 2 + np.abs(signals[3] - signals[1]) ** 2
    )  # (gain, freq)


class FreqGainExp(PersistableExperiment[FreqGainResult, FreqGainCfg]):
    AXES_SPEC = AxesSpec(
        axes=(
            Axis("freqs", "Cavity Frequency", "Hz", scale=MHZ_TO_HZ, dtype=np.float64),
            Axis(
                "gains", "Cavity drive Gain", "a.u.", scale=IDENTITY, dtype=np.float64
            ),
            Axis("phases", "Pi2 Phase", "deg", scale=IDENTITY, dtype=np.float64),
        ),
        z=ZSpec("signals", "Signal", "a.u.", dtype=np.complex128),
        result_type=FreqGainResult,
        cfg_type=FreqGainCfg,
        tag="twotone/reset/bath/freq_gain",
    )

    @record_result
    def run(
        self,
        soc,
        soccfg,
        cfg: FreqGainCfg,
        *,
        acquire_kwargs: dict[str, Any] | None = None,
    ) -> FreqGainResult:
        orig_cfg = deepcopy(cfg)
        setup_devices(cfg, progress=True)
        modules = cfg.modules

        reset_cfg = modules.tested_reset
        phases = _default_phase_values()
        gains = sweep2array(
            cfg.sweep.gain,
            "gain",
            {"soccfg": soccfg, "gen_ch": reset_cfg.cavity_tone_cfg.ch},
        )
        freqs = sweep2array(
            cfg.sweep.freq,
            "freq",
            {"soccfg": soccfg, "gen_ch": reset_cfg.cavity_tone_cfg.ch},
        )

        def measure_fn(
            ctx: TaskState[NDArray[np.complex128], Any, FreqGainCfg],
            update_hook: Callable | None,
        ) -> list[NDArray[np.float64]]:
            cfg = ctx.cfg
            modules = cfg.modules

            tested_reset = modules.tested_reset

            gain_sweep = cfg.sweep.gain
            freq_sweep = cfg.sweep.freq
            gain_param = sweep2param("gain", gain_sweep)
            freq_param = sweep2param("freq", freq_sweep)
            tested_reset.set_param("res_gain", gain_param)
            tested_reset.set_param("res_freq", freq_param)

            phase_param = QickSweep1D("phase", 0.0, 270.0) + tested_reset.pi2_cfg.phase
            tested_reset.set_param("pi2_phase", phase_param)

            return ModularProgramV2(
                soccfg,
                cfg,
                modules=[
                    Reset("reset", modules.reset),
                    Pulse("init_pulse", modules.init_pulse),
                    BathReset("tested_reset", tested_reset),
                    Readout("readout", modules.readout),
                ],
                sweep=[
                    ("phase", 4),  # 0, 90, 180, 270 degrees
                    ("gain", gain_sweep),
                    ("freq", freq_sweep),
                ],
            ).acquire(
                soc,
                progress=False,
                round_hook=update_hook,
                stop_checkers=[ctx.is_stop],
                **(acquire_kwargs or {}),
            )

        with LivePlot2D("Cavity Frequency (MHz)", "Cavity drive Gain (a.u.)") as viewer:
            signals = run_task(
                task=Task(
                    measure_fn=measure_fn,
                    result_shape=(4, len(gains), len(freqs)),
                    pbar_n=cfg.rounds,
                ),
                init_cfg=cfg,
                on_update=lambda ctx: viewer.update(
                    freqs, gains, bathreset_signal2real(ctx.root_data).T
                ),
            )

        return FreqGainResult(
            gains=gains,
            freqs=freqs,
            phases=phases,
            signals=signals,
            cfg_snapshot=orig_cfg,
        )

    @retrieve_result
    def analyze(
        self,
        result: FreqGainResult | None = None,
        smooth: float = 1.0,
        *,
        smooth_method: SmoothMethod = "wavelet",
        wavelet: str = "sym4",
        wavelet_level: int = 0,
    ) -> tuple[float, float, Figure]:
        assert result is not None, "no result found"

        gains, freqs, signals = result.gains, result.freqs, result.signals

        # Find peak in amplitude
        smooth_signals = smooth_signal_nd(
            signals,
            method=smooth_method,
            sigma=smooth,
            axes=(1, 2),
            wavelet=wavelet,
            wavelet_level=wavelet_level,
        ).astype(np.complex128)
        real_signals = bathreset_signal2real(smooth_signals)

        gain_opt = gains[np.argmax(np.max(real_signals, axis=1))]
        freq_opt = freqs[np.argmax(np.max(real_signals, axis=0))]

        fig, ax = plt.subplots()
        assert isinstance(fig, Figure)

        ax.imshow(
            real_signals,
            aspect="auto",
            origin="lower",
            interpolation="none",
            extent=(freqs[0], freqs[-1], gains[0], gains[-1]),
        )
        peak_label = f"({gain_opt:.2f} a.u., {freq_opt:.1f} MHz)"
        ax.scatter(freq_opt, gain_opt, color="r", s=40, marker="*", label=peak_label)
        ax.set_xlabel("Cavity Frequency (MHz)", fontsize="x-large")
        ax.set_ylabel("Cavity drive Gain (a.u.)", fontsize="x-large")
        ax.legend(fontsize="x-large")
        ax.tick_params(axis="both", which="major", labelsize=12)

        fig.tight_layout()

        return gain_opt, freq_opt, fig
