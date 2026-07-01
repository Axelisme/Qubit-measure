from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment import (
    MHZ_TO_HZ,
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
from zcu_tools.liveplot import LivePlot2D, LivePlot2DwithLine
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
from zcu_tools.utils.process import (
    SmoothMethod,
    minus_background,
    rotate2real,
    smooth_signal_nd,
)


def dual_reset_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return np.abs(minus_background(signals))


@dataclass(frozen=True)
class FreqResult:
    freqs1: NDArray[np.float64]
    freqs2: NDArray[np.float64]
    signals: NDArray[np.complex128]
    cfg_snapshot: FreqCfg | None = None


class FreqModuleCfg(ConfigBase):
    reset: ResetCfg | None = None
    init_pulse: PulseCfg | None = None
    tested_reset: TwoPulseResetCfg
    readout: ReadoutCfg


class FreqSweepCfg(ConfigBase):
    freq1: SweepCfg
    freq2: SweepCfg


class FreqCfg(ProgramV2Cfg, ExpCfgModel):
    modules: FreqModuleCfg
    sweep: FreqSweepCfg


class FreqExp(PersistableExperiment[FreqResult, FreqCfg]):
    # signals memory layout is (Nfreq1, Nfreq2) = (outer, inner); native save/load
    # expect z == (outer, inner) == reversed(axes lengths), so axes order is
    # (freqs2 inner, freqs1 outer). both axes store MHz on disk (disk Hz).
    AXES_SPEC = AxesSpec(
        axes=(
            Axis("freqs2", "Frequency2", "Hz", scale=MHZ_TO_HZ),
            Axis("freqs1", "Frequency1", "Hz", scale=MHZ_TO_HZ),
        ),
        z=ZSpec("signals", "Signal", "a.u."),
        result_type=FreqResult,
        cfg_type=FreqCfg,
        tag="twotone/reset/dual_tone/freq",
    )

    @record_result
    def run_soft(
        self,
        soc,
        soccfg,
        cfg: FreqCfg,
        *,
        acquire_kwargs: dict[str, Any] | None = None,
    ) -> FreqResult:
        setup_devices(cfg, progress=True)
        modules = cfg.modules

        freq1_sweep = cfg.sweep.freq1
        freq2_sweep = cfg.sweep.freq2

        reset_cfg = modules.tested_reset
        freqs1 = sweep2array(
            freq1_sweep,
            "freq",
            {"soccfg": soccfg, "gen_ch": reset_cfg.pulse1_cfg.ch},
        )
        freqs2 = sweep2array(
            freq2_sweep,
            "freq",
            {"soccfg": soccfg, "gen_ch": reset_cfg.pulse2_cfg.ch},
            allow_array=True,
        )

        def measure_fn(
            ctx: TaskState[NDArray[np.complex128], Any, FreqCfg],
            update_hook: Callable | None,
        ) -> list[NDArray[np.float64]]:
            cfg = ctx.cfg
            modules = cfg.modules

            freq1_param = sweep2param("freq1", cfg.sweep.freq1)
            modules.tested_reset.set_param("freq1", freq1_param)

            return ModularProgramV2(
                soccfg,
                cfg,
                sweep=[("freq1", cfg.sweep.freq1)],
                modules=[
                    Reset("reset", modules.reset),
                    Pulse("init_pulse", modules.init_pulse),
                    TwoPulseReset("tested_reset", modules.tested_reset),
                    Readout("readout", modules.readout),
                ],
            ).acquire(
                soc,
                progress=False,
                round_hook=update_hook,
                stop_checkers=[ctx.is_stop],
                **(acquire_kwargs or {}),
            )

        with LivePlot2DwithLine(
            "Frequency1 (MHz)", "Frequency2 (MHz)", line_axis=0
        ) as viewer:
            with MeasureSession(cfg) as run:
                signals_buffer = run.buffer(
                    (len(freqs1), len(freqs2)),
                    dtype=np.complex128,
                    on_update=lambda data: viewer.update(
                        freqs1, freqs2, dual_reset_signal2real(data)
                    ),
                )
                for step in run.scan("freq2", freqs2.tolist()):
                    step.cfg.modules.tested_reset.set_param("freq2", step.value)
                    signals_buffer[:, step].measure(measure_fn, pbar_n=step.cfg.rounds)
                signals = signals_buffer.array

        return FreqResult(freqs1, freqs2, signals, cfg_snapshot=cfg)

    @record_result
    def run_hard(
        self,
        soc,
        soccfg,
        cfg: FreqCfg,
        *,
        acquire_kwargs: dict[str, Any] | None = None,
    ) -> FreqResult:
        setup_devices(cfg, progress=True)
        modules = cfg.modules

        reset_cfg = modules.tested_reset
        freqs1 = sweep2array(
            cfg.sweep.freq1,
            "freq",
            {"soccfg": soccfg, "gen_ch": reset_cfg.pulse1_cfg.ch},
        )
        freqs2 = sweep2array(
            cfg.sweep.freq2,
            "freq",
            {"soccfg": soccfg, "gen_ch": reset_cfg.pulse2_cfg.ch},
        )

        def measure_fn(
            ctx: TaskState[NDArray[np.complex128], Any, FreqCfg],
            update_hook: Callable | None,
        ) -> list[NDArray[np.float64]]:
            cfg = ctx.cfg
            modules = cfg.modules

            freq1_param = sweep2param("freq1", cfg.sweep.freq1)
            freq2_param = sweep2param("freq2", cfg.sweep.freq2)
            modules.tested_reset.set_param("freq1", freq1_param)
            modules.tested_reset.set_param("freq2", freq2_param)

            return ModularProgramV2(
                soccfg,
                cfg,
                sweep=[("freq1", cfg.sweep.freq1), ("freq2", cfg.sweep.freq2)],
                modules=[
                    Reset("reset", modules.reset),
                    Pulse("init_pulse", modules.init_pulse),
                    TwoPulseReset("tested_reset", modules.tested_reset),
                    Readout("readout", modules.readout),
                ],
            ).acquire(
                soc,
                progress=False,
                round_hook=update_hook,
                stop_checkers=[ctx.is_stop],
                **(acquire_kwargs or {}),
            )

        with LivePlot2D("Frequency1 (MHz)", "Frequency2 (MHz)") as viewer:
            with MeasureSession(cfg) as run:
                signals_buffer = run.buffer(
                    (len(freqs1), len(freqs2)),
                    dtype=np.complex128,
                    on_update=lambda data: viewer.update(
                        freqs1, freqs2, dual_reset_signal2real(data)
                    ),
                )
                signals_buffer.measure(measure_fn, pbar_n=run.cfg.rounds)
                signals = signals_buffer.array

        return FreqResult(freqs1, freqs2, signals, cfg_snapshot=cfg)

    def run(
        self,
        soc,
        soccfg,
        cfg: FreqCfg,
        *,
        method: Literal["soft", "hard"] = "soft",
        acquire_kwargs: dict[str, Any] | None = None,
    ) -> FreqResult:
        if method == "soft":
            return self.run_soft(soc, soccfg, cfg, acquire_kwargs=acquire_kwargs)
        else:
            return self.run_hard(soc, soccfg, cfg, acquire_kwargs=acquire_kwargs)

    @retrieve_result
    def analyze(
        self,
        result: FreqResult | None = None,
        *,
        smooth: float = 1.0,
        smooth_method: SmoothMethod = "wavelet",
        wavelet: str = "sym4",
        wavelet_level: int = 0,
        xname: str | None = None,
        yname: str | None = None,
        corner_as_background: bool = False,
    ) -> tuple[float, float, Figure]:
        assert result is not None, "no result found"

        freqs1, freqs2, signals = result.freqs1, result.freqs2, result.signals

        # Apply smoothing for peak finding
        signals_smooth = smooth_signal_nd(
            signals,
            method=smooth_method,
            sigma=smooth,
            axes=(0, 1),
            wavelet=wavelet,
            wavelet_level=wavelet_level,
        )

        # Find peak in amplitude
        if corner_as_background:
            amps = np.abs(signals_smooth - signals_smooth[0, 0])
        else:
            amps = np.abs(minus_background(signals_smooth))

        freq1_opt = freqs1[np.argmax(np.max(amps, axis=1))]
        freq2_opt = freqs2[np.argmax(np.max(amps, axis=0))]

        fig, ax = plt.subplots(figsize=config.figsize)
        assert isinstance(fig, Figure)

        ax.imshow(
            rotate2real(signals.T).real,
            aspect="auto",
            origin="lower",
            interpolation="none",
            extent=(freqs1[0], freqs1[-1], freqs2[0], freqs2[-1]),
        )
        peak_label = f"({freq1_opt:.1f}, {freq2_opt:.1f}) MHz"
        ax.scatter(freq1_opt, freq2_opt, color="r", s=40, marker="*", label=peak_label)
        if xname is not None:
            ax.set_xlabel(f"{xname} Frequency (MHz)", fontsize=14)
        if yname is not None:
            ax.set_ylabel(f"{yname} Frequency (MHz)", fontsize=14)
        ax.legend(fontsize="x-large")
        ax.tick_params(axis="both", which="major", labelsize=12)

        fig.tight_layout()

        return freq1_opt, freq2_opt, fig
