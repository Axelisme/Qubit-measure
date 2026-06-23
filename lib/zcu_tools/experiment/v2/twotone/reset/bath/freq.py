from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from qick.asm_v2 import QickSweep1D

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import make_comment, parse_comment, setup_devices
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
from zcu_tools.utils.datasaver import safe_labber_filepath
from zcu_tools.utils.labber_io import load_labber_data, save_labber_data
from zcu_tools.utils.process import SmoothMethod, smooth_signal_nd


@dataclass(frozen=True)
class FreqGainResult:
    gains: NDArray[np.float64]
    freqs: NDArray[np.float64]
    signals: NDArray[np.complex128]
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


class FreqGainExp(AbsExperiment[FreqGainResult, FreqGainCfg]):
    def run(
        self,
        soc,
        soccfg,
        cfg: FreqGainCfg,
        *,
        acquire_kwargs: dict[str, Any] | None = None,
    ) -> FreqGainResult:
        setup_devices(cfg, progress=True)
        modules = cfg.modules

        reset_cfg = modules.tested_reset
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

        # Cache results
        self.last_result = FreqGainResult(gains, freqs, signals, cfg_snapshot=cfg)

        return self.last_result

    def analyze(
        self,
        result: FreqGainResult | None = None,
        smooth: float = 1.0,
        *,
        smooth_method: SmoothMethod = "wavelet",
        wavelet: str = "sym4",
        wavelet_level: int = 0,
    ) -> tuple[float, float, Figure]:
        if result is None:
            result = self.last_result
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

    def save(
        self,
        filepath: str,
        result: FreqGainResult | None = None,
        comment: str | None = None,
        tag: str = "twotone/reset/bath/freq_gain",
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        gains, freqs, signals = result.gains, result.freqs, result.signals

        _filepath = Path(filepath)

        if result.cfg_snapshot is None:
            raise ValueError("Cannot save result without configuration snapshot")
        cfg = result.cfg_snapshot
        comment = make_comment(cfg, comment)

        # one Labber file per interference phase (0/90/180/270 deg); each stores
        # z as native (Ny=freqs, Nx=gains) so the on-disk axis identity is
        # x=gain (inner) / y=freq (outer)
        for k, suffix in enumerate(("_0deg", "_90deg", "_180deg", "_270deg")):
            path = safe_labber_filepath(
                str(_filepath.with_name(_filepath.name + suffix))
            )
            save_labber_data(
                path,
                z=("Signal", "a.u.", signals[k].T),
                axes=[
                    ("Cavity drive Gain", "a.u.", gains),
                    ("Cavity Frequency", "Hz", freqs * 1e6),
                ],
                comment=comment,
                tags=tag,
            )

    def load(self, filepath: list[str]) -> FreqGainResult:
        deg0_filepath, deg90_filepath, deg180_filepath, deg270_filepath = filepath

        ld0 = load_labber_data(deg0_filepath)
        gains = np.asarray(ld0.axes[0].values, dtype=np.float64)
        freqs = np.asarray(ld0.axes[1].values, dtype=np.float64)
        assert len(gains.shape) == 1 and len(freqs.shape) == 1
        assert ld0.z.shape == (len(freqs), len(gains))
        comment = ld0.comment

        freqs = freqs * 1e-6  # Hz -> MHz
        deg0_signals = ld0.z.T  # (freqs, gains) -> (gains, freqs)

        def _load_phase(path: str) -> NDArray[np.complex128]:
            ld = load_labber_data(path)
            assert ld.z.shape == (len(freqs), len(gains))
            return ld.z.T  # (freqs, gains) -> (gains, freqs)

        deg90_signals = _load_phase(deg90_filepath)
        deg180_signals = _load_phase(deg180_filepath)
        deg270_signals = _load_phase(deg270_filepath)

        signals = np.stack(
            [deg0_signals, deg90_signals, deg180_signals, deg270_signals], axis=0
        ).astype(np.complex128)

        cfg_snapshot = None
        if comment is not None:
            cfg, _, _ = parse_comment(comment)
            if cfg is not None:
                cfg_snapshot = FreqGainCfg.validate_or_warn(
                    cfg, source=str(deg0_filepath)
                )
        self.last_result = FreqGainResult(
            gains, freqs, signals, cfg_snapshot=cfg_snapshot
        )

        return self.last_result
