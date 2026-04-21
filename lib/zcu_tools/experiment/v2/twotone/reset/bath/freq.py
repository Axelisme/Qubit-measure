from __future__ import annotations

from pathlib import Path
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter
from typeguard import check_type
from typing_extensions import (
    Any,
    NotRequired,
    Optional,
    TypeAlias,
    TypedDict,
    Callable,
    cast,
)
from qick.asm_v2 import QickSweep1D

from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.utils import setup_devices
from zcu_tools.experiment.v2.runner import Task, TaskCfg, run_task, TaskState
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.liveplot import LivePlot2D
from zcu_tools.program import SweepCfg
from zcu_tools.program.v2 import (
    ModularProgramCfg,
    ModularProgramV2,
    Pulse,
    PulseCfg,
    Readout,
    ReadoutCfg,
    Reset,
    BathReset,
    ResetCfg,
    sweep2param,
    BathResetCfg,
)
from zcu_tools.utils.datasaver import load_data, save_data

FreqGainResult: TypeAlias = tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.complex128]
]


class FreqGainModuleCfg(TypedDict, closed=True):
    reset: NotRequired[ResetCfg]
    init_pulse: NotRequired[PulseCfg]
    tested_reset: BathResetCfg
    readout: ReadoutCfg


class FreqGainCfg(ModularProgramCfg, TaskCfg):
    modules: FreqGainModuleCfg
    sweep: dict[str, SweepCfg]


def bathreset_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return (
        np.abs(signals[2] - signals[0]) ** 2 + np.abs(signals[3] - signals[1]) ** 2
    )  # (gain, freq)


class FreqGainExp(AbsExperiment[FreqGainResult, FreqGainCfg]):
    def run(
        self,
        soc,
        soccfg,
        cfg: dict[str, Any],
        *,
        acquire_kwargs: Optional[dict[str, Any]] = None,
    ) -> FreqGainResult:
        _cfg = check_type(deepcopy(cfg), FreqGainCfg)
        setup_devices(_cfg, progress=True)
        modules = _cfg["modules"]

        reset_cfg = modules["tested_reset"]
        gains = sweep2array(
            _cfg["sweep"]["gain"],
            "gain",
            {"soccfg": soccfg, "gen_ch": reset_cfg.cavity_tone_cfg.ch},
        )
        freqs = sweep2array(
            _cfg["sweep"]["freq"],
            "freq",
            {"soccfg": soccfg, "gen_ch": reset_cfg.cavity_tone_cfg.ch},
        )

        def measure_fn(
            ctx: TaskState, update_hook: Optional[Callable]
        ) -> list[NDArray[np.float64]]:
            cfg = cast(FreqGainCfg, ctx.cfg)
            modules = cfg["modules"]

            tested_reset = modules["tested_reset"]

            gain_sweep = cfg["sweep"]["gain"]
            freq_sweep = cfg["sweep"]["freq"]
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
                    Reset("reset", modules.get("reset")),
                    Pulse("init_pulse", modules.get("init_pulse")),
                    BathReset("tested_reset", tested_reset),
                    Readout("readout", modules["readout"]),
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
                **(acquire_kwargs or {}),
            )

        with LivePlot2D("Cavity Frequency (MHz)", "Cavity drive Gain (a.u.)") as viewer:
            signals = run_task(
                task=Task(
                    measure_fn=measure_fn,
                    result_shape=(4, len(gains), len(freqs)),
                    pbar_n=_cfg["rounds"],
                ),
                init_cfg=_cfg,
                on_update=lambda ctx: viewer.update(
                    freqs, gains, bathreset_signal2real(ctx.root_data).T
                ),
            )

        # Cache results
        self.last_cfg = _cfg
        self.last_result = (gains, freqs, signals)

        return gains, freqs, signals

    def analyze(
        self, result: Optional[FreqGainResult] = None, smooth: float = 1.0
    ) -> tuple[float, float, Figure]:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        gains, freqs, signals = result

        # Find peak in amplitude
        smooth_signals: NDArray[np.complex128] = gaussian_filter(
            signals, sigma=smooth, axes=(1, 2)
        )  # type: ignore
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
        result: Optional[FreqGainResult] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/reset/bath/freq_gain",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        gains, freqs, signals = result

        _filepath = Path(filepath)

        # 0 signals
        save_data(
            filepath=str(_filepath.with_name(_filepath.name + "_0deg")),
            x_info={"name": "Cavity drive Gain", "unit": "a.u.", "values": gains},
            y_info={"name": "Cavity Frequency", "unit": "Hz", "values": freqs * 1e6},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals[0].T},
            comment=comment,
            tag=tag,
            **kwargs,
        )

        # 90 signals
        save_data(
            filepath=str(_filepath.with_name(_filepath.name + "_90deg")),
            x_info={"name": "Cavity drive Gain", "unit": "a.u.", "values": gains},
            y_info={"name": "Cavity Frequency", "unit": "Hz", "values": freqs * 1e6},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals[1].T},
            comment=comment,
            tag=tag,
            **kwargs,
        )

        # 180 signals
        save_data(
            filepath=str(_filepath.with_name(_filepath.name + "_180deg")),
            x_info={"name": "Cavity drive Gain", "unit": "a.u.", "values": gains},
            y_info={"name": "Cavity Frequency", "unit": "Hz", "values": freqs * 1e6},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals[2].T},
            comment=comment,
            tag=tag,
            **kwargs,
        )

        # 270 signals
        save_data(
            filepath=str(_filepath.with_name(_filepath.name + "_270deg")),
            x_info={"name": "Cavity drive Gain", "unit": "a.u.", "values": gains},
            y_info={"name": "Cavity Frequency", "unit": "Hz", "values": freqs * 1e6},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals[3].T},
            comment=comment,
            tag=tag,
            **kwargs,
        )

    def load(self, filepath: list[str], **kwargs) -> FreqGainResult:
        deg0_filepath, deg90_filepath, deg180_filepath, deg270_filepath = filepath

        deg0_signals, gains, freqs, cfg = load_data(
            deg0_filepath, return_cfg=True, **kwargs
        )
        assert gains is not None and freqs is not None
        assert len(gains.shape) == 1 and len(freqs.shape) == 1
        assert deg0_signals.shape == (len(freqs), len(gains))

        freqs = freqs * 1e-6  # Hz -> MHz
        deg0_signals = deg0_signals.T  # transpose back

        deg90_signals, *_ = load_data(deg90_filepath, **kwargs)
        assert gains is not None and freqs is not None
        assert len(gains.shape) == 1 and len(freqs.shape) == 1
        assert deg90_signals.shape == (len(freqs), len(gains))

        deg90_signals = deg90_signals.T  # transpose back

        deg180_signals, *_ = load_data(deg180_filepath, **kwargs)
        assert gains is not None and freqs is not None
        assert len(gains.shape) == 1 and len(freqs.shape) == 1
        assert deg180_signals.shape == (len(freqs), len(gains))

        deg180_signals = deg180_signals.T  # transpose back

        deg270_signals, *_ = load_data(deg270_filepath, **kwargs)
        assert gains is not None and freqs is not None
        assert len(gains.shape) == 1 and len(freqs.shape) == 1
        assert deg270_signals.shape == (len(freqs), len(gains))

        deg270_signals = deg270_signals.T  # transpose back

        gains = gains.astype(np.float64)
        freqs = freqs.astype(np.float64)
        signals = np.stack(
            [deg0_signals, deg90_signals, deg180_signals, deg270_signals], axis=0
        ).astype(np.complex128)

        self.last_cfg = cast(FreqGainCfg, cfg)
        self.last_result = (gains, freqs, signals)

        return gains, freqs, signals
