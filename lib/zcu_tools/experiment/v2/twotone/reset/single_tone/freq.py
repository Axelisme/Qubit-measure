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
from zcu_tools.experiment.v2.runner import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.liveplot import LivePlot1D
from zcu_tools.program.v2 import (
    ProgramV2Cfg,
    Pulse,
    PulseCfg,
    PulseReset,
    PulseResetCfg,
    Readout,
    ReadoutCfg,
    Reset,
    ResetCfg,
    SweepCfg,
    sweep2param,
)
from zcu_tools.utils.fitting import fit_qubit_freq
from zcu_tools.utils.process import rotate2real


@dataclass(frozen=True)
class FreqResult:
    freqs: NDArray[np.float64]
    signals: NDArray[np.complex128]
    cfg_snapshot: FreqCfg | None = None


def reset_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return rotate2real(signals).real


class FreqModuleCfg(ConfigBase):
    reset: ResetCfg | None = None
    init_pulse: PulseCfg | None = None
    tested_reset: PulseResetCfg
    readout: ReadoutCfg


class FreqSweepCfg(ConfigBase):
    freq: SweepCfg


class FreqCfg(ProgramV2Cfg, ExpCfgModel):
    modules: FreqModuleCfg
    sweep: FreqSweepCfg


class FreqExp(PersistableExperiment[FreqResult, FreqCfg]):
    AXES_SPEC = AxesSpec(
        axes=(Axis("freqs", "Frequency", "Hz", scale=MHZ_TO_HZ),),
        z=ZSpec("signals", "Signal", "a.u.", dtype=np.complex128),
        result_type=FreqResult,
        cfg_type=FreqCfg,
        tag="twotone/reset/single_tone/freq",
    )

    @record_result
    def run(
        self, soc, soccfg, cfg: FreqCfg, *, acquire_kwargs: dict[str, Any] | None = None
    ) -> FreqResult:
        orig_cfg = deepcopy(cfg)
        setup_devices(cfg, progress=True)
        modules = cfg.modules

        reset_cfg = modules.tested_reset
        freqs = sweep2array(
            cfg.sweep.freq,
            "freq",
            {"soccfg": soccfg, "gen_ch": reset_cfg.pulse_cfg.ch},
        )

        with LivePlot1D("Frequency (MHz)", "Amplitude") as viewer:
            signals_buffer = SignalBuffer(
                (len(freqs),),
                on_update=lambda data: viewer.update(freqs, reset_signal2real(data)),
            )
            with Schedule(cfg, signals_buffer) as sched:
                modules = sched.cfg.modules
                modules.tested_reset.set_param(
                    "freq", sweep2param("freq", sched.cfg.sweep.freq)
                )
                _ = (
                    sched.prog_builder(soc, soccfg)
                    .add(
                        Reset("reset", modules.reset),
                        Pulse("init_pulse", modules.init_pulse),
                        PulseReset("tested_reset", modules.tested_reset),
                        Readout("readout", modules.readout),
                    )
                    .declare_sweep("freq", sched.cfg.sweep.freq)
                    .build_and_acquire(
                        **(acquire_kwargs or {}),
                    )
                )
                signals = signals_buffer.array

        return FreqResult(freqs, signals, cfg_snapshot=orig_cfg)

    @retrieve_result
    def analyze(self, result: FreqResult | None = None) -> tuple[float, float, Figure]:
        assert result is not None, "no result found"

        freqs, signals = result.freqs, result.signals

        # Discard NaNs (possible early abort)
        val_mask = ~np.isnan(signals)
        freqs = freqs[val_mask]
        signals = signals[val_mask]

        real_signals = reset_signal2real(signals)

        freq, freq_err, fwhm, _, y_fit, _ = fit_qubit_freq(
            freqs, real_signals, type="lor"
        )

        fig, ax = plt.subplots(figsize=config.figsize)

        ax.plot(freqs, real_signals, label="signal", marker="o", markersize=3)
        ax.plot(freqs, y_fit, label=f"fit, FWHM = {fwhm:.1g} MHz")
        label = f"f_reset = {freq:.5g} ± {freq_err:.1g} MHz"
        ax.axvline(freq, color="r", ls="--", label=label)
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Signal Real (a.u.)")
        ax.legend()
        ax.grid(True)
        ax.set_title("Sideband Reset Frequency Sweep")

        fig.tight_layout()

        return freq, fwhm, fig
