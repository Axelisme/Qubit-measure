from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment import (
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
from zcu_tools.program.v2 import SweepCfg, TwoToneCfg, sweep2param
from zcu_tools.utils.fitting import fit_qubit_freq
from zcu_tools.utils.process import minus_background


@dataclass(frozen=True)
class FreqResult:
    freqs: NDArray[np.float64]
    signals: NDArray[np.complex128]
    cfg_snapshot: FreqCfg | None = None


def qubfreq_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return np.abs(minus_background(signals))


class FreqSweepCfg(ConfigBase):
    freq: SweepCfg


class FreqCfg(TwoToneCfg, ExpCfgModel):
    sweep: FreqSweepCfg


class FreqExp(PersistableExperiment[FreqResult, FreqCfg]):
    # freq stores MHz on disk -> scale=IDENTITY (1.0)
    AXES_SPEC = AxesSpec(
        axes=(Axis("freqs", "Frequency", "MHz"),),
        z=ZSpec("signals", "Signal", "a.u."),
        result_type=FreqResult,
        cfg_type=FreqCfg,
        tag="twotone/freq",
    )

    @record_result
    def run(
        self,
        soc,
        soccfg,
        cfg: FreqCfg,
        *,
        acquire_kwargs: dict[str, Any] | None = None,
    ) -> FreqResult:
        orig_cfg = deepcopy(cfg)
        setup_devices(cfg, progress=True)
        modules = cfg.modules

        # predicted sweep points before FPGA coercion
        freqs = sweep2array(
            cfg.sweep.freq, "freq", {"soccfg": soccfg, "gen_ch": modules.qub_pulse.ch}
        )

        with LivePlot1D("Frequency (MHz)", "Amplitude") as viewer:
            signals_buffer = SignalBuffer(
                (len(freqs),),
                on_update=lambda data: viewer.update(freqs, qubfreq_signal2real(data)),
            )

            with Schedule(cfg, signals_buffer) as sched:
                cfg = sched.cfg
                modules = cfg.modules

                freq_sweep = cfg.sweep.freq
                modules.qub_pulse.set_param("freq", sweep2param("freq", freq_sweep))

                _ = (
                    sched.prog_builder(soc, soccfg)
                    .add_reset("reset", modules.reset)
                    .add_pulse("qub_pulse", modules.qub_pulse)
                    .add_readout("readout", modules.readout)
                    .declare_sweep("freq", freq_sweep)
                    .build_and_acquire(
                        **(acquire_kwargs or {}),
                    )
                )

            return FreqResult(
                freqs=freqs,
                signals=signals_buffer.array,
                cfg_snapshot=orig_cfg,
            )

    @retrieve_result
    def analyze(
        self,
        result: FreqResult | None = None,
        *,
        model_type: Literal["lor", "sinc"] = "lor",
        plot_fit: bool = True,
    ) -> tuple[float, float, float, float, Figure]:
        assert result is not None, "no result found"

        freqs = result.freqs
        signals = result.signals

        # discard NaNs (possible early abort)
        val_mask = ~np.isnan(signals)
        freqs = freqs[val_mask]
        signals = signals[val_mask]

        real_signals = qubfreq_signal2real(signals)

        # fit_qubit_freq computes both fit uncertainties; surface them so the GUI
        # summary carries freq_err / fwhm_err (the figure title already shows them).
        freq, freq_err, fwhm, fwhm_err, y_fit, _ = fit_qubit_freq(
            freqs, real_signals, model_type
        )

        fig, ax = plt.subplots(figsize=config.figsize)
        assert isinstance(fig, Figure)

        ax.plot(freqs, real_signals, label="signal", marker="o", markersize=3)
        if plot_fit:
            ax.plot(freqs, y_fit, label=f"fit, FWHM={fwhm:.1g} MHz")
            label = f"f_q = {freq:.5g} ± {freq_err:.1g} MHz"
            ax.axvline(freq, color="r", ls="--", label=label)
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Signal Real (a.u.)")
        ax.legend()
        ax.grid(True)

        fig.tight_layout()

        return freq, freq_err, fwhm, fwhm_err, fig
