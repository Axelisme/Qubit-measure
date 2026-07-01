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
    SweepCfg,
    TwoToneCfg,
    sweep2param,
)
from zcu_tools.utils.fitting import fit_rabi
from zcu_tools.utils.process import rotate2real


@dataclass(frozen=True)
class LenRabiResult:
    lengths: NDArray[np.float64]
    signals: NDArray[np.complex128]
    cfg_snapshot: LenRabiCfg | None = None


def rabi_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return rotate2real(signals).real


class LenRabiSweepCfg(ConfigBase):
    length: SweepCfg


class LenRabiCfg(TwoToneCfg, ExpCfgModel):
    sweep: LenRabiSweepCfg


class LenRabiExp(PersistableExperiment[LenRabiResult, LenRabiCfg]):
    # lengths stored in seconds on disk (mem us) -> scale=US_TO_S; z complex
    AXES_SPEC = AxesSpec(
        axes=(Axis("lengths", "Length", "s", US_TO_S),),
        z=ZSpec("signals", "Signal", "a.u."),
        result_type=LenRabiResult,
        cfg_type=LenRabiCfg,
        tag="twotone/ge/rabi_length",
    )

    def _run_for_flat(
        self,
        soc,
        soccfg,
        cfg: LenRabiCfg,
        acquire_kwargs: dict[str, Any] | None = None,
    ) -> LenRabiResult:
        orig_cfg = deepcopy(cfg)

        setup_devices(cfg, progress=True)
        modules = cfg.modules

        assert modules.qub_pulse.waveform.style in ["const", "flat_top"], (
            "This method only supports const and flat_top pulse style"
        )

        # initial values, may be rounded later
        lengths = sweep2array(
            cfg.sweep.length,
            "time",
            {"soccfg": soccfg, "gen_ch": modules.qub_pulse.ch},
        )

        with LivePlot1D("Length (us)", "Signal") as viewer:
            signals_buffer = SignalBuffer(
                (len(lengths),),
                on_update=lambda data: viewer.update(lengths, rabi_signal2real(data)),
            )
            with Schedule(cfg, signals_buffer) as sched:
                cfg = sched.cfg
                modules = cfg.modules
                length_sweep = cfg.sweep.length
                modules.qub_pulse.set_param(
                    "length", sweep2param("length", length_sweep)
                )

                _ = (
                    sched.prog_builder(soc, soccfg)
                    .add_reset("reset", modules.reset)
                    .add_pulse("init_pulse", modules.init_pulse)
                    .add_pulse("qubit_pulse", modules.qub_pulse)
                    .add_readout("readout", modules.readout)
                    .declare_sweep("length", length_sweep)
                    .build_and_acquire(
                        **(acquire_kwargs or {}),
                    )
                )
            return LenRabiResult(
                lengths=lengths,
                signals=signals_buffer.array,
                cfg_snapshot=orig_cfg,
            )

    def _run_for_arb(
        self,
        soc,
        soccfg,
        cfg: LenRabiCfg,
        acquire_kwargs: dict[str, Any] | None = None,
    ) -> LenRabiResult:
        orig_cfg = deepcopy(cfg)

        setup_devices(cfg, progress=True)
        modules = cfg.modules

        rounds = cfg.rounds
        _cfg = cfg.model_copy(deep=True)
        _cfg.rounds = 1  # we'll handle the rounds in the task loop

        length_sweep = _cfg.sweep.length

        lengths = sweep2array(
            length_sweep,
            "time",
            {"soccfg": soccfg, "gen_ch": modules.qub_pulse.ch},
        )
        lengths = np.unique(lengths)  # remove duplicates

        programs: dict[float, Any] = {}

        def average_round(signals: NDArray[np.complex128]) -> NDArray[np.complex128]:
            _signals = np.asarray(signals)  # shape: (rounds, len(lengths))
            mask = np.any(~np.isnan(_signals), axis=0)
            mean_signals = np.full(_signals.shape[1], np.nan, dtype=np.complex128)
            mean_signals[mask] = np.nanmean(_signals[:, mask], axis=0)
            return mean_signals

        with LivePlot1D("Length (us)", "Signal") as viewer:
            length_values = lengths.tolist()
            signals_buffer = SignalBuffer(
                (rounds, len(lengths)),
                on_update=lambda data: viewer.update(
                    lengths, rabi_signal2real(average_round(data))
                ),
            )
            with Schedule(_cfg, signals_buffer) as sched:
                for _, rep in sched.repeat("round", rounds):
                    for length, step in rep.scan("length", length_values):
                        modules = step.cfg.modules
                        modules.qub_pulse.set_param("length", length)
                        builder = (
                            step.prog_builder(soc, soccfg)
                            .add_reset("reset", modules.reset)
                            .add_pulse("init_pulse", modules.init_pulse)
                            .add_pulse("qubit_pulse", modules.qub_pulse)
                            .add_readout("readout", modules.readout)
                        )
                        length_key = float(length)
                        if length_key not in programs:
                            programs[length_key] = builder.build()
                        _ = builder.run_program(
                            programs[length_key],
                            **(acquire_kwargs or {}),
                        )
            return LenRabiResult(
                lengths=lengths,
                signals=average_round(signals_buffer.array),
                cfg_snapshot=orig_cfg,
            )

    @record_result
    def run(
        self,
        soc,
        soccfg,
        cfg: LenRabiCfg,
        *,
        acquire_kwargs: dict[str, Any] | None = None,
    ) -> LenRabiResult:
        modules = cfg.modules
        qub_waveform = modules.qub_pulse.waveform

        if qub_waveform.style in ["const", "flat_top"]:
            # use hard sweep for flat top pulse
            return self._run_for_flat(soc, soccfg, cfg, acquire_kwargs=acquire_kwargs)
        else:
            # use soft sweep for arb pulse
            return self._run_for_arb(soc, soccfg, cfg, acquire_kwargs=acquire_kwargs)

    @retrieve_result
    def analyze(
        self, result: LenRabiResult | None = None, *, decay: bool = True
    ) -> tuple[float, float, float, float, float, float, Figure]:
        assert result is not None, "no result found"

        lens, signals = result.lengths, result.signals

        real_signals = rabi_signal2real(signals)

        nan_mask = np.isnan(real_signals)
        if np.all(nan_mask):
            raise ValueError("All data are NaN!")

        lens = lens[~nan_mask]
        real_signals = real_signals[~nan_mask]

        pi_len, pi_len_err, pi2_len, pi2_len_err, freq, freq_err, y_fit, _ = fit_rabi(
            lens, real_signals, decay=decay, init_phase=None
        )

        fig, ax = plt.subplots(figsize=config.figsize)
        assert isinstance(fig, Figure)

        ax.plot(lens, real_signals, label="meas", ls="-", marker="o", markersize=3)
        ax.plot(lens, y_fit, label="fit")
        ax.axvline(
            pi_len,
            ls="--",
            c="red",
            label=f"pi = {pi_len:.3g} ± {pi_len_err:.2g} μs",
        )
        ax.axvspan(pi_len - pi_len_err, pi_len + pi_len_err, color="red", alpha=0.2)
        ax.axvline(
            pi2_len,
            ls="--",
            c="red",
            label=f"pi/2 = {pi2_len:.3g} ± {pi2_len_err:.2g} μs",
        )
        ax.axvspan(pi2_len - pi2_len_err, pi2_len + pi2_len_err, color="red", alpha=0.2)
        ax.set_xlabel("Pulse length (μs)")
        ax.set_ylabel("Signal Real (a.u.)")
        ax.set_title(f"Rabi Oscillation (f={freq:.3f} ± {freq_err:.3f} MHz)")
        ax.legend(loc=4)
        ax.grid(True)

        fig.tight_layout()

        # fit_rabi computes the per-quantity fit uncertainties; surface them so the
        # GUI summary carries pi_len_err / pi2_len_err / rabi_f_err (the figure
        # labels already show pi/pi2 errors and the title shows the freq error).
        return pi_len, pi_len_err, pi2_len, pi2_len_err, freq, freq_err, fig
