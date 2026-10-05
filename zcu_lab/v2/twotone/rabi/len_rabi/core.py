from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from zcu_tools.analysis.fitting import FitQuality, compute_fit_quality, fit_rabi
from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment import (
    US_TO_S,
    AxesSpec,
    Axis,
    PersistableExperiment,
    ZSpec,
    config,
)
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.utils import setup_devices
from zcu_tools.experiment.v2.runtime.schedule import Schedule
from zcu_tools.experiment.v2.runtime.schedule import SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    ProgramV2Cfg,
    PulseCfg,
    ReadoutCfg,
    ResetCfg,
    SweepCfg,
    sweep2param,
)
from zcu_tools.utils.process import rotate2real


@dataclass(frozen=True)
class LenRabiResult:
    lengths: NDArray[np.float64]
    signals: NDArray[np.complex128]


@dataclass(frozen=True)
class LenRabiAnalyzeOptions:
    decay: bool = True
    fit_phase: bool = False


@dataclass(frozen=True)
class LenRabiAnalysis:
    pi_len: float
    pi_len_err: float
    pi2_len: float
    pi2_len_err: float
    rabi_f: float
    rabi_f_err: float
    fit_quality: Mapping[str, FitQuality] | None = None


def rabi_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return rotate2real(signals).real


class LenRabiSweepCfg(ConfigBase):
    length: SweepCfg


class LenRabiModuleCfg(ConfigBase):
    reset: ResetCfg | None = None
    init_pulse: PulseCfg | None = None
    qub_pulse: PulseCfg
    readout: ReadoutCfg


class LenRabiCfg(ProgramV2Cfg, ExpCfgModel):
    modules: LenRabiModuleCfg
    sweep: LenRabiSweepCfg


class LenRabiExp(PersistableExperiment[LenRabiResult, LenRabiCfg]):
    Options: ClassVar[type[LenRabiAnalyzeOptions]] = LenRabiAnalyzeOptions

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
        cfg: LenRabiCfg,
        *,
        context: RunContext,
    ) -> LenRabiResult:
        cfg = deepcopy(cfg)
        soc, soccfg = context.soc, context.soccfg

        setup_devices(
            cfg,
            context.devices,
            progress=True,
            cancel_signal=context.cancel_signal,
        )
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

        viewer = context.plots.liveplot_1d("measurement", "Length (us)", "Signal")
        signals_buffer = SignalBuffer(
            (len(lengths),),
            on_update=lambda data: viewer.update(lengths, rabi_signal2real(data)),
        )
        with Schedule(cfg, signals_buffer, stop=context.cancel_signal) as sched:
            cfg = sched.cfg
            modules = cfg.modules
            length_sweep = cfg.sweep.length
            modules.qub_pulse.set_param("length", sweep2param("length", length_sweep))

            _ = (
                sched.prog_builder(soc, soccfg)
                .add_reset("reset", modules.reset)
                .add_pulse("init_pulse", modules.init_pulse)
                .add_pulse("qubit_pulse", modules.qub_pulse)
                .add_readout("readout", modules.readout)
                .declare_sweep("length", length_sweep)
                .build_and_acquire()
            )
        return LenRabiResult(
            lengths=lengths,
            signals=signals_buffer.array,
        )

    def _run_for_arb(
        self,
        cfg: LenRabiCfg,
        *,
        context: RunContext,
    ) -> LenRabiResult:
        cfg = deepcopy(cfg)
        soc, soccfg = context.soc, context.soccfg

        setup_devices(
            cfg,
            context.devices,
            progress=True,
            cancel_signal=context.cancel_signal,
        )
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

        viewer = context.plots.liveplot_1d("measurement", "Length (us)", "Signal")
        length_values = lengths.tolist()
        signals_buffer = SignalBuffer(
            (rounds, len(lengths)),
            on_update=lambda data: viewer.update(
                lengths, rabi_signal2real(average_round(data))
            ),
        )
        with Schedule(_cfg, signals_buffer, stop=context.cancel_signal) as sched:
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
                    )
        return LenRabiResult(
            lengths=lengths,
            signals=average_round(signals_buffer.array),
        )

    def run(
        self,
        cfg: LenRabiCfg,
        *,
        context: RunContext,
    ) -> LenRabiResult:
        modules = cfg.modules
        qub_waveform = modules.qub_pulse.waveform

        if qub_waveform.style in ["const", "flat_top"]:
            # use hard sweep for flat top pulse
            return self._run_for_flat(cfg, context=context)
        # use soft sweep for arb pulse
        return self._run_for_arb(cfg, context=context)

    def analyze(
        self,
        source: RunRecord[LenRabiCfg, LenRabiResult],
        options: LenRabiAnalyzeOptions,
        *,
        plots: Plots,
    ) -> LenRabiAnalysis:
        result = source.result

        lens, signals = result.lengths, result.signals

        real_signals = rabi_signal2real(signals)

        nan_mask = np.isnan(real_signals)
        if np.all(nan_mask):
            raise ValueError("All data are NaN!")

        lens = lens[~nan_mask]
        real_signals = real_signals[~nan_mask]

        (
            pi_len,
            pi_len_err,
            pi2_len,
            pi2_len_err,
            freq,
            freq_err,
            y_fit,
            (pOpt, pCov),
        ) = fit_rabi(
            # Signed amplitude covers both zero-drive extrema when phase is fixed.
            lens,
            real_signals,
            decay=options.decay,
            init_phase=None if options.fit_phase else 0.0,
        )

        names = ("y0", "yscale", "freq", "phase") + (
            ("decay_time",) if options.decay else ()
        )
        quality = compute_fit_quality(
            real_signals,
            y_fit,
            {name: float(value) for name, value in zip(names, pOpt, strict=True)},
            pCov,
        )

        fig, ax = plots.subplots("fit", figsize=config.figsize)

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
        return LenRabiAnalysis(
            pi_len,
            pi_len_err,
            pi2_len,
            pi2_len_err,
            freq,
            freq_err,
            fit_quality={"fit": quality},
        )
