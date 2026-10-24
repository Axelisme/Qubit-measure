from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar

import numpy as np
from numpy.typing import NDArray
from zcu_tools.analysis.fitting import FitQuality, compute_fit_quality, fit_rabi
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment, config
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting import live_line
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import Pulse, Readout, Reset, SweepCfg, sweep2param
from zcu_tools.utils.process import rotate2real


@dataclass(frozen=True)
class LenRabiResult:
    lengths: NDArray[np.float64]
    signals: NDArray[np.complex128]


@dataclass(frozen=True)
class LenRabiAnalyzeOptions:
    decay: Annotated[bool, FieldMeta(label="Fit decay envelope")] = True
    fit_phase: Annotated[bool, FieldMeta(label="Fit phase offset")] = False


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


class LenRabiCfg(ConfigBase):
    reps: Annotated[
        int, FieldMeta(label="Reps", description="Number of repetitions per round.")
    ] = 1
    rounds: Annotated[
        int,
        FieldMeta(label="Rounds", description="Number of software acquisition rounds."),
    ] = 1
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relax delay",
            unit="us",
            decimals=3,
            description="Relaxation delay after each repetition.",
        ),
    ] = 1.0
    reset: Annotated[Reset.Cfg | None, FieldMeta(label="Reset")] = None
    init_pulse: Annotated[Pulse.Cfg | None, FieldMeta(label="Init Pulse")] = None
    qub_pulse: Annotated[Pulse.Cfg, FieldMeta(label="Rabi Pulse")]
    readout: Annotated[Readout.Cfg, FieldMeta(label="Readout")]
    length: Annotated[SweepCfg, FieldMeta(label="Duration", unit="us")]


class LenRabiExp(Experiment[LenRabiResult, LenRabiCfg]):
    Options: ClassVar[type[LenRabiAnalyzeOptions]] = LenRabiAnalyzeOptions

    # lengths stored in seconds on disk (mem us) -> scale=1e-6; z complex
    AXES_SPEC: ClassVar[AxesSpec[LenRabiResult, LenRabiCfg]] = AxesSpec(
        axes=(AxesSpec.Axis("lengths", "Length", "s", 1e-6),),
        z=AxesSpec.Z("signals", "Signal", "a.u."),
        result_type=LenRabiResult,
        cfg_type=LenRabiCfg,
        cfg_schema_version="3.0",
        tag="twotone/ge/rabi_length",
    )

    def run(
        self,
        cfg: LenRabiCfg,
        *,
        context: RunContext,
    ) -> LenRabiResult:
        """Measure Rabi oscillations by sweeping drive length at fixed gain.

        Apply optional reset and initialization, then the length-swept qubit
        pulse and readout. Const/flat-top pulses sweep on the FPGA; other
        waveforms scan on the host and reuse each length's compiled program.
        Return quantized lengths in us and mean complex IQ at each length,
        retaining acquired data on cancellation and NaN for unmeasured points.
        Fit the oscillation with analyze() to obtain pi and pi/2 lengths.
        Reject native arb with ValueError before preparing coordinates or plots:
        its asset-derived length cannot be swept by this experiment.
        """
        if cfg.qub_pulse.waveform.style == "arb":
            raise ValueError(
                "Len Rabi cannot sweep an arb waveform's asset-derived length. "
                "Select a waveform with an editable length instead."
            )

        # 1. Prepare length coordinates without changing the caller's cfg.
        cfg = deepcopy(cfg)
        soc, soccfg = context.soc, context.soccfg
        modules = cfg
        lengths = sweep2array(
            cfg.length,
            "time",
            {"soccfg": soccfg, "gen_ch": modules.qub_pulse.ch},
        )

        if modules.qub_pulse.waveform.style in ["const", "flat_top"]:
            viewer = live_line(context.plots, "measurement", "Length (us)", "Signal")
            signals_buffer = SignalBuffer(
                (len(lengths),),
                on_update=lambda data: viewer.update(lengths, rabi_signal2real(data)),
            )
            with Schedule(cfg, signals_buffer, scope=context.scope) as sched:
                modules = sched.cfg
                length_sweep = sched.cfg.length
                modules.qub_pulse.set_param(
                    "length", sweep2param("length", length_sweep)
                )

                # 2. Sweep length on the FPGA: reset -> init -> drive -> readout.
                _ = (
                    sched.prog_builder(
                        soc,
                        soccfg,
                        reps=sched.cfg.reps,
                        relax_delay=sched.cfg.relax_delay,
                    )
                    .add_reset("reset", modules.reset)
                    .add_pulse("init_pulse", modules.init_pulse)
                    .add_pulse("qubit_pulse", modules.qub_pulse)
                    .add_readout("readout", modules.readout)
                    .declare_sweep("length", length_sweep)
                    .build_and_acquire(rounds=cfg.rounds)
                )

            # 3. The program acquisition supplies the completed-round mean IQ.
            return LenRabiResult(lengths=lengths, signals=signals_buffer.array)

        # 2. Other waveforms need a separate program for each distinct length.
        rounds = cfg.rounds
        cfg = cfg.model_copy(deep=True)
        cfg.rounds = 1  # Host repeats own the rounds; FPGA still owns reps.
        lengths = np.unique(lengths)
        programs: dict[float, Any] = {}

        def average_round(signals: NDArray[np.complex128]) -> NDArray[np.complex128]:
            _signals = np.asarray(signals)  # shape: (rounds, len(lengths))
            mask = np.any(~np.isnan(_signals), axis=0)
            mean_signals = np.full(_signals.shape[1], np.nan, dtype=np.complex128)
            mean_signals[mask] = np.nanmean(_signals[:, mask], axis=0)
            return mean_signals

        viewer = live_line(context.plots, "measurement", "Length (us)", "Signal")
        length_values = lengths.tolist()
        signals_buffer = SignalBuffer(
            (rounds, len(lengths)),
            on_update=lambda data: viewer.update(
                lengths, rabi_signal2real(average_round(data))
            ),
        )
        with Schedule(cfg, signals_buffer, scope=context.scope) as sched:
            for _, rep in sched.repeat("round", rounds):
                for length, step in rep.scan("length", length_values):
                    modules = step.cfg
                    modules.qub_pulse.set_param("length", length)

                    # 3. Reset -> init -> shaped drive -> readout at this length.
                    builder = (
                        step.prog_builder(
                            soc,
                            soccfg,
                            reps=step.cfg.reps,
                            relax_delay=step.cfg.relax_delay,
                        )
                        .add_reset("reset", modules.reset)
                        .add_pulse("init_pulse", modules.init_pulse)
                        .add_pulse("qubit_pulse", modules.qub_pulse)
                        .add_readout("readout", modules.readout)
                    )
                    length_key = float(length)
                    if length_key not in programs:
                        programs[length_key] = builder.build()
                    _ = builder.run_program(programs[length_key], rounds=cfg.rounds)

        # 4. Average available rounds per length, keeping partial scans and NaNs.
        return LenRabiResult(
            lengths=lengths,
            signals=average_round(signals_buffer.array),
        )

    def analyze(
        self,
        source: RunRecord[LenRabiCfg, LenRabiResult],
        options: LenRabiAnalyzeOptions,
        *,
        plots: Plots,
    ) -> LenRabiAnalysis:
        """Fit the measured IQ oscillation to derive pi and pi/2 lengths in us.

        Rotate source.result IQ to a real trace and discard unmeasured NaNs.
        options.decay enables a decay envelope; options.fit_phase frees the
        otherwise fixed zero phase (signed amplitude also covers 180 degrees).
        Publish the fit plot and return lengths, Rabi frequency in MHz, their
        uncertainties and fit quality. source.cfg may be None. All-NaN data
        raise ValueError; other invalid traces follow fit_rabi's failure rules.
        """
        # 1. Project mean IQ and keep only acquired length points.
        result = source.result

        lens, signals = result.lengths, result.signals

        real_signals = rabi_signal2real(signals)

        nan_mask = np.isnan(real_signals)
        if np.all(nan_mask):
            raise ValueError("All data are NaN!")

        lens = lens[~nan_mask]
        real_signals = real_signals[~nan_mask]

        # 2. Fit this length scan with the user's envelope and phase choices.
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

        # 3. Show the oscillation and the fitted calibration lengths.
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
