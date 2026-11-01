from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar

import numpy as np
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray
from pydantic import Field
from pydantic.dataclasses import dataclass as validated_dataclass
from zcu_tools.analysis.fitting import FitQuality, compute_fit_quality, fit_rabi
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment, config
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime import RunOptions
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.experiment.v2.utils.snr import snr_checker
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import GainSweep, Pulse, Readout, Reset, TimelineBuilder
from zcu_tools.utils.process import rotate2real

from zcu_lab.v2.support import live_line


@dataclass(frozen=True)
class AmpRabiResult:
    """Drive gains (amps) and mean complex readout IQ, both shaped (Ngain,)."""

    amps: NDArray[np.float64]
    signals: NDArray[np.complex128]


@validated_dataclass(frozen=True)
class AmpRabiAnalyzeOptions:
    skip: Annotated[int, FieldMeta(label="Skip points"), Field(ge=0)] = 0


@dataclass(frozen=True)
class AmpRabiAnalysis:
    pi_amp: float
    pi_amp_err: float
    pi2_amp: float
    pi2_amp_err: float
    fit_quality: Mapping[str, FitQuality] | None = None


def rabi_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return rotate2real(signals).real


class AmpRabiCfg(ConfigBase):
    reset: Annotated[Reset.Cfg | None, FieldMeta(label="Reset")] = None
    init_pulse: Annotated[
        Pulse.Cfg | None,
        FieldMeta(
            label="Init Pulse",
            description="Fixed preparation pulse before each gain-swept Rabi pulse; None skips preparation.",
        ),
    ] = None
    qub_pulse: Annotated[Pulse.Cfg, FieldMeta(label="Rabi Pulse")]
    readout: Annotated[Readout.Cfg, FieldMeta(label="Readout")]
    gain: Annotated[GainSweep, FieldMeta(label="Gain", unit="a.u.")]
    reps: Annotated[
        int, FieldMeta(label="Reps", description="Number of repetitions per round.")
    ] = 1
    rounds: Annotated[
        int,
        FieldMeta(label="Rounds", description="Number of software acquisition rounds."),
    ] = 1
    acquire_retry: Annotated[
        int,
        FieldMeta(
            label="Acquire retry",
            description="Retries after transient acquisition failure.",
        ),
    ] = Field(default=0, ge=0)
    earlystop_snr: Annotated[
        float | None,
        FieldMeta(
            label="Early-stop SNR",
            description="Stop after a completed round reaches this SNR; None disables.",
        ),
    ] = Field(default=None, gt=0)
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relax delay",
            unit="us",
            decimals=3,
            description="Relaxation delay after each repetition.",
        ),
    ] = 1.0


class AmpRabiExp(Experiment[AmpRabiResult, AmpRabiCfg]):
    Options: ClassVar[type[AmpRabiAnalyzeOptions]] = AmpRabiAnalyzeOptions

    AXES_SPEC: ClassVar[AxesSpec[AmpRabiResult, AmpRabiCfg]] = AxesSpec(
        axes=(AxesSpec.Axis("amps", label="Gain", unit=""),),
        z=AxesSpec.Z("signals", label="Signal", unit="a.u."),
        result_type=AmpRabiResult,
        cfg_type=AmpRabiCfg,
        cfg_schema_version="3.0",
        tag="twotone/ge/rabi_gain",
    )

    def run(
        self,
        cfg: AmpRabiCfg,
        *,
        context: RunContext,
    ) -> AmpRabiResult:
        """Measure Rabi oscillations versus qubit drive gain at fixed duration.

        Reset, apply the optional init pulse, play the gain-swept qubit
        pulse, then read out. Return gains and mean complex IQ; subsequent
        analysis uses the oscillation period to infer pi and pi/2 gains.
        acquire_retry retries transient acquisition failures; earlystop_snr stops
        averaging at completed-round boundaries. None disables SNR early stop.
        """
        # Prepare the hardware gain grid and display the mean IQ projection.
        cfg = deepcopy(cfg)
        soc, soccfg = context.soc, context.soccfg

        gains = sweep2array(
            cfg.gain,
            "gain",
            {"soccfg": soccfg, "gen_ch": cfg.qub_pulse.ch},
        )

        fig, ax = context.plots.subplots(num="measurement")
        ax.set(xlabel="Pulse gain", ylabel="Amplitude")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.grid()
        viewer = live_line(ax)

        @context.plots.on_plot_thread
        def update(data):
            return viewer.set(gains, rabi_signal2real(data))

        fig.show()
        signals_buffer = SignalBuffer(
            (len(gains),),
            on_update=update,
        )

        # Keep pulse duration fixed while the FPGA sweeps its rotation amplitude.
        with Schedule(signals_buffer, scope=context.scope) as sched:
            builder = TimelineBuilder(
                soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay
            )
            gain_sweep = cfg.gain
            gain_param = builder.add_sweep("gain", gain_sweep)

            if cfg.reset is not None:
                builder.emit(cfg.reset.build("reset"))
            builder.emit(Pulse("init_pulse", cfg.init_pulse))
            builder.emit(Pulse("qub_pulse", cfg.qub_pulse, gain=gain_param))
            builder.emit(Readout("readout", cfg.readout))
            prog = builder.build()
            sched.run_acquire(
                prog,
                soc,
                rounds=cfg.rounds,
                options=RunOptions(
                    retry=cfg.acquire_retry,
                    stop_condition=snr_checker(
                        signals_buffer.at(), cfg.earlystop_snr, rabi_signal2real
                    ),
                ),
            )

        return AmpRabiResult(amps=gains, signals=signals_buffer.array)

    def analyze(
        self,
        source: RunRecord[AmpRabiCfg, AmpRabiResult],
        options: AmpRabiAnalyzeOptions,
        *,
        plots: Plots,
    ) -> AmpRabiAnalysis:
        """Fit a nondecaying Rabi oscillation and return pi/pi2 gains and errors.

        Skip the first ``options.skip`` points and project IQ onto a real
        contrast axis. The point nearest zero gain fixes the phase to 0 or
        180 degrees. Publish ``fit`` and report fit quality; these are pulse
        amplitude calibrations, not gate fidelities. Source cfg may be None.
        """
        result = source.result

        gains, signals = result.amps, result.signals
        gains = gains[options.skip :]
        signals = signals[options.skip :]

        real_signals = rabi_signal2real(signals)

        zero_signal = real_signals[np.argmin(np.abs(gains))]
        if zero_signal > 0.5 * (np.max(real_signals) + np.min(real_signals)):
            init_phase = 0.0
        else:
            init_phase = 180.0

        fit = fit_rabi(gains, real_signals, decay=False, init_phase=init_phase)
        pi_amp, pi_amp_err, pi2_amp, pi2_amp_err, y_fit, pOpt, pCov = (
            fit.pi_x,
            fit.pi_x_err,
            fit.pi2_x,
            fit.pi2_x_err,
            fit.curve,
            fit.params,
            fit.cov,
        )

        names = ("y0", "yscale", "freq", "phase")
        quality = compute_fit_quality(
            real_signals,
            y_fit,
            {name: float(value) for name, value in zip(names, pOpt, strict=True)},
            pCov,
        )

        fig, ax = plots.subplots(num="fit", figsize=config.figsize)

        ax.plot(gains, real_signals, label="meas", ls="-", marker="o", markersize=3)
        ax.plot(gains, y_fit, label="fit")
        ax.axvline(
            pi_amp, ls="--", c="red", label=f"pi = {pi_amp:.3g} ± {pi_amp_err:.2g}"
        )
        ax.axvspan(pi_amp - pi_amp_err, pi_amp + pi_amp_err, color="red", alpha=0.2)
        ax.axvline(
            pi2_amp,
            ls="--",
            c="red",
            label=f"pi/2 = {pi2_amp:.3g} ± {pi2_amp_err:.2g}",
        )
        ax.axvspan(pi2_amp - pi2_amp_err, pi2_amp + pi2_amp_err, color="red", alpha=0.2)
        ax.set_xlabel("Pulse gain (a.u.)")
        ax.set_ylabel("Signal Real (a.u.)")
        ax.legend(loc=4)
        ax.grid(True)

        fig.tight_layout()

        # fit_rabi computes the per-gain fit uncertainties; surface them so the GUI
        # summary carries the gain errors (the figure labels already show them).
        return AmpRabiAnalysis(
            pi_amp, pi_amp_err, pi2_amp, pi2_amp_err, fit_quality={"fit": quality}
        )
