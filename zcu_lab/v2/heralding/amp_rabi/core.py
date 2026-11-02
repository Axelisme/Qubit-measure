"""Single-qubit amplitude Rabi with ground-herald post-selection."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar

import numpy as np
from numpy.typing import NDArray
from zcu_tools.analysis.fitting import FitQuality, fit_rabi
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import (
    Experiment,
    GroupedAxesSpec,
    GroupedLoadData,
    VariableAxisSpec,
    VariableSpec,
    VariableZSpec,
    config,
)
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime.run_acquire import RunOptions
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    IQ,
    Classify,
    Pulse,
    Reset,
    SweepCfg,
    TimelineBuilder,
    shot_sums,
    split_sums,
)
from zcu_tools.program.v2.modules import PulseReadout
from zcu_tools.utils.process import rotate2real

from zcu_lab.v2.support import live_line
from zcu_lab.v2.twotone.rabi.amp_rabi.core import AmpRabiAnalyzeOptions


@dataclass(frozen=True)
class HeraldingAmpRabiResult:
    """amps is the sweep grid (native gain); sums has shape (points, classes, 3).

    Herald states follow the saved classifier's state_names order and length.
    Components are count/sum I/sum Q.
    Unmeasured sums are NaN. Conditional means at zero count are NaN.
    """

    amps: NDArray[np.float64]
    sums: NDArray[np.float64]


@dataclass(frozen=True)
class HeraldingAmpRabiAnalysis:
    """Herald-g amplitude calibration, not gate fidelity.

    Attributes:
        pi_amp: Fitted pi-pulse gain in native gain units (a.u.); never None.
        pi_amp_err: Pi-gain standard uncertainty in a.u.; never None.
        pi2_amp: Fitted pi/2-pulse gain in a.u.; never None.
        pi2_amp_err: Pi/2-gain standard uncertainty in a.u.; never None.
        fit_quality: Fitter diagnostics keyed by "fit", including dimensionless
            residual and relative-error metrics and fixed-phase diagnostics;
            None when not supplied. analyze always supplies them.
    """

    pi_amp: float
    pi_amp_err: float
    pi2_amp: float
    pi2_amp_err: float
    fit_quality: Mapping[str, FitQuality] | None = None


class HeraldingAmpRabiCfg(ConfigBase):
    """Fixed-duration drive gain sweep with one calibrated pulse readout."""

    reset: Annotated[Reset.Cfg | None, FieldMeta(label="Reset")] = None
    init_pulse: Annotated[Pulse.Cfg | None, FieldMeta(label="Init Pulse")] = None
    qub_pulse: Annotated[Pulse.Cfg, FieldMeta(label="Rabi Pulse")]
    readout: Annotated[PulseReadout.Cfg, FieldMeta(label="Readout")]
    gain: Annotated[SweepCfg, FieldMeta(label="Gain", unit="a.u.")]
    reps: Annotated[int, FieldMeta(label="Reps")] = 1000
    rounds: Annotated[int, FieldMeta(label="Rounds")] = 10
    relax_delay: Annotated[
        float, FieldMeta(label="Relax delay", unit="us", decimals=3)
    ] = 1.0


def restore_result(
    data: GroupedLoadData[HeraldingAmpRabiCfg],
) -> HeraldingAmpRabiResult:
    """Restore per-class shot sums; old files fail naming the missing sums variable."""
    sums = data.variable("sums")
    return HeraldingAmpRabiResult(
        amps=sums.axes[2].astype(np.float64),
        sums=sums.z.astype(np.float64),
    )


class HeraldingAmpRabiExp(Experiment[HeraldingAmpRabiResult, HeraldingAmpRabiCfg]):
    """Herald ground, sweep drive gain, and store per-class count/IQ sums.

    Store all herald classes in classifier state_names order.
    Live plots and fits use only the herald-g conditional mean IQ.
    """

    Options: ClassVar[type[AmpRabiAnalyzeOptions]] = AmpRabiAnalyzeOptions
    AXES_SPEC: ClassVar[
        GroupedAxesSpec[HeraldingAmpRabiResult, HeraldingAmpRabiCfg]
    ] = GroupedAxesSpec(
        variables=(
            VariableSpec(
                variable="sums",
                axes=(
                    VariableAxisSpec.generated_arange(label="components", unit=""),
                    VariableAxisSpec.generated_arange(label="herald_states", unit=""),
                    VariableAxisSpec(label="Gain", unit="", field_name="amps"),
                ),
                z=VariableZSpec(
                    field_name="sums", label="Shot sums", unit="", dtype=np.float64
                ),
            ),
        ),
        result_type=HeraldingAmpRabiResult,
        cfg_type=HeraldingAmpRabiCfg,
        tag="heralding/amp_rabi",
        result_builder=restore_result,
    )

    def run(
        self, cfg: HeraldingAmpRabiCfg, *, context: RunContext
    ) -> HeraldingAmpRabiResult:
        """Acquire (Ngain, classes, 3) shot sums versus hardware-quantized gain.

        After optional reset, herald, apply optional init and fixed-duration
        gain-swept drive, then read signal. Both readouts share cfg.readout and
        its classifier. Herald states follow classifier state_names; components are
        count/sum I/sum Q. Rounds add sums. Live and analyze use herald-g
        conditional IQ; zero-count means are NaN. Unmeasured sums remain NaN.
        cfg is not mutated. Pulse duration uses us, gain uses native units.
        Heralding itself adds no active reset.
        """
        if cfg.readout.classifier is None:
            raise ValueError("Heralding requires a readout classifier")
        num_states = len(cfg.readout.classifier.state_names)
        cfg = deepcopy(cfg)
        soccfg = context.soccfg
        gains = sweep2array(
            cfg.gain, "gain", {"soccfg": soccfg, "gen_ch": cfg.qub_pulse.ch}
        )
        fig, ax = context.plots.subplots(num="measurement")
        ax.set(xlabel="Pulse gain", ylabel="IQ magnitude (a.u.)")
        viewer = live_line(ax)

        @context.plots.on_plot_thread
        def update(data):
            return viewer.set(gains, np.abs(split_sums(data)[1][..., 0]))

        fig.show()
        buffer = SignalBuffer(
            (len(gains), num_states, 3),
            dtype=np.float64,
            on_update=update,
        )

        with Schedule(buffer, scope=context.scope) as sched:
            builder = TimelineBuilder(
                soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay
            )
            gain = builder.add_sweep("gain", cfg.gain)
            if cfg.reset is not None:
                builder.emit(Reset("reset", cfg.reset))
            builder.emit(PulseReadout("herald", cfg.readout))
            builder.emit(Pulse("init_pulse", cfg.init_pulse))
            builder.emit(Pulse("qub_pulse", cfg.qub_pulse, gain=gain))
            builder.emit(PulseReadout("signal", cfg.readout))
            prog = builder.build()
            sched.run_acquire_summed(
                prog,
                context.soc,
                rounds=cfg.rounds,
                processor=shot_sums(prog, [Classify("herald"), IQ("signal")]),
                options=RunOptions(
                    converter=np.copy,
                ),
            )

        return HeraldingAmpRabiResult(gains, buffer.array)

    def analyze(
        self,
        source: RunRecord[HeraldingAmpRabiCfg, HeraldingAmpRabiResult],
        options: AmpRabiAnalyzeOptions,
        *,
        plots: Plots,
    ) -> HeraldingAmpRabiAnalysis:
        """Fit herald-g IQ and return pi/pi2 gains, errors and fit quality.

        Skip leading points on the original gain grid, discard nonfinite gain
        and IQ, then rotate IQ onto a real contrast axis. The point nearest zero
        gain fixes the nondecaying Rabi phase to 0 or 180 degrees. Publish a
        fit plot with pi/pi2 markers and uncertainty spans, and return the
        fitter's quality diagnostics. Pass fraction is
        diagnostic, not a fit weight. Saved cfg is not required. Negative skip
        is rejected by Options; empty or insufficient traces propagate errors.
        """
        gains = source.result.amps[options.skip :]
        signals = split_sums(source.result.sums)[1][..., 0][options.skip :]
        finite = np.isfinite(gains) & np.isfinite(signals)
        gains, signals = gains[finite], signals[finite]
        real_signals = rotate2real(signals).real

        zero_signal = real_signals[np.argmin(np.abs(gains))]
        if zero_signal > 0.5 * (np.max(real_signals) + np.min(real_signals)):
            init_phase = 0.0
        else:
            init_phase = 180.0

        fit = fit_rabi(gains, real_signals, decay=False, init_phase=init_phase)
        pi_amp, pi_amp_err, pi2_amp, pi2_amp_err, y_fit = (
            fit.pi_x,
            fit.pi_x_err,
            fit.pi2_x,
            fit.pi2_x_err,
            fit.curve,
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
        return HeraldingAmpRabiAnalysis(
            pi_amp, pi_amp_err, pi2_amp, pi2_amp_err, fit_quality={"fit": fit.quality}
        )
