"""Single-qubit AmpRabi from self-calibrated herald/signal second moments."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar, Literal

import numpy as np
from numpy.typing import NDArray
from zcu_tools.analysis.fitting import FitQuality, fit_rabi
from zcu_tools.analysis.herald_transfer import selfcal_transfer
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
    Pulse,
    Reset,
    SweepCfg,
    TimelineBuilder,
    shot_sums,
)
from zcu_tools.program.v2.modules import PulseReadout

from zcu_lab.v2.heralding._live import live_transfer
from zcu_lab.v2.twotone.rabi.amp_rabi.core import AmpRabiAnalyzeOptions


@dataclass(frozen=True)
class HeraldingAmpRabiAnalyzeOptions(AmpRabiAnalyzeOptions):
    """Fit C using two or three states; second-order IQ identifies at most three."""

    n_states: Annotated[Literal[2, 3], FieldMeta(label="Number of states")] = 2


@dataclass(frozen=True)
class HeraldingAmpRabiResult:
    """amps is the sweep grid (native gain); sums is float64 (N, 6, 6).

    Herald and signal axes use (1, I, Q, I², IQ, Q²); sums[:, 0, 0] is
    the shot count. Unmeasured sums are NaN.
    """

    amps: NDArray[np.float64]
    sums: NDArray[np.float64]


@dataclass(frozen=True)
class HeraldingAmpRabiAnalysis:
    """C-based amplitude calibration, not gate fidelity.

    Attributes:
        pi_amp: Fitted pi-pulse gain in native gain units (a.u.); never None.
        pi_amp_err: Pi-gain standard uncertainty in a.u.; never None.
        pi2_amp: Fitted pi/2-pulse gain in a.u.; never None.
        pi2_amp_err: Pi/2-gain standard uncertainty in a.u.; never None.
        fit_quality: Fitter diagnostics keyed by "fit", including dimensionless
            residual and relative-error metrics and fixed-phase diagnostics;
            None when not supplied. analyze always supplies them.
    pi_hat is the post-herald population tuple in descending order; herald
    decay can reduce its excited population relative to the initial thermal state.
    singular_values contains the leading n_states+1 singular values normalized
    by the largest. Neither field contains per-point values.
    """

    pi_amp: float
    pi_amp_err: float
    pi2_amp: float
    pi2_amp_err: float
    pi_hat: tuple[float, ...]
    singular_values: tuple[float, ...]
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
    """Restore second-order sums; incompatible axis labels fail native loading."""
    sums = data.variable("sums")
    return HeraldingAmpRabiResult(
        amps=sums.axes[2].astype(np.float64),
        sums=sums.z.astype(np.float64),
    )


class HeraldingAmpRabiExp(Experiment[HeraldingAmpRabiResult, HeraldingAmpRabiCfg]):
    """Measure shared-readout herald/signal moments without a classifier.

    Optional reset, herald readout, optional init, gain-swept drive, signal.
    Live plots and fits use only state-change probability C = 1 - tr J.
    K[i,j] = P(i -> j) is a diagnostic, not the fitted observable.
    """

    Options: ClassVar[type[HeraldingAmpRabiAnalyzeOptions]] = (
        HeraldingAmpRabiAnalyzeOptions
    )
    AXES_SPEC: ClassVar[
        GroupedAxesSpec[HeraldingAmpRabiResult, HeraldingAmpRabiCfg]
    ] = GroupedAxesSpec(
        variables=(
            VariableSpec(
                variable="sums",
                axes=(
                    VariableAxisSpec(
                        label="signal_features", unit="", values=tuple(range(6))
                    ),
                    VariableAxisSpec.generated_arange(label="herald_features", unit=""),
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
        """Acquire (N, 6, 6) moment sums without classifier calibration.

        Optional reset, herald readout, optional init, gain-swept drive, signal.
        Both readouts use cfg.readout. Rounds add raw sums. cfg is copied.
        Live C uses two-state self-calibration with warm starts. Before its
        first success show Calibrating; identification failures keep the last
        plot and do not stop acquisition. Unmeasured sums remain NaN.
        Pulse times use us. Heralding adds no active reset or feedback.
        """
        cfg = deepcopy(cfg)
        soccfg = context.soccfg
        gains = sweep2array(
            cfg.gain, "gain", {"soccfg": soccfg, "gen_ch": cfg.qub_pulse.ch}
        )
        fig, ax = context.plots.subplots(num="measurement")
        ax.set(xlabel="Pulse gain", ylabel="IQ magnitude (a.u.)")
        update = live_transfer(context.plots, ax, gains)

        fig.show()
        buffer = SignalBuffer(
            (len(gains), 6, 6),
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
                processor=shot_sums(prog, [IQ("herald"), IQ("signal")]),
                options=RunOptions(
                    converter=np.copy,
                ),
            )

        return HeraldingAmpRabiResult(gains, buffer.array)

    def analyze(
        self,
        source: RunRecord[HeraldingAmpRabiCfg, HeraldingAmpRabiResult],
        options: HeraldingAmpRabiAnalyzeOptions,
        *,
        plots: Plots,
    ) -> HeraldingAmpRabiAnalysis:
        """Fully self-calibrate all sums and fit only C = 1 - tr J.

        Skip options.skip leading gain points after calibration. Fit a nondecaying
        cosine, fixing phase to 0 or 180 degrees from C nearest zero gain.
        Return pi/pi2 gains and their standard errors in native gain units.
        options.n_states selects 2 or 3, the second-order identifiability limit.
        Three collinear centers fail. Readout tails can bias the calibration.
        pi_hat describes populations entering the operation after herald;
        herald-readout decay can lower it relative to the initial thermal state.
        Draw only C and the fit, with pi_hat in the legend. Return fit parameters,
        pi_hat and normalized singular values, without pointwise arrays.
        Identification and fitting failures propagate; saved cfg is not required.
        """
        gains = source.result.amps[options.skip :]
        transfer = selfcal_transfer(source.result.sums, n_states=options.n_states)
        signals = transfer.change_probability[options.skip :]
        finite = np.isfinite(gains) & np.isfinite(signals)
        gains, signals = gains[finite], signals[finite]
        real_signals = signals

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

        ax.plot(
            gains,
            real_signals,
            label=f"C, π̂={transfer.populations.round(3).tolist()}",
            ls="-",
            marker="o",
            markersize=3,
        )
        ax.plot(gains, y_fit, label="fit")
        ax.set_xlabel("Pulse gain (a.u.)")
        ax.set_ylabel("Change probability C")
        ax.legend(loc=4)
        ax.grid(True)

        fig.tight_layout()

        # fit_rabi computes the per-gain fit uncertainties; surface them so the GUI
        # summary carries the gain errors (the figure labels already show them).
        return HeraldingAmpRabiAnalysis(
            pi_amp,
            pi_amp_err,
            pi2_amp,
            pi2_amp_err,
            pi_hat=tuple(float(p) for p in transfer.populations),
            singular_values=tuple(float(s) for s in transfer.singular_values),
            fit_quality={"fit": fit.quality},
        )
