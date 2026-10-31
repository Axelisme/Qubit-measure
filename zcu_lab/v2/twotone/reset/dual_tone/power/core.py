from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar

import numpy as np
from matplotlib.artist import Artist
from numpy.typing import NDArray
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment, config
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    GainSweep,
    Pulse,
    Readout,
    Reset,
    TimelineBuilder,
    TwoPulseReset,
)
from zcu_tools.utils.process import SmoothingOptions, SmoothMethod, smooth_signal_nd

from zcu_lab.v2.support import live_heatmap


@dataclass(frozen=True)
class PowerResult:
    """gain in a.u. within [-1, 1] axes and mean complex IQ (gain1, gain2)."""

    gains1: NDArray[np.float64]
    gains2: NDArray[np.float64]
    signals: NDArray[np.complex128]


class PowerCfg(ConfigBase):
    reset: Annotated[Reset.Cfg | None, FieldMeta(label="Reset")] = None
    init_pulse: Annotated[Pulse.Cfg | None, FieldMeta(label="Init pulse")] = None
    tested_reset: Annotated[TwoPulseReset.Cfg, FieldMeta(label="Tested reset")]
    readout: Annotated[Readout.Cfg, FieldMeta(label="Readout")]
    gain1: Annotated[GainSweep, FieldMeta(label="Gain1", unit="a.u.")]
    gain2: Annotated[GainSweep, FieldMeta(label="Gain2", unit="a.u.")]

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


@dataclass(frozen=True)
class PowerAnalyzeOptions:
    smooth: float = 1.0
    smooth_method: SmoothMethod = "wavelet"
    wavelet: str = "sym4"
    wavelet_level: int = 0
    xname: str | None = None
    yname: str | None = None


@dataclass(frozen=True)
class PowerAnalysis:
    gain1: float
    gain2: float


class PowerExp(Experiment[PowerResult, PowerCfg]):
    Options: ClassVar[type[PowerAnalyzeOptions]] = PowerAnalyzeOptions

    # Both axes are gains in a.u. -> scale=IDENTITY (default)
    AXES_SPEC: ClassVar[AxesSpec[PowerResult, PowerCfg]] = AxesSpec(
        axes=(
            AxesSpec.Axis("gains2", label="Power2", unit="a.u."),  # inner
            AxesSpec.Axis("gains1", label="Power1", unit="a.u."),  # outer
        ),
        z=AxesSpec.Z("signals", label="Signal", unit="a.u."),
        result_type=PowerResult,
        cfg_type=PowerCfg,
        cfg_schema_version="3.0",
        tag="twotone/reset/dual_tone/power",
    )

    def run(self, cfg: PowerCfg, *, context: RunContext) -> PowerResult:
        """Measure the dual-tone reset response across both pulse gains.

        Apply optional reset and initialization, start the two gain-swept
        reset pulses together, then read out after the longer pulse finishes.
        Return gains in a.u. within [-1, 1] and mean complex IQ shaped (Ngain1, Ngain2).
        The display uses the minimum-gain corner as background; this contrast
        is not a calibrated reset fidelity.
        """
        # Prepare each reset pulse's gain grid on its own generator channel.
        cfg = deepcopy(cfg)
        soc, soccfg = context.soc, context.soccfg

        reset_cfg = cfg.tested_reset
        gains1 = sweep2array(
            cfg.gain1,
            "gain",
            {"soccfg": soccfg, "gen_ch": reset_cfg.pulse1_cfg.ch},
        )
        gains2 = sweep2array(
            cfg.gain2,
            "gain",
            {"soccfg": soccfg, "gen_ch": reset_cfg.pulse2_cfg.ch},
        )

        def dual_reset_gain_signal2real(
            signals: NDArray[np.complex128],
        ) -> NDArray[np.float64]:
            # Choose reference point based on sweep direction (use minimum power point)
            ref_i = 0 if gains1[0] < gains1[-1] else -1
            ref_j = 0 if gains2[0] < gains2[-1] else -1
            return np.abs(signals - signals[ref_i, ref_j])

        # Display contrast relative to the minimum-gain corner in either scan direction.
        plots = context.plots
        fig, ax = plots.subplots(num="measurement")
        ax.set(xlabel="Gain1 (a.u.)", ylabel="Gain2 (a.u.)")
        viewer = live_heatmap(ax)

        @plots.on_plot_thread
        def update(data: NDArray[np.complex128]) -> tuple[Artist, ...]:
            return viewer.set(gains1, gains2, dual_reset_gain_signal2real(data))

        fig.show()
        signals_buffer = SignalBuffer((len(gains1), len(gains2)), on_update=update)

        # Sweep both reset gains in hardware and collect mean IQ.
        with Schedule(None, signals_buffer, scope=context.scope) as sched:
            builder = TimelineBuilder(
                soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay
            )
            gain1 = builder.add_sweep("gain1", cfg.gain1)
            gain2 = builder.add_sweep("gain2", cfg.gain2)
            builder.emit(Reset("reset", cfg.reset))
            builder.emit(Pulse("init_pulse", cfg.init_pulse))
            builder.emit(
                TwoPulseReset(
                    "tested_reset", cfg.tested_reset, gain1=gain1, gain2=gain2
                )
            )
            builder.emit(Readout("readout", cfg.readout))
            sched.run_acquire(builder.build(), soc, rounds=cfg.rounds)

        return PowerResult(gains1, gains2, signals_buffer.array)

    def analyze(
        self,
        source: RunRecord[PowerCfg, PowerResult],
        options: PowerAnalyzeOptions,
        *,
        plots: Plots,
    ) -> PowerAnalysis:
        """Select reset gains from the smoothed, background-subtracted IQ map.

        options controls smoothing and optional axis names. Use the
        minimum-gain corner as background, then choose a peak or dip per axis
        according to the first corner's contrast relative to the map mean.
        Return gain1/gain2 in a.u. within [-1, 1] and publish fit; source.cfg may be None.
        """
        result = source.result

        gains1, gains2, signals = result.gains1, result.gains2, result.signals

        # Apply smoothing for peak finding
        signals_smooth = smooth_signal_nd(
            signals,
            options=SmoothingOptions(
                method=options.smooth_method,
                sigma=options.smooth,
                wavelet=options.wavelet,
                wavelet_level=options.wavelet_level,
            ),
            axes=(0, 1),
        )

        ref_i = 0 if gains1[0] < gains1[-1] else -1
        ref_j = 0 if gains2[0] < gains2[-1] else -1
        amp2D = np.abs(signals_smooth - signals_smooth[ref_i, ref_j])

        # Determine if we should look for max or min
        if amp2D[0, 0] < np.mean(amp2D):
            gain1_opt = gains1[np.argmax(np.max(amp2D, axis=1))]
            gain2_opt = gains2[np.argmax(np.max(amp2D, axis=0))]
        else:
            gain1_opt = gains1[np.argmin(np.min(amp2D, axis=1))]
            gain2_opt = gains2[np.argmin(np.min(amp2D, axis=0))]
            amp2D = np.mean(amp2D) - amp2D

        fig, ax = plots.subplots("fit", figsize=config.figsize)

        ax.imshow(
            amp2D.T,
            aspect="auto",
            origin="lower",
            interpolation="none",
            extent=(gains1[0], gains1[-1], gains2[0], gains2[-1]),
        )
        peak_label = f"({gain1_opt:.1f}, {gain2_opt:.1f}) a.u."
        ax.scatter(gain1_opt, gain2_opt, color="r", s=40, marker="*", label=peak_label)
        if options.xname is not None:
            ax.set_xlabel(f"{options.xname} gain (a.u.)", fontsize=14)
        if options.yname is not None:
            ax.set_ylabel(f"{options.yname} gain (a.u.)", fontsize=14)
        ax.legend(fontsize="x-large")
        ax.tick_params(axis="both", which="major", labelsize=12)

        fig.tight_layout()

        return PowerAnalysis(gain1=float(gain1_opt), gain2=float(gain2_opt))
