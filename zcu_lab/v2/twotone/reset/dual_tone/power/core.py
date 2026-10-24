from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar

import numpy as np
from numpy.typing import NDArray
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment, config
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    Pulse,
    Readout,
    Reset,
    SweepCfg,
    TwoPulseReset,
    sweep2param,
)
from zcu_tools.utils.process import SmoothingOptions, SmoothMethod, smooth_signal_nd

from zcu_lab.v2._support import live_heatmap


@dataclass(frozen=True)
class PowerResult:
    """Normalized gain axes and mean complex IQ (gain1, gain2)."""

    gains1: NDArray[np.float64]
    gains2: NDArray[np.float64]
    signals: NDArray[np.complex128]


class PowerCfg(ConfigBase):
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
    init_pulse: Annotated[Pulse.Cfg | None, FieldMeta(label="Init pulse")] = None
    tested_reset: Annotated[TwoPulseReset.Cfg, FieldMeta(label="Tested reset")]
    readout: Annotated[Readout.Cfg, FieldMeta(label="Readout")]
    gain1: Annotated[SweepCfg, FieldMeta(label="Gain1", unit="a.u.")]
    gain2: Annotated[SweepCfg, FieldMeta(label="Gain2", unit="a.u.")]


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
            AxesSpec.Axis("gains2", "Power2", "a.u."),  # inner
            AxesSpec.Axis("gains1", "Power1", "a.u."),  # outer
        ),
        z=AxesSpec.Z("signals", "Signal", "a.u."),
        result_type=PowerResult,
        cfg_type=PowerCfg,
        cfg_schema_version="3.0",
        tag="twotone/reset/dual_tone/power",
    )

    def run(self, cfg: PowerCfg, *, context: RunContext) -> PowerResult:
        """Measure the dual-tone reset response across both pulse gains.

        Apply optional reset and initialization, start the two gain-swept
        reset pulses together, then read out after the longer pulse finishes.
        Return normalized gains and mean complex IQ shaped (Ngain1, Ngain2).
        The display uses the minimum-gain corner as background; this contrast
        is not a calibrated reset fidelity.
        """
        # Prepare each reset pulse's gain grid on its own generator channel.
        cfg = deepcopy(cfg)
        soc, soccfg = context.soc, context.soccfg

        modules = cfg

        reset_cfg = modules.tested_reset
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
        viewer = live_heatmap(
            context.plots, "measurement", "Gain1 (a.u.)", "Gain2 (a.u.)"
        )
        signals_buffer = SignalBuffer(
            (len(gains1), len(gains2)),
            on_update=lambda data: viewer.update(
                gains1, gains2, dual_reset_gain_signal2real(data)
            ),
        )

        # Sweep both reset gains in hardware and collect mean IQ.
        with Schedule(cfg, signals_buffer, scope=context.scope) as sched:
            modules = sched.cfg
            modules.tested_reset.set_param(
                "gain1", sweep2param("gain1", sched.cfg.gain1)
            )
            modules.tested_reset.set_param(
                "gain2", sweep2param("gain2", sched.cfg.gain2)
            )
            _ = (
                sched.prog_builder(
                    soc, soccfg, reps=sched.cfg.reps, relax_delay=sched.cfg.relax_delay
                )
                .add(
                    Reset("reset", modules.reset),
                    Pulse("init_pulse", modules.init_pulse),
                    TwoPulseReset("tested_reset", modules.tested_reset),
                    Readout("readout", modules.readout),
                )
                .declare_sweep("gain1", sched.cfg.gain1)
                .declare_sweep("gain2", sched.cfg.gain2)
                .build_and_acquire(rounds=cfg.rounds)
            )

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
        Return normalized gain1/gain2 and publish fit; source.cfg may be None.
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
