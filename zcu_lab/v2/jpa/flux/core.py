from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar, Literal

import numpy as np
from matplotlib.artist import Artist
from numpy.typing import NDArray
from pydantic import Field
from scipy.ndimage import gaussian_filter1d
from zcu_tools.cfg_model import ConfigBase, DeviceName, FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment, config
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime import RunOptions, Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.experiment.v2.utils.snr import snr_as_signal
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    IQ,
    Pulse,
    Readout,
    Reset,
    SweepCfg,
    TimelineBuilder,
    shot_sums,
)

from zcu_lab.v2.support import live_line
from zcu_lab.v2.support.device import set_flux


@dataclass(frozen=True)
class FluxResult:
    """JPA flux device values and moment-based GE SNR at each value."""

    fluxes: NDArray[np.float64]
    signals: NDArray[np.float64]


@dataclass(frozen=True)
class FluxAnalysis:
    best_flux: float


class FluxCfg(ConfigBase):
    reset: Annotated[
        Reset.Cfg | None,
        FieldMeta(
            label="Reset",
            description="Select a qubit reset before readout; leave empty to skip it.",
        ),
    ] = None
    pi_pulse: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Qubit π pulse",
            description="Select the calibrated g→e pulse for the excited preparation only.",
        ),
    ]
    readout: Annotated[
        Readout.Cfg,
        FieldMeta(
            label="Readout",
            description="Select the calibrated resonator pulse and IQ integration for both preparations.",
        ),
    ]
    jpa_flux_dev: Annotated[
        DeviceName | None,
        FieldMeta(
            label="JPA flux-bias source",
            description="Select the DC source that sets the JPA bias; leave empty to scan coordinates without changing it.",
        ),
    ] = None
    flux_unit: Annotated[
        Literal["A", "V", "native"],
        FieldMeta(
            label="JPA bias unit",
            description="Match the source unit: amps, volts, or its native coordinate; the sweep values are not converted.",
        ),
    ]
    jpa_flux: Annotated[
        SweepCfg,
        FieldMeta(
            label="JPA flux-bias value",
            description="Enter the bounds and step in JPA bias unit and the number of bias points.",
        ),
    ]
    skew_penalty: Annotated[
        float,
        FieldMeta(
            label="IQ shape penalty",
            unit="a.u.",
            decimals=3,
            description="Enter a nonnegative weight for unequal IQ widths and skew; 0 uses separation SNR alone.",
        ),
    ] = Field(default=0.0, ge=0.0)
    reps: Annotated[
        int,
        FieldMeta(
            label="Reps",
            unit="count",
            description="Enter the shots per preparation at each setting in one round.",
        ),
    ]
    rounds: Annotated[
        int,
        FieldMeta(
            label="Rounds",
            unit="count",
            description="Enter the number of acquisition rounds per setting.",
        ),
    ] = 1
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relaxation wait",
            unit="µs",
            decimals=3,
            description="Enter the wait after readout for relaxation before the next shot (µs).",
        ),
    ]


class FluxExp(Experiment[FluxResult, FluxCfg]):
    # jpa_flux stored as-is (a.u.) on disk -> scale=IDENTITY; signals are float64
    AXES_SPEC: ClassVar[AxesSpec[FluxResult, FluxCfg]] = AxesSpec(
        axes=(AxesSpec.Axis("fluxes", label="JPA Flux value", unit="a.u."),),
        z=AxesSpec.Z("signals", label="Signal", unit="a.u.", dtype=np.float64),
        result_type=FluxResult,
        cfg_type=FluxCfg,
        tag="jpa/flux",
        cfg_schema_version="3.0",
    )

    def run(self, cfg: FluxCfg, *, context: RunContext) -> FluxResult:
        """Measure qubit readout SNR versus JPA flux bias.

        At each bias, reset, select no pulse or a pi pulse, then read out
        both preparation branches. IQ moments give a real SNR with the
        configured skew penalty, not an assignment fidelity. Return flux
        device values and one SNR per point; no flux-unit conversion occurs.

        ``jpa_flux_dev`` names a device in ``context.devices`` and
        ``flux_unit`` selects its working unit. None skips bias control
        while retaining the requested coordinates in the result.
        """
        # 1. Prepare the flux coordinates and an SNR trace for the bias scan.
        cfg = deepcopy(cfg)
        soc, soccfg = context.soc, context.soccfg
        jpa_fluxs = sweep2array(cfg.jpa_flux, allow_array=True)

        unit = "a.u." if cfg.flux_unit == "native" else cfg.flux_unit
        fig, ax = context.plots.subplots(num="measurement")
        title = "JPA sweep · flux bias · g/e separation score"
        ax.set(
            title=title,
            xlabel=f"JPA flux-bias value ({unit})",
            ylabel="g/e separation score (a.u.)",
        )
        ax.ticklabel_format(axis="x", style="plain", useOffset=False)
        line = live_line(ax)
        ax.lines[0].set(linestyle="None", marker="o", markersize=4, color="C0")
        retained_shots = np.zeros(len(jpa_fluxs), dtype=np.int64)

        @context.plots.on_plot_thread
        def update(data: NDArray[np.float64]) -> Iterable[Artist]:
            """Draw acquired SNR on the measurement axes; return changed artists.

            Read the scan-shaped data without changing it. Update the live
            artists over the prepared coordinates; plot errors propagate.
            """
            progress = ax.set_title(
                f"{title}\n{np.count_nonzero(np.isfinite(data))}/{len(jpa_fluxs)} settings sampled"
                f" · {retained_shots.sum():,} retained shots"
            )
            return (*line.set(jpa_fluxs, np.abs(data)), progress)

        fig.show()
        signals_buffer = SignalBuffer(
            (len(jpa_fluxs),),
            dtype=np.float64,
            on_update=update,
        )

        # 2. Move the external flux bias before each readout comparison.
        with Schedule(signals_buffer, scope=context.scope) as sched:
            for index, (jpa_flux, step) in enumerate(
                sched.scan("JPA flux bias", jpa_fluxs.tolist())
            ):
                if cfg.jpa_flux_dev is not None:
                    set_flux(
                        context.devices[cfg.jpa_flux_dev],
                        jpa_flux,
                        unit=cfg.flux_unit,
                        stop=context.scope.stop,
                        progress=False,
                    )

                # 3. Compare reset-only and pi-prepared readout in one GE sweep.
                builder = TimelineBuilder(
                    soccfg,
                    reps=cfg.reps,
                    relax_delay=cfg.relax_delay,
                )
                builder.add_loop("ge", 2)
                if cfg.reset is not None:
                    builder.emit(Reset("reset", cfg.reset))
                builder.delay_by_timeline()
                # Empty g case has no pi-length padding; preserve each case's readout time.
                with (
                    builder.branch(by="ge", align_timeline=False) as cases,
                    cases.case(1),
                ):
                    builder.emit(Pulse("pi_pulse", cfg.pi_pulse))
                builder.delay_by_timeline()
                builder.emit(Readout("readout", cfg.readout))
                prog = builder.build()

                def score(
                    sums: NDArray[np.float64], *, point: int = index
                ) -> NDArray[np.float64]:
                    """Track cumulative retained shots from both preparations and return SNR."""
                    retained_shots[point] = int(sums[..., 0].sum())
                    return snr_as_signal(sums, skew_penalty=cfg.skew_penalty)

                step.run_acquire_summed(
                    prog,
                    soc,
                    reducer=shot_sums(prog, [IQ("readout", order=3)]),
                    options=RunOptions(converter=score),
                    rounds=cfg.rounds,
                )

        # 4. Return the SNR trace in the requested flux-coordinate order.
        signals = signals_buffer.array

        return FluxResult(fluxes=jpa_fluxs, signals=signals)

    def analyze(
        self, source: RunRecord[FluxCfg, FluxResult], options: None, *, plots: Plots
    ) -> FluxAnalysis:
        """Select the sampled flux with peak Gaussian-smoothed SNR.

        Smooth ``source.result.signals`` with sigma=1 sample, plot its
        magnitude, and return the peak's device value without conversion.
        ``options`` is None; this is a smoothed peak, not a model fit.
        """
        # 1. Choose the sampled bias at peak SNR after one-sample Gaussian smoothing.
        del options
        result = source.result

        jpa_fluxs = result.fluxes
        signals = result.signals
        signals = gaussian_filter1d(signals, sigma=1)
        snrs = np.abs(signals)

        max_idx = np.argmax(snrs)
        best_jpa_flux = jpa_fluxs[max_idx]

        # 2. Mark the selected bias on the smoothed SNR trace.
        unit = source.cfg.flux_unit if source.cfg is not None else "native units"
        unit = "a.u." if unit == "native" else unit
        _, ax = plots.subplots(num="fit", figsize=config.figsize)
        ax.set(
            title="JPA sweep · flux bias · g/e separation score",
            xlabel=f"JPA flux-bias value ({unit})",
            ylabel="g/e separation score (a.u.)",
        )
        ax.ticklabel_format(axis="x", style="plain", useOffset=False)
        ax.plot(
            jpa_fluxs,
            np.abs(result.signals),
            "o",
            color="C0",
            markersize=4,
            label="Measured",
        )
        ax.plot(jpa_fluxs, snrs, "-", color="C1", label="Gaussian smooth (1 point)")
        ax.axvline(
            best_jpa_flux,
            color="C3",
            linestyle="--",
            label=f"Smoothed-peak flux bias = {best_jpa_flux:.6g} {unit}",
        )
        ax.margins(y=0.25)
        ax.legend(loc="upper center")
        return FluxAnalysis(best_flux=float(best_jpa_flux))
