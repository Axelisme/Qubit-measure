from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar, Literal

import numpy as np
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
from zcu_tools.experiment.v2.utils.tracker.moment import MomentTracker
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import Pulse, Readout, Reset, SweepCfg, TimelineBuilder

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
    reset: Annotated[Reset.Cfg | None, FieldMeta(label="Reset")] = None
    pi_pulse: Annotated[Pulse.Cfg, FieldMeta(label="Pi pulse")]
    readout: Annotated[Readout.Cfg, FieldMeta(label="Readout")]
    jpa_flux_dev: Annotated[
        DeviceName | None,
        FieldMeta(
            label="JPA flux device",
            description="Connected flux device name; None skips bias control.",
        ),
    ] = None
    flux_unit: Annotated[
        Literal["A", "V", "native"],
        FieldMeta(
            label="Flux unit",
            description="Working unit for the selected flux driver; no conversion.",
        ),
    ]
    jpa_flux: Annotated[
        SweepCfg,
        FieldMeta(
            label="JPA flux device value",
            description="Flux bounds in flux_unit coordinates.",
        ),
    ]
    skew_penalty: Annotated[
        float,
        FieldMeta(
            label="Skew penalty",
            decimals=3,
            description="Nonnegative SNR skew penalty.",
        ),
    ] = Field(default=0.0, ge=0.0)
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

        fig, ax = context.plots.subplots(num="measurement")
        ax.set(xlabel="JPA Flux value (a.u.)", ylabel="Signal Difference")
        line = live_line(ax)

        @context.plots.on_plot_thread
        def update(data):
            return line.set(jpa_fluxs, np.abs(data))

        fig.show()
        signals_buffer = SignalBuffer(
            (len(jpa_fluxs),),
            dtype=np.float64,
            on_update=update,
        )

        # 2. Move the external flux bias before each readout comparison.
        with Schedule(signals_buffer, scope=context.scope) as sched:
            for jpa_flux, step in sched.scan("JPA Flux value", jpa_fluxs.tolist()):
                if cfg.jpa_flux_dev is not None:
                    set_flux(
                        context.devices[cfg.jpa_flux_dev],
                        jpa_flux,
                        unit=cfg.flux_unit,
                        stop=context.scope.stop,
                        progress=False,
                    )

                # 3. Compare reset-only and pi-prepared readout in one GE sweep.
                tracker = MomentTracker()
                builder = TimelineBuilder(
                    soccfg,
                    reps=cfg.reps,
                    relax_delay=cfg.relax_delay,
                )
                builder.add_loop("ge", 2)
                reset = Reset("reset", cfg.reset)
                if reset is not None:
                    builder.emit(reset)
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
                step.run_acquire(
                    prog,
                    soc,
                    # A host scalar step stores exactly one tracker's SNR.
                    options=RunOptions(
                        converter=lambda _raw, tracker=tracker: snr_as_signal(
                            [tracker],
                            ge_axis=1,
                            skew_penalty=cfg.skew_penalty,
                        ).reshape(()),
                    ),
                    trackers=[tracker],
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
        del options
        result = source.result

        jpa_fluxs = result.fluxes
        signals = result.signals
        signals = gaussian_filter1d(signals, sigma=1)
        snrs = np.abs(signals)

        max_idx = np.argmax(snrs)
        best_jpa_flux = jpa_fluxs[max_idx]

        fig, ax = plots.subplots(num="fit", figsize=config.figsize)
        ax.plot(jpa_fluxs, snrs, label="signal difference")
        ax.axvline(
            best_jpa_flux,
            color="r",
            ls="--",
            label=f"best JPA flux = {best_jpa_flux:.2g} a.u.",
        )
        ax.set_xlabel("JPA Flux value (a.u.)")
        ax.set_ylabel("Signal Difference (a.u.)")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()

        return FluxAnalysis(best_flux=float(best_jpa_flux))
