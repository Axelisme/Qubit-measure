from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import Field
from scipy.ndimage import gaussian_filter1d
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import (
    IDENTITY,
    AxesSpec,
    Axis,
    PersistableExperiment,
    ZSpec,
    config,
)
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.utils import set_flux
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.experiment.v2.utils.snr import snr_as_signal
from zcu_tools.experiment.v2.utils.tracker.moment import MomentTracker
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    Branch,
    ProgramV2Cfg,
    Pulse,
    PulseCfg,
    Readout,
    ReadoutCfg,
    Reset,
    ResetCfg,
    SweepCfg,
)


@dataclass(frozen=True)
class FluxResult:
    """JPA flux device values and moment-based GE SNR at each value."""

    fluxes: NDArray[np.float64]
    signals: NDArray[np.float64]


@dataclass(frozen=True)
class FluxAnalysis:
    best_flux: float


class FluxModuleCfg(ConfigBase):
    reset: Annotated[ResetCfg | None, FieldMeta(label="Reset")] = None
    pi_pulse: Annotated[PulseCfg, FieldMeta(label="Pi pulse")]
    readout: Annotated[ReadoutCfg, FieldMeta(label="Readout")]


class FluxSweepCfg(ConfigBase):
    jpa_flux: Annotated[
        SweepCfg,
        FieldMeta(
            label="JPA flux device value",
            description="Flux bounds in flux_unit coordinates.",
        ),
    ]


class FluxCfg(ProgramV2Cfg, ExpCfgModel):
    jpa_flux_dev: Annotated[
        str | None,
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
    modules: FluxModuleCfg
    sweep: FluxSweepCfg
    skew_penalty: Annotated[
        float,
        FieldMeta(
            label="Skew penalty",
            decimals=3,
            description="Nonnegative SNR skew penalty.",
        ),
    ] = Field(default=0.0, ge=0.0)


class FluxExp(PersistableExperiment[FluxResult, FluxCfg]):
    # jpa_flux stored as-is (a.u.) on disk -> scale=IDENTITY; signals are float64
    AXES_SPEC = AxesSpec(
        axes=(Axis("fluxes", "JPA Flux value", "a.u.", scale=IDENTITY),),
        z=ZSpec("signals", "Signal", "a.u.", dtype=np.float64),
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
        jpa_fluxs = sweep2array(cfg.sweep.jpa_flux, allow_array=True)

        viewer = context.plots.liveplot_1d(
            "measurement", "JPA Flux value (a.u.)", "Signal Difference"
        )
        signals_buffer = SignalBuffer(
            (len(jpa_fluxs),),
            dtype=np.float64,
            on_update=lambda data: viewer.update(jpa_fluxs, np.abs(data)),
        )

        # 2. Move the external flux bias before each readout comparison.
        with Schedule(cfg, signals_buffer, stop=context.cancel_signal) as sched:
            for jpa_flux, step in sched.scan("JPA Flux value", jpa_fluxs.tolist()):
                if step.cfg.jpa_flux_dev is not None:
                    set_flux(
                        context.devices[step.cfg.jpa_flux_dev],
                        jpa_flux,
                        unit=step.cfg.flux_unit,
                        stop=context.cancel_signal,
                        progress=False,
                    )

                # 3. Compare reset-only and pi-prepared readout in one GE sweep.
                modules = step.cfg.modules
                tracker = MomentTracker()
                _ = (
                    step.prog_builder(soc, soccfg)
                    .add(
                        Reset("reset", modules.reset),
                        Branch("ge", [], Pulse("pi_pulse", modules.pi_pulse)),
                        Readout("readout", modules.readout),
                    )
                    .declare_sweep("ge", 2)
                    .build_and_acquire(
                        raw2signal_fn=lambda raw, tracker=tracker, skew_penalty=step.cfg.skew_penalty: (
                            snr_as_signal(
                                [tracker],
                                ge_axis=1,
                                skew_penalty=skew_penalty,
                            )
                        ),
                        trackers=[tracker],
                    )
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

        fig, ax = plots.subplots("fit", figsize=config.figsize)
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
