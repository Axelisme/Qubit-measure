from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar, Literal

import numpy as np
from numpy.typing import NDArray
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.program.v2 import PulseReadout, Reset, SweepCfg, sweep2param

from zcu_lab.v2.support import live_heatmap_with_line
from zcu_lab.v2.support.device import set_flux


@dataclass(frozen=True)
class OneToneFluxResult:
    """Flux device values, readout MHz, and complex IQ [flux, frequency]."""

    fluxes: NDArray[np.float64]
    freqs: NDArray[np.float64]
    signals: NDArray[np.complex128]


class OneToneFluxCfg(ConfigBase):
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
    flux_unit: Annotated[
        Literal["A", "V", "native"],
        FieldMeta(
            label="Flux unit",
            description="Working unit for the selected flux driver; no conversion.",
        ),
    ]
    reset: Annotated[Reset.Cfg | None, FieldMeta(label="Reset")] = None
    readout: Annotated[PulseReadout.Cfg, FieldMeta(label="Readout")]
    jpa_flux_dev: Annotated[
        str,
        FieldMeta(
            label="JPA flux device",
            description="Connected flux device driven by the survey.",
        ),
    ]
    jpa_flux: Annotated[
        SweepCfg,
        FieldMeta(
            label="JPA flux device value",
            description="Flux bounds in flux_unit coordinates.",
        ),
    ]
    freq: Annotated[SweepCfg, FieldMeta(label="Freq", unit="MHz")]


class OneToneFluxExp(Experiment[OneToneFluxResult, OneToneFluxCfg]):
    # Axes are declared inner-first (ADR-0063): inner (fastest-varying) axis is
    # freqs (stored as Hz on disk, MHz in memory); outer axis is jpa flux (a.u.).
    # signals.shape == (len(fluxes), len(freqs)); non-square sweeps round-trip.
    AXES_SPEC: ClassVar[AxesSpec[OneToneFluxResult, OneToneFluxCfg]] = AxesSpec(
        axes=(
            AxesSpec.Axis("freqs", "Readout frequency", "Hz", scale=1e6),
            AxesSpec.Axis("fluxes", "JPA Flux value", "a.u."),
        ),
        z=AxesSpec.Z("signals", "Signal", "a.u."),
        result_type=OneToneFluxResult,
        cfg_type=OneToneFluxCfg,
        tag="jpa/flux_onetone",
        cfg_schema_version="3.0",
    )

    def run(self, cfg: OneToneFluxCfg, *, context: RunContext) -> OneToneFluxResult:
        """Map the readout reflection spectrum versus JPA flux bias.

        Step the external flux bias on the host. At each bias, reset and
        apply a readout pulse while sweeping its frequency on the board.
        Return flux device values, readout frequencies in MHz, and mean
        complex IQ shaped [flux, frequency]. The map is a spectral survey,
        not a qubit SNR measurement or an automatic bias selection.

        ``jpa_flux_dev`` must name a device in ``context.devices``;
        ``flux_unit`` selects its working unit without conversion.
        """
        # 1. Prepare host flux values and the board's readout-frequency grid.
        cfg = deepcopy(cfg)
        soc, soccfg = context.soc, context.soccfg
        jpa_flux_sweep = cfg.jpa_flux

        jpa_fluxs = sweep2array(jpa_flux_sweep, allow_array=True)
        freqs = sweep2array(
            cfg.freq,
            "freq",
            {"soccfg": soccfg, "gen_ch": cfg.readout.gen_ch},
            allow_array=True,
        )

        viewer = live_heatmap_with_line(
            context.plots,
            "measurement",
            "JPA Flux value (a.u.)",
            "Readout frequency (MHz)",
            line_axis=1,
            num_lines=5,
        )
        signals_buffer = SignalBuffer(
            (len(jpa_fluxs), len(freqs)),
            on_update=lambda data: viewer.update(jpa_fluxs, freqs, np.abs(data)),
        )

        # 2. Set the external flux bias before measuring each spectrum.
        with Schedule(cfg, signals_buffer, scope=context.scope) as sched:
            for jpa_flux, step in sched.scan("JPA Flux value", jpa_fluxs.tolist()):
                set_flux(
                    context.devices[step.cfg.jpa_flux_dev],
                    jpa_flux,
                    unit=step.cfg.flux_unit,
                    stop=context.scope.stop,
                    progress=False,
                )

                # 3. Reset, then read out at each board-swept frequency.
                step.cfg.readout.set_param("freq", sweep2param("freq", step.cfg.freq))
                _ = (
                    step.prog_builder(
                        soc,
                        soccfg,
                        reps=step.cfg.reps,
                        relax_delay=step.cfg.relax_delay,
                    )
                    .add(
                        Reset("reset", step.cfg.reset),
                        PulseReadout("readout", step.cfg.readout),
                    )
                    .declare_sweep("freq", step.cfg.freq)
                    .build_and_acquire(rounds=cfg.rounds)
                )

        # 4. Preserve complex IQ in flux-major order for the reflection map.
        signals = signals_buffer.array

        return OneToneFluxResult(fluxes=jpa_fluxs, freqs=freqs, signals=signals)
