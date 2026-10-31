from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar, Literal

import numpy as np
from numpy.typing import NDArray
from zcu_tools.cfg_model import ConfigBase, DeviceName, FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.program.v2 import PulseReadout, Reset, SweepCfg, TimelineBuilder

from zcu_lab.v2.support import live_heatmap, live_line
from zcu_lab.v2.support.device import set_flux


@dataclass(frozen=True)
class OneToneFluxResult:
    """Flux device values, readout MHz, and complex IQ [flux, frequency]."""

    fluxes: NDArray[np.float64]
    freqs: NDArray[np.float64]
    signals: NDArray[np.complex128]


class OneToneFluxCfg(ConfigBase):
    reset: Annotated[Reset.Cfg | None, FieldMeta(label="Reset")] = None
    readout: Annotated[PulseReadout.Cfg, FieldMeta(label="Readout")]
    jpa_flux_dev: Annotated[
        DeviceName,
        FieldMeta(
            label="JPA flux device",
            description="Connected flux device driven by the survey.",
        ),
    ]
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
    freq: Annotated[SweepCfg, FieldMeta(label="Freq", unit="MHz")]
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


class OneToneFluxExp(Experiment[OneToneFluxResult, OneToneFluxCfg]):
    # Axes are declared inner-first (ADR-0063): inner (fastest-varying) axis is
    # freqs (stored as Hz on disk, MHz in memory); outer axis is jpa flux (a.u.).
    # signals.shape == (len(fluxes), len(freqs)); non-square sweeps round-trip.
    AXES_SPEC: ClassVar[AxesSpec[OneToneFluxResult, OneToneFluxCfg]] = AxesSpec(
        axes=(
            AxesSpec.Axis("freqs", label="Readout frequency", unit="Hz", scale=1e6),
            AxesSpec.Axis("fluxes", label="JPA Flux value", unit="a.u."),
        ),
        z=AxesSpec.Z("signals", label="Signal", unit="a.u."),
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

        fig, _ = context.plots.subplots(1, 2, num="measurement")
        heat_ax, line_ax = fig.axes
        heat_ax.set(xlabel="JPA Flux value (a.u.)", ylabel="Readout frequency (MHz)")
        line_ax.set_xlabel("Readout frequency (MHz)")
        heatmap = live_heatmap(heat_ax, uniform=True)
        lines = live_line(line_ax, num_lines=5)
        for artist in line_ax.lines[:-1]:
            artist.set(marker="None", alpha=0.3, color="red")
        line_ax.lines[-1].set(label="current line", marker=".", color="C0")

        @context.plots.on_plot_thread
        def update(data):
            real = np.abs(data)
            valid = np.flatnonzero(~np.all(np.isnan(real), axis=1))
            current = int(valid[-1]) if valid.size else -1
            history = np.full((5, len(freqs)), np.nan)
            for offset in range(min(5, current + 1)):
                history[-offset - 1] = real[current - offset]
            return (*heatmap.set(jpa_fluxs, freqs, real), *lines.set(freqs, history))

        fig.show()
        signals_buffer = SignalBuffer(
            (len(jpa_fluxs), len(freqs)),
            on_update=update,
        )

        # 2. Set the external flux bias before measuring each spectrum.
        with Schedule(None, signals_buffer, scope=context.scope) as sched:
            for jpa_flux, step in sched.scan("JPA Flux value", jpa_fluxs.tolist()):
                set_flux(
                    context.devices[cfg.jpa_flux_dev],
                    jpa_flux,
                    unit=cfg.flux_unit,
                    stop=context.scope.stop,
                    progress=False,
                )

                # 3. Reset, then read out at each board-swept frequency.
                builder = TimelineBuilder(
                    soccfg,
                    reps=cfg.reps,
                    relax_delay=cfg.relax_delay,
                )
                freq = builder.add_sweep("freq", cfg.freq)
                reset = Reset("reset", cfg.reset)
                if reset is not None:
                    builder.emit(reset)
                builder.emit(PulseReadout("readout", cfg.readout, freq=freq))
                step.run_acquire(builder.build(), soc, rounds=cfg.rounds)

        # 4. Preserve complex IQ in flux-major order for the reflection map.
        signals = signals_buffer.array

        return OneToneFluxResult(fluxes=jpa_fluxs, freqs=freqs, signals=signals)
