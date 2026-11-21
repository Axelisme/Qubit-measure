from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar, Literal

import numpy as np
from matplotlib.artist import Artist
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
    reset: Annotated[
        Reset.Cfg | None,
        FieldMeta(
            label="Reset",
            description="Select a qubit reset before readout; leave empty to skip it.",
        ),
    ] = None
    readout: Annotated[
        PulseReadout.Cfg,
        FieldMeta(
            label="Frequency-swept readout",
            description="Select the resonator pulse and IQ integration; the sweep sets the RF frequency.",
        ),
    ]
    jpa_flux_dev: Annotated[
        DeviceName,
        FieldMeta(
            label="JPA flux-bias source",
            description="Select the DC source that sets the JPA bias.",
        ),
    ]
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
    freq: Annotated[
        SweepCfg,
        FieldMeta(
            label="Readout RF frequency",
            unit="MHz",
            description="Enter Start and Stop in absolute probe RF MHz and Expts as the number of frequency samples (count).",
        ),
    ]
    reps: Annotated[
        int,
        FieldMeta(
            label="Reps",
            unit="count",
            description="Enter the shots per frequency at each bias in one round.",
        ),
    ]
    rounds: Annotated[
        int,
        FieldMeta(
            label="Rounds",
            unit="count",
            description="Enter the number of acquisition rounds per bias.",
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

        fig, _ = context.plots.subplots(2, 1, num="measurement", figsize=(7, 7))
        heat_ax, line_ax = fig.axes
        title = "JPA bias survey · readout resonator · reflection"
        fig.suptitle(title)
        unit = "a.u." if cfg.flux_unit == "native" else cfg.flux_unit
        heat_ax.set(
            xlabel=f"JPA flux-bias value ({unit})", ylabel="Readout RF frequency (MHz)"
        )
        heat_ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        heat_ax.grid(False)
        line_ax.set(
            title="Latest bias spectrum",
            xlabel="Readout RF frequency (MHz)",
            ylabel="Magnitude of mean IQ (a.u.)",
        )
        line_ax.ticklabel_format(axis="x", style="plain", useOffset=False)
        heatmap = live_heatmap(heat_ax, uniform=True)
        fig.colorbar(heat_ax.images[0], ax=heat_ax, label="Magnitude of mean IQ (a.u.)")
        line = live_line(line_ax)
        line_ax.lines[0].set(marker="o", markersize=4, color="C0")

        @context.plots.on_plot_thread
        def update(data: NDArray[np.complex128]) -> Iterable[Artist]:
            """Draw acquired complex IQ on the measurement axes; return changed artists.

            Read the scan-shaped data without changing it. Update the live
            artists over the prepared coordinates; plot errors propagate.
            """
            real = np.abs(data)
            valid = np.flatnonzero(~np.all(np.isnan(real), axis=1))
            progress = fig.suptitle(
                f"{title}\n{len(valid)}/{len(jpa_fluxs)} bias spectra sampled"
            )
            image = heatmap.set(jpa_fluxs, freqs, real)
            if not valid.size:
                return (*image, progress)
            current = int(valid[-1])
            spectrum = line_ax.set_title(
                f"Latest bias spectrum: {jpa_fluxs[current]:.6g} {unit}"
            )
            return (*image, *line.set(freqs, real[current]), progress, spectrum)

        fig.show()
        signals_buffer = SignalBuffer(
            (len(jpa_fluxs), len(freqs)),
            on_update=update,
        )

        # 2. Set the external flux bias before measuring each spectrum.
        with Schedule(signals_buffer, scope=context.scope) as sched:
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
                if cfg.reset is not None:
                    builder.emit(Reset("reset", cfg.reset))
                builder.emit(PulseReadout("readout", cfg.readout, freq=freq))
                step.run_acquire(builder.build(), soc, rounds=cfg.rounds)

        # 4. Preserve complex IQ in flux-major order for the reflection map.
        signals = signals_buffer.array

        return OneToneFluxResult(fluxes=jpa_fluxs, freqs=freqs, signals=signals)
