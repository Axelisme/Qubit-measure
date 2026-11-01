from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar, Literal

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
class FluxDepResult:
    values: NDArray[np.float64]
    freqs: NDArray[np.float64]
    signals: NDArray[np.complex128]


def fluxdep_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return np.abs(signals)


class FluxDepCfg(ConfigBase):
    reset: Annotated[Reset.Cfg | None, FieldMeta(label="Reset")] = None
    readout: Annotated[PulseReadout.Cfg, FieldMeta(label="Readout")]
    flux_dev: Annotated[
        DeviceName,
        FieldMeta(
            label="Flux Device",
            description="Name of the connected flux-bias device driven by the sweep.",
        ),
    ]
    flux_unit: Annotated[
        Literal["A", "V", "native"],
        FieldMeta(
            label="Flux unit",
            description="Working unit for the selected flux driver; no conversion.",
        ),
    ]
    flux: Annotated[
        SweepCfg,
        FieldMeta(
            label="Flux device value",
            description="Flux device sweep in flux_unit coordinates.",
        ),
    ]
    freq: Annotated[
        SweepCfg,
        FieldMeta(
            label="Frequency",
            unit="MHz",
            description="Readout frequency sweep.",
        ),
    ]
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


class FluxDepExp(Experiment[FluxDepResult, FluxDepCfg]):
    # inner axis (fastest-varying) = freqs (MHz in memory, Hz on disk);
    # outer axis = flux device values (a.u.).
    AXES_SPEC: ClassVar[AxesSpec[FluxDepResult, FluxDepCfg]] = AxesSpec(
        axes=(
            AxesSpec.Axis("freqs", label="Frequency", unit="Hz", scale=1e6),
            AxesSpec.Axis("values", label="Flux device value", unit="a.u."),
        ),
        z=AxesSpec.Z("signals", label="Signal", unit="a.u."),
        result_type=FluxDepResult,
        cfg_type=FluxDepCfg,
        tag="onetone/flux_dep",
        cfg_schema_version="3.0",
    )

    def run(
        self,
        config: FluxDepCfg,
        *,
        context: RunContext,
        acquire_kwargs: dict[str, Any] | None = None,
    ) -> FluxDepResult:
        """Map the resonator response versus flux bias and readout frequency.

        At each host-side flux value, apply optional reset and sweep the
        readout tone frequency on the board. Return flux-device values in
        cfg.flux_unit, frequencies in MHz, and mean complex IQ shaped
        (Nflux, Nfreq). The map reveals resonance tracks; line selection and
        flux calibration are separate analysis steps.

        cfg.flux_dev must name a connected device supporting cfg.flux_unit.
        acquire_kwargs forwards optional acquisition controls to the builder.
        """

        # Prepare the device coordinates and the matched generator/ADC grid.
        soc, soccfg = context.soc, context.soccfg
        cfg = deepcopy(config)
        freq_sweep = cfg.freq
        flux_sweep = cfg.flux

        dev_values = sweep2array(flux_sweep, allow_array=True)
        freqs = sweep2array(
            freq_sweep,
            "freq",
            {
                "soccfg": soccfg,
                "gen_ch": cfg.readout.gen_ch,
                "ro_ch": cfg.readout.ro_ch,
            },
        )

        # Move to the first bias with progress before collecting the map.
        set_flux(
            context.devices[cfg.flux_dev],
            dev_values[0],
            unit=cfg.flux_unit,
            stop=context.scope.stop,
            progress=True,
        )

        # Display magnitude, but retain complex IQ for each flux/frequency point.
        fig, _ = context.plots.subplots(1, 2, num="measurement")
        heat_ax, line_ax = fig.axes
        heat_ax.set(xlabel="Flux device value", ylabel="Frequency (MHz)")
        line_ax.set_xlabel("Frequency (MHz)")
        heatmap = live_heatmap(heat_ax, uniform=False)
        lines = live_line(line_ax, num_lines=10)
        for artist in line_ax.lines[:-1]:
            artist.set(marker="None", alpha=0.3, color="red")
        line_ax.lines[-1].set(label="current line", marker=".", color="C0")

        @context.plots.on_plot_thread
        def update(data):
            real = fluxdep_signal2real(data)
            valid = np.flatnonzero(~np.all(np.isnan(real), axis=1))
            current = int(valid[-1]) if valid.size else -1
            history = np.full((10, len(freqs)), np.nan)
            for offset in range(min(10, current + 1)):
                history[-offset - 1] = real[current - offset]
            return (
                *heatmap.set(dev_values, freqs, real),
                *lines.set(freqs, history),
            )

        fig.show()
        signals_buffer = SignalBuffer(
            (len(dev_values), len(freqs)),
            on_update=update,
        )
        with Schedule( signals_buffer, scope=context.scope) as sched:
            for _, step in sched.scan("flux", dev_values.tolist()):
                set_flux(
                    context.devices[cfg.flux_dev],
                    step.value,
                    unit=cfg.flux_unit,
                    stop=context.scope.stop,
                    progress=False,
                )

                # Reset at each frequency, then excite and measure the resonator.
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
                step.run_acquire(
                    builder.build(),
                    soc,
                    **(acquire_kwargs or {}),
                    rounds=cfg.rounds,
                )

        # Publish the final map after a normal or cooperatively stopped return.
        signals_buffer.trigger_update(flush=True)
        return FluxDepResult(
            values=dev_values,
            freqs=freqs,
            signals=signals_buffer.array,
        )
