from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar, Literal

import numpy as np
from matplotlib.artist import Artist
from numpy.typing import NDArray
from zcu_tools.analysis.fluxdep.line_state import FluxPickAnalysis
from zcu_tools.cfg_model import ConfigBase, DeviceName, FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import PulseReadout, Reset, SweepCfg, TimelineBuilder

from zcu_lab.v2.support import live_heatmap, live_line
from zcu_lab.v2.support.device import set_flux


@dataclass(frozen=True)
class FluxDepResult:
    values: NDArray[np.float64]
    freqs: NDArray[np.float64]
    signals: NDArray[np.complex128]


def fluxdep_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    """Return the real-valued measurement view of complex IQ without changing the input."""
    return np.abs(signals)


class FluxDepCfg(ConfigBase):
    reset: Annotated[Reset.Cfg | None, FieldMeta(label="Reset")] = None
    readout: Annotated[
        PulseReadout.Cfg,
        FieldMeta(
            label="Readout",
            description="Set the pulse and ADC window used at each bias and frequency.",
        ),
    ]
    flux_dev: Annotated[
        DeviceName,
        FieldMeta(
            label="Flux-bias device",
            description="Select the device that applies the swept flux bias.",
        ),
    ]
    flux_unit: Annotated[
        Literal["A", "V", "native"],
        FieldMeta(
            label="Bias unit",
            description="Choose the selected device's native bias unit, A, V or native; the sweep does not convert values.",
        ),
    ]
    flux: Annotated[
        SweepCfg,
        FieldMeta(
            label="Flux bias",
            description="Enter the start and stop in Bias unit, not in flux quanta.",
        ),
    ]
    freq: Annotated[
        SweepCfg,
        FieldMeta(
            label="Readout RF frequency",
            unit="MHz",
            description="Enter the absolute RF sweep endpoints in MHz for each flux bias.",
        ),
    ]
    reps: Annotated[
        int,
        FieldMeta(
            label="Reps",
            unit="count",
            description="Enter the shots per frequency and bias in each round.",
        ),
    ]
    rounds: Annotated[
        int,
        FieldMeta(
            label="Rounds",
            unit="count",
            description="Enter how many frequency sweeps to average at each bias.",
        ),
    ] = 1
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relax delay",
            unit="µs",
            decimals=3,
            description="Enter the relaxation wait in µs after each shot.",
        ),
    ]


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

        # 1. Prepare the device coordinates and the matched generator/ADC grid.
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

        # 2. Move to the first bias with progress before collecting the map.
        set_flux(
            context.devices[cfg.flux_dev],
            dev_values[0],
            unit=cfg.flux_unit,
            stop=context.scope.stop,
            progress=True,
        )

        # 3. Display magnitude, but retain complex IQ for each flux/frequency point.
        fig, _ = context.plots.subplots(2, 1, num="measurement", layout="constrained")
        fig.suptitle("One-tone flux dependence · Resonator · IQ magnitude")
        heat_ax, line_ax = fig.axes
        unit = "device units" if cfg.flux_unit == "native" else cfg.flux_unit
        heat_ax.set(xlabel=f"Flux bias ({unit})", ylabel="Readout RF frequency (MHz)")
        heat_ax.grid(False)
        line_ax.set(
            xlabel="Readout RF frequency (MHz)", ylabel="Magnitude of mean IQ (a.u.)"
        )
        heatmap = live_heatmap(heat_ax, uniform=False)
        fig.colorbar(heat_ax.images[0], ax=heat_ax, label="Magnitude of mean IQ (a.u.)")
        line = live_line(line_ax)
        progress = heat_ax.set_title(
            f"Bias settings measured: 0 / {len(dev_values)}", fontsize=11
        )
        slice_label = line_ax.set_title("Latest bias spectrum", fontsize=11)

        @context.plots.on_plot_thread
        def update(data: NDArray[np.complex128]) -> Iterable[Artist]:
            """Display measured bias columns and the latest available spectrum."""
            real = fluxdep_signal2real(data)
            valid = np.flatnonzero(np.any(np.isfinite(real), axis=1))
            if not valid.size:
                return ()
            current = valid[-1]
            progress.set_text(
                f"Bias settings measured: {valid.size} / {len(dev_values)}"
            )
            slice_label.set_text(f"Latest bias: {dev_values[current]:.4g} {unit}")
            return (
                *heatmap.set(dev_values, freqs, real),
                *line.set(freqs, real[current]),
                progress,
                slice_label,
            )

        fig.show()
        signals_buffer = SignalBuffer(
            (len(dev_values), len(freqs)),
            on_update=update,
        )
        with Schedule(signals_buffer, scope=context.scope) as sched:
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
                if cfg.reset is not None:
                    builder.emit(Reset("reset", cfg.reset))
                builder.emit(PulseReadout("readout", cfg.readout, freq=freq))
                step.run_acquire(
                    builder.build(),
                    soc,
                    **(acquire_kwargs or {}),
                    rounds=cfg.rounds,
                )

        # 4. Publish the final map after a normal or cooperatively stopped return.
        signals_buffer.trigger_update(flush=True)
        return FluxDepResult(
            values=dev_values,
            freqs=freqs,
            signals=signals_buffer.array,
        )

    @staticmethod
    def plot_pick(
        source: RunRecord[FluxDepCfg, FluxDepResult],
        pick: FluxPickAnalysis,
        *,
        plots: Plots,
    ) -> None:
        """Overlay the accepted half/integer picks and period on the measured MHz map.

        Use the source's saved bias unit, or label as device units when absent.
        The caller supplies the already validated pick; this method draws only.
        """
        result, cfg = source.result, source.cfg
        unit = (
            "device units"
            if cfg is None or cfg.flux_unit == "native"
            else cfg.flux_unit
        )
        fig, ax = plots.subplots(num="pick", layout="constrained")
        fig.suptitle("One-tone flux dependence · Resonator · IQ magnitude")
        ax.set(
            xlabel=f"Flux bias ({unit})",
            ylabel="Readout RF frequency (MHz)",
            title=f"Bias period = {pick.flux_period:.6g} {unit}",
        )
        ax.grid(False)
        heatmap = live_heatmap(ax, uniform=False)
        (image,) = heatmap.set(
            result.values, result.freqs, fluxdep_signal2real(result.signals)
        )
        fig.colorbar(image, ax=ax, label="Magnitude of mean IQ (a.u.)")
        ax.axvline(
            pick.flux_half,
            color="red",
            linestyle="--",
            label=f"Half-flux bias = {pick.flux_half:.6g} {unit}",
        )
        ax.axvline(
            pick.flux_int,
            color="blue",
            linestyle=":",
            label=f"Integer-flux bias = {pick.flux_int:.6g} {unit}",
        )
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), fontsize=10)
