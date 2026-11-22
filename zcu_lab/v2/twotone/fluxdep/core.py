from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar, Literal

import numpy as np
from matplotlib.artist import Artist
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from numpy.typing import NDArray
from pydantic import Field
from zcu_tools.analysis.fluxdep.line_state import FluxPickAnalysis
from zcu_tools.cfg_model import ConfigBase, DeviceName, FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment, RunRecord
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.v2.runtime import RunOptions
from zcu_tools.experiment.v2.runtime.schedule import (
    Schedule,
    SignalBuffer,
)
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import Pulse, Readout, Reset, SweepCfg, TimelineBuilder
from zcu_tools.utils.process import minus_background

from zcu_lab.v2.support import live_heatmap, live_line
from zcu_lab.v2.support.device import set_flux


@dataclass(frozen=True)
class FreqFluxResult:
    """Flux-device values, frequencies in MHz, and mean IQ shaped (Nflux, Nfreq)."""

    values: NDArray[np.float64]
    freqs: NDArray[np.float64]
    signals: NDArray[np.complex128]


class FreqFluxCfg(ConfigBase):
    flux_unit: Annotated[
        Literal["A", "V", "native"],
        FieldMeta(
            label="Flux unit",
            description="Choose A, V, or native to match the selected DC bias source; the scan values are not converted.",
        ),
    ]  # External flux driver unit; no conversion.
    reset: Annotated[
        Reset.Cfg | None,
        FieldMeta(
            label="Reset",
            description="Select a reset before each probe; leave blank to omit it.",
        ),
    ] = None
    init_pulse: Annotated[Pulse.Cfg | None, FieldMeta(label="Init Pulse")] = None
    qub_pulse: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Probe Pulse",
            description="Fill in the qubit drive waveform played at each RF frequency.",
        ),
    ]
    readout: Annotated[
        Readout.Cfg,
        FieldMeta(
            label="Readout",
            description="Fill in the calibrated resonator readout applied after the probe.",
        ),
    ]
    flux_dev: Annotated[
        DeviceName,
        FieldMeta(
            label="Flux Device",
            description="Fill in the connected DC bias source to step between spectra.",
        ),
    ]
    flux: Annotated[
        SweepCfg,
        FieldMeta(
            label="DC bias",
            decimals=6,
            description="Fill in the bias endpoints in the selected Flux unit and the number of spectra.",
        ),
    ]
    freq: Annotated[
        SweepCfg,
        FieldMeta(
            label="Probe RF frequency",
            unit="MHz",
            description="Fill in the qubit drive RF endpoints in MHz and the points per spectrum.",
        ),
    ]
    reps: Annotated[
        int,
        FieldMeta(
            label="Reps", unit="count", description="Number of repetitions per round."
        ),
    ]
    rounds: Annotated[
        int,
        FieldMeta(
            label="Rounds",
            unit="count",
            description="Number of software acquisition rounds.",
        ),
    ] = 1
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relax delay",
            unit="µs",
            decimals=3,
            description="Relaxation delay after each repetition.",
        ),
    ]
    acquire_retry: Annotated[
        int,
        FieldMeta(
            label="Acquisition retries",
            unit="count",
            description="Set the number of retries at one bias after a transient acquisition error.",
        ),
    ] = Field(default=0, ge=0)


class FreqFluxExp(Experiment[FreqFluxResult, FreqFluxCfg]):
    # inner freqs stores MHz on disk -> scale=1e6; outer values raw -> IDENTITY
    AXES_SPEC: ClassVar[AxesSpec[FreqFluxResult, FreqFluxCfg]] = AxesSpec(
        axes=(
            AxesSpec.Axis("freqs", label="Frequency", unit="Hz", scale=1e6),
            AxesSpec.Axis("values", label="Flux device value", unit="a.u."),
        ),
        z=AxesSpec.Z("signals", label="Signal", unit="a.u."),
        result_type=FreqFluxResult,
        cfg_type=FreqFluxCfg,
        tag="twotone/flux_dep/freq",
        cfg_schema_version="3.1",
    )

    def run(
        self,
        cfg: FreqFluxCfg,
        *,
        context: RunContext,
    ) -> FreqFluxResult:
        """Map qubit spectroscopy versus the selected device's flux bias.

        At each host-set flux value, reset, apply the optional init pulse,
        sweep the qubit probe frequency on the FPGA, then read out. Return
        mean complex IQ on the (flux, frequency) grid; no line is selected.
        ``flux_dev`` must name a context device accepting ``flux_unit``;
        values are passed in that unit without conversion. ``acquire_retry``
        controls acquisition retries at each flux point.
        """
        # 1. Prepare the device-value axis and the probe's hardware frequency grid.
        cfg = deepcopy(cfg)
        soc, soccfg = context.soc, context.soccfg

        value_sweep = cfg.flux
        freq_sweep = cfg.freq

        dev_values = sweep2array(value_sweep, allow_array=True)
        freqs = sweep2array(
            freq_sweep, "freq", {"soccfg": soccfg, "gen_ch": cfg.qub_pulse.ch}
        )

        signals_buffer = self._live_spectra(context.plots, cfg, dev_values, freqs)

        # 3. Set the external flux bias before acquiring each complete spectrum.
        with Schedule(signals_buffer, scope=context.scope) as sched:
            for _, step in sched.scan("flux", dev_values.tolist()):
                set_flux(
                    context.devices[cfg.flux_dev],
                    step.value,
                    unit=cfg.flux_unit,
                    stop=context.scope.stop,
                    progress=False,
                )

                # Reset and initialize before each frequency-swept probe and readout.
                builder = TimelineBuilder(
                    soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay
                )
                freq_sweep = cfg.freq
                freq_param = builder.add_sweep("freq", freq_sweep)

                if cfg.reset is not None:
                    builder.emit(cfg.reset.build("reset"))
                builder.emit(Pulse("init_pulse", cfg.init_pulse))
                builder.emit(Pulse("qub_pulse", cfg.qub_pulse, freq=freq_param))
                builder.emit(Readout("readout", cfg.readout))
                prog = builder.build()
                step.run_acquire(
                    prog,
                    soc,
                    options=RunOptions(retry=cfg.acquire_retry),
                    rounds=cfg.rounds,
                )

        return FreqFluxResult(
            values=dev_values,
            freqs=freqs,
            signals=signals_buffer.array,
        )

    def _live_spectra(
        self,
        plots: Plots,
        cfg: FreqFluxCfg,
        dev_values: NDArray[np.float64],
        freqs: NDArray[np.float64],
    ) -> SignalBuffer:
        """Attach the map and recent spectra to the run's live buffer."""
        # 2. Track the flux-dependent contrast map and recent frequency traces.
        fig, _ = plots.subplots(
            1, 2, num="measurement", figsize=(10, 4.8), layout="constrained"
        )
        heat_ax, line_ax = fig.axes
        fig.suptitle(f"Two-tone flux map · {cfg.flux_dev} · qubit spectrum")
        unit = cfg.flux_unit if cfg.flux_unit != "native" else "device units"
        heat_ax.set(xlabel=f"DC bias ({unit})", ylabel="Probe RF frequency (MHz)")
        contrast_label = "Background-subtracted IQ magnitude (a.u.)"
        line_ax.set(xlabel="Probe RF frequency (MHz)", ylabel=contrast_label)
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-2, 2))
        heat_ax.xaxis.set_major_formatter(formatter)
        heat_ax.grid(False)
        heatmap = live_heatmap(heat_ax, uniform=False, cmap="viridis")
        (image,) = heatmap.set(
            dev_values, freqs, np.full((len(dev_values), len(freqs)), np.nan)
        )
        fig.colorbar(image, ax=heat_ax, label=contrast_label)
        line_ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        lines = live_line(line_ax, num_lines=2)
        line_ax.lines[0].set(
            label="Previous bias", marker="o", markersize=3, color="0.55"
        )
        line_ax.lines[1].set(label="Latest bias", marker="o", markersize=3, color="C0")
        line_ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncols=2)

        @plots.on_plot_thread
        def update(data: NDArray[np.complex128]) -> tuple[Artist, ...]:
            """Refresh the flux contrast map and two recent frequency traces.

            data is complex IQ shaped (flux bias, frequency); NaN marks unmeasured points.
            Return the changed artists without modifying the input array.
            """
            # Display each row's contrast relative to its own median background.
            real = np.abs(minus_background(data, axis=1))
            valid = np.flatnonzero(~np.all(np.isnan(real), axis=1))
            current = int(valid[-1]) if valid.size else -1
            history = np.full((2, len(freqs)), np.nan)
            for offset, row in enumerate(valid[-2:][::-1]):
                history[-offset - 1] = real[row]
            progress = heat_ax.set_title(
                f"Bias rows with data: {len(valid)}/{len(dev_values)}"
            )
            if current >= 0:
                line_ax.set_title(f"Latest DC bias: {dev_values[current]:.6g} {unit}")
            return (
                *heatmap.set(dev_values, freqs, real),
                *lines.set(freqs, history),
                progress,
            )

        fig.show()
        return SignalBuffer(
            (len(dev_values), len(freqs)),
            on_update=update,
        )

    def plot_pick(
        self,
        source: RunRecord[FreqFluxCfg, FreqFluxResult],
        selection: FluxPickAnalysis,
        *,
        plots: Plots,
    ) -> None:
        """Plot the recorded map and accepted half/integer selections as ``pick``.

        The source supplies mean complex IQ and scan axes; historical cfg, when
        available, supplies the bias label and unit. Selection values use the
        recorded device coordinate and come from the picker, not a fit here.
        The figure uses the same IQ contrast as Run; input arrays are unchanged.
        Figure registration and plotting errors propagate.
        """
        result = source.result
        cfg = source.cfg
        unit = (
            "device units"
            if cfg is None or cfg.flux_unit == "native"
            else cfg.flux_unit
        )
        target = "qubit" if cfg is None else cfg.flux_dev
        fig, ax = plots.subplots(num="pick", figsize=(8, 5), layout="constrained")
        ax.set(
            title=f"Two-tone flux map · {target} · flux-line picks",
            xlabel=f"DC bias ({unit})",
            ylabel="Probe RF frequency (MHz)",
        )
        ax.grid(False)
        heatmap = live_heatmap(ax, uniform=False, cmap="viridis")
        contrast = np.abs(minus_background(result.signals, axis=1))
        (image,) = heatmap.set(result.values, result.freqs, contrast)
        fig.colorbar(image, ax=ax, label="Background-subtracted IQ magnitude (a.u.)")
        ax.axvline(selection.flux_half, color="red", linestyle="--", label="Half flux")
        ax.axvline(
            selection.flux_int, color="blue", linestyle="--", label="Integer flux"
        )
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncols=2)
        ax.set_title(
            f"Two-tone flux map · {target} · flux-line picks\n"
            f"Half flux = {selection.flux_half:.6g} {unit} · "
            f"Integer flux = {selection.flux_int:.6g} {unit}\n"
            f"Flux period = {selection.flux_period:.6g} {unit}"
        )
