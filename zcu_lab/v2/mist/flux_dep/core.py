from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Annotated, ClassVar, Literal

import numpy as np
from matplotlib.artist import Artist
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray
from pydantic.dataclasses import dataclass as pydantic_dataclass
from zcu_tools.cfg_model import ConfigBase, DeviceName, FieldMeta
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
    TimelineBuilder,
)
from zcu_tools.simulate import value2flux

from zcu_lab.v2.support import live_heatmap, live_line
from zcu_lab.v2.support.device import set_flux

_TITLE = "MIST flux map · Resonator probe"
_GAIN_LABEL = "Probe gain (a.u.)"
_SHIFT_LABEL = "Magnitude of mean IQ shift (a.u.)"
_CMAP = "viridis"
_HISTORY = 4


@dataclass(frozen=True)
class FluxDepResult:
    """Flux device values, probe gains, and mean complex IQ [flux, gain]."""

    values: NDArray[np.float64]
    gains: NDArray[np.float64]
    signals: NDArray[np.complex128]


@pydantic_dataclass(frozen=True)
class FluxDepAnalyzeOptions:
    flux_half: float | None = None
    flux_period: float | None = None
    ac_coeff: float | None = None
    auto_range: bool = True


def mist_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    """Return clipped IQ displacement from each row's low-gain mean IQ."""
    avg_len = max(int(0.05 * signals.shape[1]), 1)

    mist_signals = np.abs(
        signals - np.mean(signals[:, :avg_len], axis=1, keepdims=True)
    )
    if np.all(np.isnan(mist_signals)):
        return mist_signals

    ref_signals = np.sort(mist_signals.flatten())[: int(0.5 * mist_signals.size)]
    return np.clip(mist_signals, 0, 10 * np.nanmedian(ref_signals))


class FluxDepCfg(ConfigBase):
    reset: Annotated[
        Reset.Cfg | None,
        FieldMeta(
            label="Reset",
            description="Select the reset before each shot, or leave off to skip reset.",
        ),
    ] = None
    init_pulse: Annotated[
        Pulse.Cfg | None,
        FieldMeta(
            label="Preparation pulse",
            description="Select the qubit pulse after reset, or leave off to skip preparation.",
        ),
    ] = None
    probe_pulse: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Resonator probe",
            description="Enter the resonator drive played before readout; Probe gain sweep sets its gain.",
        ),
    ]
    readout: Annotated[
        Readout.Cfg,
        FieldMeta(
            label="Readout",
            description="Enter the calibrated readout that measures complex IQ after the probe.",
        ),
    ]
    flux_unit: Annotated[
        Literal["A", "V", "native"],
        FieldMeta(
            label="Flux-bias unit",
            description="Enter the flux device's unit: A for current, V for voltage, or native for an unscaled device value.",
        ),
    ]
    flux_dev: Annotated[
        DeviceName,
        FieldMeta(
            label="Flux device",
            description="Enter the connected DC bias source to step between gain sweeps.",
        ),
    ]
    flux: Annotated[
        SweepCfg,
        FieldMeta(
            label="Flux-bias sweep",
            unit="device units",
            description="Enter the start, stop and bias point count in the selected Flux-bias unit; no conversion is applied.",
        ),
    ]
    gain: Annotated[
        SweepCfg,
        FieldMeta(
            label="Probe gain sweep",
            unit="a.u.",
            description="Enter the start, stop (a.u.) and point count for the resonator probe gain.",
        ),
    ]
    reps: Annotated[
        int,
        FieldMeta(
            label="Reps",
            unit="count",
            description="Enter the shots per gain point at each bias in one round.",
        ),
    ]
    rounds: Annotated[
        int,
        FieldMeta(
            label="Rounds",
            unit="count",
            description="Enter the number of gain sweeps to average at each bias.",
        ),
    ] = 1
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relax delay",
            unit="µs",
            decimals=3,
            description="Enter the wait after each shot (µs) before the next shot begins.",
        ),
    ]


def _flux_label(cfg: FluxDepCfg | None) -> str:
    """Label recorded flux-bias coordinates without assuming a missing unit."""
    unit = "device units" if cfg is None or cfg.flux_unit == "native" else cfg.flux_unit
    return f"Flux bias ({unit})"


class FluxDepExp(Experiment[FluxDepResult, FluxDepCfg]):
    Options: ClassVar[type[FluxDepAnalyzeOptions]] = FluxDepAnalyzeOptions

    AXES_SPEC: ClassVar[AxesSpec[FluxDepResult, FluxDepCfg]] = AxesSpec(
        axes=(
            AxesSpec.Axis("gains", label="Power", unit="a.u."),
            AxesSpec.Axis("values", label="Flux value", unit="a.u."),
        ),
        z=AxesSpec.Z("signals", label="Signal", unit="a.u."),
        result_type=FluxDepResult,
        cfg_type=FluxDepCfg,
        tag="mist/flux_dep",
        cfg_schema_version="3.0",
    )

    def run(
        self,
        config: FluxDepCfg,
        *,
        context: RunContext,
    ) -> FluxDepResult:
        """Map the MIST readout response versus flux bias and probe gain.

        Step flux on the host. At each bias, reset, apply the optional init
        pulse, apply the gain-swept probe, then read out. The board sweeps
        probe gain. Return flux device values, hardware gain coordinates,
        and mean complex IQ shaped [flux, gain]. Changes in IQ reveal the
        probe-induced response; they are not calibrated state populations.

        ``flux_dev`` must name a device in ``context.devices``;
        ``flux_unit`` selects its working unit without conversion.
        """
        # 1. Prepare external flux values and the board's probe-gain coordinates.
        cfg = config
        soc, soccfg = context.soc, context.soccfg

        values = sweep2array(cfg.flux, allow_array=True)
        gains = sweep2array(
            cfg.gain,
            "gain",
            {"soccfg": soccfg, "gen_ch": cfg.probe_pulse.ch},
        )

        fig, _ = context.plots.subplots(1, 2, num="measurement", figsize=(10, 4))
        heat_ax, line_ax = fig.axes
        fig.suptitle(_TITLE)
        heat_ax.set(
            xlabel=_flux_label(cfg),
            ylabel=_GAIN_LABEL,
        )
        line_ax.set(
            xlabel=_GAIN_LABEL,
            ylabel=_SHIFT_LABEL,
            title="Latest bias row and recent rows",
        )
        heat_ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        line_ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        heat_ax.grid(False)
        heatmap = live_heatmap(heat_ax)
        image = heat_ax.images[0]
        image.set_cmap(_CMAP)
        fig.colorbar(image, ax=heat_ax, label=_SHIFT_LABEL)
        lines = live_line(line_ax, num_lines=_HISTORY)
        for artist in line_ax.lines[:-1]:
            artist.set(marker="None", color="0.6")
        line_ax.lines[-1].set(marker=".", color="C0")

        @context.plots.on_plot_thread
        def update(data: NDArray[np.complex128]) -> Iterable[Artist]:
            """Draw acquired complex IQ on the measurement axes; return changed artists.

            Read the scan-shaped data without changing it. Update the live
            artists over the prepared coordinates; plot errors propagate.
            """
            real = mist_signal2real(data)
            valid = np.flatnonzero(~np.all(np.isnan(real), axis=1))
            current = int(valid[-1]) if valid.size else -1
            history = np.full((_HISTORY, len(gains)), np.nan)
            for offset, index in enumerate(valid[-_HISTORY:][::-1]):
                history[-offset - 1] = real[index]
            title = line_ax.set_title(
                f"Latest: {values[current]:.4g} {cfg.flux_unit} (blue)"
                if current >= 0
                else "Latest bias row and recent rows"
            )
            return (
                *heatmap.set(values, gains, real),
                *lines.set(gains, history),
                title,
            )

        fig.show()
        signals_buffer = SignalBuffer(
            (len(values), len(gains)),
            on_update=update,
        )

        # 2. Set each external bias before acquiring its full gain trace.
        with Schedule(signals_buffer, scope=context.scope) as sched:
            for flux, step in sched.scan("flux", values.tolist()):
                set_flux(
                    context.devices[cfg.flux_dev],
                    flux,
                    unit=cfg.flux_unit,
                    stop=context.scope.stop,
                    progress=False,
                )

                # 3. Reset and initialize before probing each gain at this fixed flux bias.
                builder = TimelineBuilder(
                    soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay
                )
                gain = builder.add_sweep("gain", cfg.gain)
                if cfg.reset is not None:
                    builder.emit(Reset("reset", cfg.reset))
                builder.emit(Pulse("init_pulse", cfg.init_pulse))
                builder.emit(Pulse("probe_pulse", cfg.probe_pulse, gain=gain))
                builder.emit(Readout("readout", cfg.readout))
                prog = builder.build()
                step.run_acquire(prog, soc, rounds=cfg.rounds)

        # 4. Preserve raw complex response; baseline subtraction is display-only.
        return FluxDepResult(values=values, gains=gains, signals=signals_buffer.array)

    def analyze(
        self,
        source: RunRecord[FluxDepCfg, FluxDepResult],
        options: FluxDepAnalyzeOptions,
        *,
        plots: Plots,
    ) -> None:
        """Plot the baseline-subtracted MIST magnitude map from ``source``.

        Each flux row uses its first 5% of gains (at least one) as a mean
        IQ reference; display clipping limits strong outliers. ``ac_coeff``
        maps gain squared to photon number when supplied. Flux conversion
        uses ``flux_half`` and ``flux_period`` together, with device-value
        ticks on top. If either is missing, show only device values below.
        No state-population fit or calibration value is returned.
        """
        # 1. Reference each flux row to low-gain IQ and apply optional axis calibrations.
        result = source.result
        flux_half, flux_period = options.flux_half, options.flux_period

        dev_values, gains, signals = result.values, result.gains, result.signals
        if flux_half is not None and flux_period is not None:
            xs = np.asarray(value2flux(dev_values, flux_half, flux_period))
            xlabel = r"Qubit flux ($\Phi_0$)"
        else:
            xs = dev_values
            xlabel = _flux_label(source.cfg)

        amp_diff = mist_signal2real(signals)
        if options.ac_coeff is None:
            ys = gains
            ylabel = _GAIN_LABEL
        else:
            ys = options.ac_coeff * gains**2
            ylabel = r"Mean photon number $\bar n$"
        ys = np.asarray(ys)

        # 2. Show clipped MIST contrast with device ticks when flux is calibrated.
        fig, ax = plots.subplots(num="fit", figsize=config.figsize)
        ax.set(xlabel=xlabel, ylabel=ylabel)
        ax.grid(False)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        image = ax.pcolormesh(xs, ys, amp_diff.T, shading="nearest", cmap=_CMAP)
        fig.colorbar(image, ax=ax, label=_SHIFT_LABEL)

        if flux_half is not None and flux_period is not None:
            # Retain device-value references when the main axis uses calibrated flux.
            tick_indices = np.round(
                np.linspace(0, len(xs) - 1, min(5, len(xs)))
            ).astype(int)
            secondary = ax.secondary_xaxis("top")
            secondary.set_xlabel(_flux_label(source.cfg))
            secondary.set_xticks(
                xs[tick_indices],
                labels=[f"{value:.3g}" for value in dev_values[tick_indices]],
            )
            fig.suptitle(_TITLE)
        else:
            ax.set_title(_TITLE)

        if options.auto_range:
            ax.set_xlim(xs[0], xs[-1])
            ax.set_ylim(ys[0], ys[-1])
