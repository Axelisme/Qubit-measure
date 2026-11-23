"""Independent local line DC scan with mux twotone spectra."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Annotated, ClassVar, Literal

import numpy as np
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray
from pydantic import Field, StrictInt, model_validator
from zcu_tools.cfg_model import ConfigBase, DeviceName, FieldMeta
from zcu_tools.experiment import Experiment
from zcu_tools.experiment.axes_spec import GroupedAxesSpec
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.utils.sweep import make_sweep
from zcu_tools.experiment.v2.runtime import RunOptions, Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    Reset,
    ResetCfg,
    TimelineBuilder,
    sweep2param,
)
from zcu_tools.program.v2.modules import MuxReadout, MuxReadoutCfg, Pulse
from zcu_tools.utils.process import minus_background

from zcu_lab.v2.support import live_heatmap, live_line
from zcu_lab.v2.support.device import check_flux_unit, set_flux
from zcu_lab.v2.support.mux import averaged_mux_signal
from zcu_lab.v2.support.mux_flux import MuxFluxResult, mux_flux_spec
from zcu_lab.v2.support.mux_sweep_cfg import (
    MuxDriveToneCfg,
    ToneRangeCfg,
    validate_mux_scan,
)


class TwotoneFluxLocalToneCfg(MuxDriveToneCfg):
    """One original qubit's drive, MHz range and local DC source/range.

    name, drive and freq retain the spectrum tone's meaning. flux_dev names a
    connected DC source even for inactive tones. flux gives finite native-unit
    bounds, required only when selected. All selected lines advance together
    using the parent flux_expts, not as a Cartesian product.
    """

    freq: Annotated[
        ToneRangeCfg | None,
        FieldMeta(
            label="Frequency",
            unit="MHz",
            description="Enter this tone's qubit drive RF frequency range in MHz.",
        ),
    ] = None
    flux_dev: Annotated[
        DeviceName,
        FieldMeta(
            label="Flux source",
            description="Enter the DC source that sets the qubit's local flux bias.",
        ),
    ] = Field(min_length=1)
    flux: Annotated[
        ToneRangeCfg | None,
        FieldMeta(
            label="DC bias",
            description="Enter the bias sweep endpoints in the selected Flux unit.",
        ),
    ] = None

    def require_freq(self) -> ToneRangeCfg:
        """Return selected MHz bounds, or raise ValueError naming the qubit."""
        if self.freq is None:
            raise ValueError(f"{self.name}: active tone requires freq")
        return self.freq

    def require_flux(self) -> ToneRangeCfg:
        """Return selected DC bounds, or raise ValueError naming the qubit."""
        if self.flux is None:
            raise ValueError(f"{self.name}: active tone requires DC sweep")
        return self.flux


class TwotoneFluxLocalCfg(ConfigBase):
    """Full original mux tones with zipped local DC and inner MHz scans.

    reset is optional; readout contains every original readout position. tones
    pairs each position with its named drive, absolute MHz bounds, source and
    native DC bounds. flux_expts is the positive shared outer count; expts is
    the positive shared hardware frequency count. participants selects unique
    original indices in acquisition/result order, without slicing either list.
    Selected tones require drive, freq and flux, with distinct DC sources.
    flux_unit is A, V or native without conversion. settle is a nonnegative
    finite wait in seconds after each point's DC writes. reps/rounds are
    positive averaging counts; initial_delay/relax_delay are finite nonnegative
    microseconds. Invalid identities, drives, timing or scan bounds raise.
    """

    reset: Annotated[
        ResetCfg | None,
        FieldMeta(
            label="Reset",
            description="Choose the common preparation reset; leave blank to omit it.",
        ),
    ] = None
    readout: Annotated[
        MuxReadoutCfg,
        FieldMeta(
            label="Readout",
            description="Enter the calibrated multiplexed readout used after the qubit drive sequence.",
        ),
    ]
    tones: Annotated[list[TwotoneFluxLocalToneCfg], FieldMeta(element_label="name")]
    flux_expts: Annotated[int, FieldMeta(label="Flux points", unit="count")] = Field(
        default=101, strict=True, gt=0
    )
    expts: Annotated[int, FieldMeta(label="Expts", unit="count")] = Field(
        strict=True,
        gt=0,
        description="Hardware frequency point count shared by all tones.",
    )
    participants: Annotated[
        list[StrictInt],
        FieldMeta(
            description="Enter the original zero-based tone indices in measurement order."
        ),
    ]
    flux_unit: Annotated[
        Literal["A", "V", "native"],
        FieldMeta(
            label="Flux unit",
            description="Choose the DC source's configured unit; values are not converted.",
        ),
    ] = "A"
    settle: Annotated[
        float,
        FieldMeta(
            label="Settle time",
            unit="s",
            description="Enter the wait after setting the DC source in seconds.",
        ),
    ] = Field(default=0.1, ge=0, allow_inf_nan=False)

    reps: Annotated[
        int,
        FieldMeta(
            label="Reps", unit="count", description="Number of repetitions per round."
        ),
    ] = Field(gt=0)
    rounds: Annotated[
        int,
        FieldMeta(
            label="Rounds",
            unit="count",
            description="Number of software acquisition rounds.",
        ),
    ] = Field(default=1, gt=0)
    initial_delay: Annotated[
        float,
        FieldMeta(
            label="Initial delay",
            unit="µs",
            description="Delay before the first repetition.",
        ),
    ] = Field(default=1.0, ge=0, allow_inf_nan=False)
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relax delay",
            unit="µs",
            decimals=3,
            description="Relaxation delay after each repetition.",
        ),
    ] = Field(ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_scan(self) -> TwotoneFluxLocalCfg:
        """Reject invalid selected drives, duplicate sources and inconsistent ranges."""
        validate_mux_scan(self, self.tones, self.participants, self.expts)
        devices = [self.tones[index].flux_dev for index in self.participants]
        if len(set(devices)) != len(devices):
            raise ValueError("local flux lines must use distinct devices")
        return self


def _require_finite_spectrum(
    acquired: NDArray[np.complex128], *, point: int, result: MuxFluxResult
) -> None:
    """Raise ValueError for nonfinite completed IQ, reporting planned coordinates.

    acquired is one tone-major (K,P) complex spectrum. point is its zero-based
    DC index in result, which supplies original tone and planned DC/MHz axes
    for the error message. Return None for all-finite IQ; mutate no inputs.
    """
    nonfinite = ~np.isfinite(acquired)
    if nonfinite.any():
        row, column = np.argwhere(nonfinite)[0]
        raise ValueError(
            f"Nonfinite inner IQ at flux point {point} (zero-based): "
            f"{np.count_nonzero(nonfinite)}/{acquired.size} invalid cells; "
            f"first at tone {result.tone_indices[row]}, "
            f"DC={result.flux_values[row, point]} (native units), "
            f"frequency={result.frequencies[row, column]} MHz: "
            f"{acquired[row, column]!r}"
        )


def _plot_flux_maps_and_make_updater(
    plots: Plots, result: MuxFluxResult, flux_unit: str, tone_names: tuple[str, ...]
) -> Callable[[NDArray[np.complex128]], None]:
    """Display per-tone IQ contrast after subtracting each bias spectrum's median.

    result supplies tone-major (K,F) bias, (K,P) RF and (K,F,P) IQ arrays.
    tone_names maps original tone positions to labels. flux_unit labels bias
    without conversion. The returned plot-thread updater replaces the images
    and current/previous acquired spectra; it does not change data.
    """
    flux_values, frequencies = result.flux_values, result.frequencies
    unit = "device units" if flux_unit == "native" else flux_unit
    rf_label = "Qubit drive RF frequency (MHz)"
    iq_label = "Background-subtracted IQ magnitude (a.u.)"
    panels = []
    for index in result.tone_indices:
        fig, _ = plots.subplots(
            1, 2, num=f"flux_{index}", figsize=(8, 4.5), layout="constrained"
        )
        heat_ax, line_ax = fig.axes
        fig.suptitle(
            f"Mux two-tone · Local DC scan · {tone_names[index]} · IQ contrast"
        )
        heat_ax.set(xlabel=f"DC bias ({unit})", ylabel=rf_label)
        heat_ax.grid(False)
        line_ax.set(xlabel=rf_label, ylabel=iq_label)
        line_ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        heatmap = live_heatmap(heat_ax, uniform=False)
        fig.colorbar(heat_ax.images[0], ax=heat_ax, label=iq_label)
        line = live_line(line_ax, num_lines=2)
        line_ax.lines[0].set(
            marker="None", alpha=0.5, color="0.5", label="Previous bias"
        )
        line_ax.lines[1].set(label="Latest bias", marker="o", markersize=3, color="C0")
        line_ax.legend(fontsize=10)
        panels.append((fig, heatmap, line, line_ax.title))

    @plots.on_plot_thread
    def update(data: NDArray[np.complex128]):
        artists = []
        for row, (_, heatmap, line, slice_label) in enumerate(panels):
            # ADR-0069: remove each bias spectrum's IQ background before taking magnitude.
            contrast = np.abs(minus_background(data[row], axis=1))
            valid = np.flatnonzero(np.all(np.isfinite(data[row]), axis=1))
            history = np.full((2, frequencies.shape[1]), np.nan)
            for offset, bias_index in enumerate(valid[-2:][::-1]):
                history[-offset - 1] = contrast[bias_index]
            latest = (
                f"Bias = {flux_values[row, valid[-1]]:.5g} {unit}"
                if valid.size
                else "No acquired spectrum"
            )
            slice_label.set_text(latest)
            artists.extend(heatmap.set(flux_values[row], frequencies[row], contrast))
            artists.extend(line.set(frequencies[row], history))
            artists.append(slice_label)
        return artists

    update(result.signals)
    for fig, *_ in panels:
        fig.show()
    return update


class TwotoneFluxLocalExp(Experiment[MuxFluxResult, TwotoneFluxLocalCfg]):
    """Set DC, settle, acquire spectra and preserve completed flux points."""

    AXES_SPEC: ClassVar[GroupedAxesSpec[MuxFluxResult, TwotoneFluxLocalCfg]] = (
        mux_flux_spec(
            TwotoneFluxLocalCfg,
            tag="mux/twotone_flux_local",
            cfg_schema_version="5.0",
        )
    )

    def run(self, config: TwotoneFluxLocalCfg, *, context: RunContext) -> MuxFluxResult:
        """Run config without mutation using context's drivers, board and Stop.

        Return tone-major maps in absolute MHz and native DC coordinates.
        Cancelling an inner acquire leaves that point NaN and scans the next.
        Cancelling the Run scope stops the scan and retains completed points.
        Driver and acquisition errors propagate.
        Sources remain at the final written value. No output is enabled here.

        """
        # 1. Pair local-flux grids with each participant's quantized drive-frequency axis.
        cfg = config.model_copy(deep=True)
        indices = list(cfg.participants)
        devices = [cfg.tones[i].flux_dev for i in indices]
        for name in devices:
            check_flux_unit(context.devices[name].get_info(), cfg.flux_unit)
        frequencies = np.asarray(
            [
                sweep2array(
                    make_sweep(
                        cfg.tones[i].require_freq().start,
                        cfg.tones[i].require_freq().stop,
                        cfg.expts,
                    ),
                    "freq",
                    {
                        "soccfg": context.soccfg,
                        "gen_ch": cfg.tones[i].require_drive().ch,
                        "mixer_freq": cfg.tones[i].require_drive().mixer_freq,
                    },
                )
                for i in indices
            ],
            dtype=np.float64,
        )
        flux_values = np.asarray(
            [
                np.linspace(
                    cfg.tones[i].require_flux().start,
                    cfg.tones[i].require_flux().stop,
                    cfg.flux_expts,
                )
                for i in indices
            ],
            dtype=np.float64,
        )
        signals = np.full(
            (len(indices), cfg.flux_expts, cfg.expts),
            np.nan + 1j * np.nan,
            dtype=np.complex128,
        )
        result = MuxFluxResult(
            np.asarray(indices, dtype=np.int64), flux_values, frequencies, signals
        )
        update = _plot_flux_maps_and_make_updater(
            context.plots, result, cfg.flux_unit, tuple(tone.name for tone in cfg.tones)
        )

        # Reset, start drives together, then read out after the longest branch.
        builder = TimelineBuilder(
            context.soccfg,
            reps=cfg.reps,
            initial_delay=cfg.initial_delay,
            relax_delay=cfg.relax_delay,
        )
        builder.add_loop("point", cfg.expts)
        if cfg.reset is not None:
            builder.emit(Reset("reset", cfg.reset))
        timelines = [f"drive_{i}" for i in indices]
        builder.fork("default", into=timelines)
        for index, timeline in zip(indices, timelines, strict=True):
            tone = cfg.tones[index]
            freq = sweep2param(
                "point",
                make_sweep(
                    tone.require_freq().start, tone.require_freq().stop, cfg.expts
                ),
            )
            builder.emit(
                Pulse(f"drive_{index}", tone.require_drive(), freq=freq),
                timeline=timeline,
            )
        builder.join(*timelines, into="default")
        readout = MuxReadout(
            "readout",
            cfg.readout.with_updates(tones=[cfg.readout.tones[i] for i in indices]),
        )
        builder.emit(readout)
        prog = builder.build()

        # Local lines advance together. Each point has its own acquire scope.
        # 2. Advance local lines together and reuse the hardware spectrum program after settling.
        for point in range(cfg.flux_expts):
            if context.scope.stop.is_set():
                break
            for row, name in enumerate(devices):
                set_flux(
                    context.devices[name],
                    float(flux_values[row, point]),
                    unit=cfg.flux_unit,
                    stop=context.scope.stop,
                )
            if context.scope.stop.event.wait(cfg.settle):
                break
            stopped = False

            def on_acquire_stop() -> None:
                nonlocal stopped
                stopped = True

            acquired = SignalBuffer((cfg.expts, len(indices)))
            with Schedule(acquired, scope=context.scope) as sched:
                sched.run_acquire(
                    prog,
                    context.soc,
                    rounds=cfg.rounds,
                    options=RunOptions(
                        converter=partial(averaged_mux_signal, prog, readout),
                        on_stop=on_acquire_stop,
                    ),
                )
            if stopped or context.scope.stop.is_set():
                continue
            iq = acquired.array.T
            _require_finite_spectrum(iq, point=point, result=result)
            signals[:, point, :] = iq
            update(signals)
        # 3. Keep only complete finite spectra; cancelled local-flux points remain NaN in the maps.
        return result
