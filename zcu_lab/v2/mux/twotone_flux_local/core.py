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

from zcu_lab.v2.support import live_heatmap, live_line
from zcu_lab.v2.support.device import check_flux_unit, set_flux
from zcu_lab.v2.support.mux import averaged_mux_signal
from zcu_lab.v2.support.mux_flux import MuxFluxResult, mux_flux_spec
from zcu_lab.v2.support.mux_sweep_cfg import (
    MuxDriveToneCfg,
    ToneRangeCfg,
    require_tone_value,
    validate_mux_scan,
)


class TwotoneFluxLocalToneCfg(MuxDriveToneCfg):
    """One original qubit's drive, MHz range and local DC source/range.

    name, drive and freq retain the spectrum tone's meaning. flux_dev names a
    connected DC source even for inactive tones. flux gives finite native-unit
    bounds, required only when selected. All selected lines advance together
    using the parent flux_expts, not as a Cartesian product.
    """

    freq: Annotated[ToneRangeCfg | None, FieldMeta(unit="MHz")] = None
    flux_dev: DeviceName = Field(min_length=1)
    flux: ToneRangeCfg | None = None

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

    reset: ResetCfg | None = None
    readout: MuxReadoutCfg
    tones: Annotated[list[TwotoneFluxLocalToneCfg], FieldMeta(element_label="name")]
    flux_expts: int = Field(default=101, strict=True, gt=0)
    expts: int = Field(
        strict=True,
        gt=0,
        description="Hardware frequency point count shared by all tones.",
    )
    participants: list[StrictInt]
    flux_unit: Literal["A", "V", "native"] = "A"
    settle: Annotated[float, FieldMeta(unit="s")] = Field(
        default=0.1, ge=0, allow_inf_nan=False
    )

    reps: Annotated[
        int, FieldMeta(label="Reps", description="Number of repetitions per round.")
    ] = 1
    rounds: Annotated[
        int,
        FieldMeta(label="Rounds", description="Number of software acquisition rounds."),
    ] = 1
    initial_delay: Annotated[
        float,
        FieldMeta(
            label="Initial delay",
            unit="us",
            description="Delay before the first repetition.",
        ),
    ] = 1.0
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relax delay",
            unit="us",
            decimals=3,
            description="Relaxation delay after each repetition.",
        ),
    ] = 1.0

    @model_validator(mode="after")
    def validate_scan(self) -> TwotoneFluxLocalCfg:
        """Reject invalid selected drives, duplicate sources and inconsistent ranges."""
        validate_mux_scan(self, self.tones, self.participants, self.expts)
        devices = [self.tones[index].flux_dev for index in self.participants]
        if len(set(devices)) != len(devices):
            raise ValueError("local flux lines must use distinct devices")
        for index in self.participants:
            tone = self.tones[index]
            require_tone_value(tone.freq, ("tones", index, "freq"))
            require_tone_value(tone.flux, ("tones", index, "flux"))
            make_sweep(
                tone.require_flux().start, tone.require_flux().stop, self.flux_expts
            )
        return self


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
        cfg = TwotoneFluxLocalCfg.model_validate(config.model_dump())
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
        update = _make_flux_update(context.plots, result, cfg.flux_unit)

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
        return result


def _require_finite_spectrum(
    acquired: NDArray[np.complex128], *, point: int, result: MuxFluxResult
) -> None:
    """Report the first invalid completed IQ cell with its planned coordinates."""
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


def _make_flux_update(
    plots: Plots, result: MuxFluxResult, flux_unit: str
) -> Callable[[NDArray[np.complex128]], None]:
    """Publish planned axes and update each tone's magnitude map and line history."""
    flux_values, frequencies = result.flux_values, result.frequencies
    figures, heatmaps, lines = [], [], []
    for index in result.tone_indices:
        fig, _ = plots.subplots(1, 2, num=f"flux_{index}")
        heat_ax, line_ax = fig.axes
        heat_ax.set(xlabel=f"DC value ({flux_unit})", ylabel="Frequency (MHz)")
        line_ax.set_xlabel("Frequency (MHz)")
        line_ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        line_ax.grid()
        heatmaps.append(live_heatmap(heat_ax, uniform=False))
        lines.append(live_line(line_ax, num_lines=10))
        for artist in line_ax.lines[:-1]:
            artist.set(marker="None", alpha=0.3, color="red")
        line_ax.lines[-1].set(label="current line", marker=".", color="C0")
        line_ax.legend()
        figures.append(fig)

    @plots.on_plot_thread
    def update(data: NDArray[np.complex128]):
        artists = []
        for row, (heatmap, line) in enumerate(zip(heatmaps, lines, strict=True)):
            real = np.abs(data[row])
            valid = np.flatnonzero(~np.all(np.isnan(real), axis=1))
            current = int(valid[-1]) if valid.size else -1
            history = np.full((10, frequencies.shape[1]), np.nan)
            for offset in range(min(10, current + 1)):
                history[-offset - 1] = real[current - offset]
            artists.extend(heatmap.set(flux_values[row], frequencies[row], real))
            artists.extend(line.set(frequencies[row], history))
        return artists

    update(result.signals)
    for fig in figures:
        fig.show()
    return update
