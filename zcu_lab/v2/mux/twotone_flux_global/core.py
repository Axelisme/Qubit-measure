"""Global coil DC scan with mux twotone spectra."""

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
from zcu_tools.program.v2 import Reset, ResetCfg, SweepCfg, TimelineBuilder, sweep2param
from zcu_tools.program.v2.modules import MuxReadout, MuxReadoutCfg, Pulse

from zcu_lab.v2.support import live_heatmap, live_line
from zcu_lab.v2.support.device import check_flux_unit, set_flux
from zcu_lab.v2.support.mux import averaged_mux_signal
from zcu_lab.v2.support.mux_flux import MuxFluxResult, mux_flux_spec
from zcu_lab.v2.support.mux_sweep_cfg import (
    MuxDriveToneCfg,
    ToneRangeCfg,
    validate_mux_scan,
)


class MuxFreqToneCfg(MuxDriveToneCfg):
    """One named drive and its independent freq bounds in MHz."""

    freq: Annotated[ToneRangeCfg | None, FieldMeta(unit="MHz")] = None

    def require_freq(self) -> ToneRangeCfg:
        """Return active freq bounds, or raise ValueError naming this tone."""
        if self.freq is None:
            raise ValueError(f"{self.name}: active tone requires freq")
        return self.freq


class TwotoneFluxGlobalCfg(ConfigBase):
    """Simultaneous qubit spectra at each shared coil coordinate.

    reset is optional; readout and tones retain every original group position.
    Each tone supplies its fixed name, drive and absolute MHz frequency bounds.
    flux_dev names one connected DC driver; flux is its finite native-unit sweep.
    expts is the positive hardware frequency point count shared by all tones.
    participants selects unique original indices in acquisition/result order.
    flux_unit is A, V or native without conversion; settle is a finite
    nonnegative wait in seconds after each coil write. reps/rounds are positive
    averaging counts; initial_delay/relax_delay are finite nonnegative us.
    Selected tones need drive and frequency settings, with independent drive
    generators separate from readout. Invalid settings raise before any DC write.
    """

    reset: ResetCfg | None = None
    readout: MuxReadoutCfg
    tones: Annotated[list[MuxFreqToneCfg], FieldMeta(element_label="name")]
    flux_dev: DeviceName = Field(min_length=1)
    flux: SweepCfg
    expts: int = Field(
        strict=True,
        gt=0,
        title="Expts",
        description="Hardware frequency point count shared by all tones.",
    )
    participants: list[StrictInt]
    flux_unit: Literal["A", "V", "native"] = "A"
    settle: Annotated[float, FieldMeta(unit="s")] = Field(
        default=0.1, ge=0, allow_inf_nan=False
    )

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
    def validate_scan(self) -> "TwotoneFluxGlobalCfg":
        """Reject invalid participants, active drives, timing or DC coordinates."""
        validate_mux_scan(self, self.tones, self.participants, self.expts)
        return self


def _plot_flux_maps_and_make_updater(
    plots: Plots, result: MuxFluxResult, flux_unit: str
) -> Callable[[NDArray[np.complex128]], None]:
    """Show per-tone magnitude maps and return a stateful plot-thread updater.

    result supplies original tone indices, DC coordinates (K,F), MHz coordinates
    (K,P) and initial complex IQ (K,F,P). flux_unit labels DC without conversion.
    The returned callback accepts complete IQ arrays in that same order and
    replaces the heatmaps and up to ten recent frequency traces. It captures
    result's coordinate arrays but never writes result or the incoming IQ.
    plots owns the displayed figures and their lifetime; plotting errors propagate.
    """
    flux_values, frequencies = result.flux_values, result.frequencies
    figures, heatmaps, lines = [], [], []
    for index in result.tone_indices:
        fig, _ = plots.subplots(1, 2, num=f"flux_{index}")
        heat_ax, line_ax = fig.axes
        heat_ax.set(xlabel=f"DC value ({flux_unit})", ylabel="Frequency (MHz)")
        line_ax.set(xlabel="Frequency (MHz)", ylabel="Signal (a.u.)")
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


class TwotoneFluxGlobalExp(Experiment[MuxFluxResult, TwotoneFluxGlobalCfg]):
    """Set DC, settle, acquire spectra and preserve completed flux points."""

    AXES_SPEC: ClassVar[GroupedAxesSpec[MuxFluxResult, TwotoneFluxGlobalCfg]] = (
        mux_flux_spec(
            TwotoneFluxGlobalCfg,
            tag="mux/twotone_flux_global",
            cfg_schema_version="5.0",
        )
    )

    def run(
        self, config: TwotoneFluxGlobalCfg, *, context: RunContext
    ) -> MuxFluxResult:
        """Run config without mutation using context's drivers, board and Stop.

        At each coil point, apply optional reset, simultaneous drive sweeps,
        then common mux readout. Return tone-major maps in absolute MHz and
        native DC coordinates. Cancelling an inner acquire leaves that point
        NaN and scans the next. Cancelling the Run scope stops the scan and
        retains completed points. Driver and acquisition errors propagate.
        Sources remain at the final written value. No output is enabled here.

        """
        # 1. Pair one shared-coil grid with each participant's quantized drive-frequency axis.
        cfg = config.model_copy(deep=True)
        indices = list(cfg.participants)
        device = context.devices[cfg.flux_dev]
        check_flux_unit(device.get_info(), cfg.flux_unit)
        sweeps = [
            make_sweep(
                cfg.tones[i].require_freq().start,
                cfg.tones[i].require_freq().stop,
                cfg.expts,
            )
            for i in indices
        ]
        frequencies = np.asarray(
            [
                sweep2array(
                    sweep,
                    "freq",
                    {
                        "soccfg": context.soccfg,
                        "gen_ch": cfg.tones[i].require_drive().ch,
                        "mixer_freq": cfg.tones[i].require_drive().mixer_freq,
                    },
                )
                for i, sweep in zip(indices, sweeps, strict=True)
            ],
            dtype=np.float64,
        )
        values = np.linspace(cfg.flux.start, cfg.flux.stop, cfg.flux.expts)
        flux_values = np.tile(values, (len(indices), 1))
        signals = np.full(
            (len(indices), len(values), cfg.expts),
            np.nan + 1j * np.nan,
            dtype=np.complex128,
        )
        result = MuxFluxResult(
            np.asarray(indices, dtype=np.int64), flux_values, frequencies, signals
        )

        update = _plot_flux_maps_and_make_updater(context.plots, result, cfg.flux_unit)

        # Keep the register-loop spectrum here, not in another experiment's run.
        readout = cfg.readout.with_updates(
            tones=[cfg.readout.tones[i] for i in indices]
        )
        # 2. Settle the shared coil, then sweep simultaneous drives in hardware before mux readout.
        for point, value in enumerate(values):
            if context.scope.stop.is_set():
                break
            set_flux(device, float(value), unit=cfg.flux_unit, stop=context.scope.stop)
            if context.scope.stop.event.wait(cfg.settle):
                break
            stopped = False

            def on_acquire_stop() -> None:
                nonlocal stopped
                stopped = True

            acquired = SignalBuffer((cfg.expts, len(indices)))
            with Schedule(acquired, scope=context.scope) as sched:
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
                for i, sweep, timeline in zip(indices, sweeps, timelines, strict=True):
                    builder.emit(
                        Pulse(
                            f"drive_{i}",
                            cfg.tones[i].require_drive(),
                            freq=sweep2param("point", sweep),
                        ),
                        timeline=timeline,
                    )
                builder.join(*timelines, into="default")
                readout_module = MuxReadout("readout", readout)
                builder.emit(readout_module)
                prog = builder.build()
                _ = sched.run_acquire(
                    prog,
                    context.soc,
                    rounds=cfg.rounds,
                    options=RunOptions(
                        converter=partial(averaged_mux_signal, prog, readout_module),
                        on_stop=on_acquire_stop,
                    ),
                )
            if stopped or context.scope.stop.is_set():
                continue
            spectrum = acquired.data.T
            nonfinite = ~np.isfinite(spectrum)
            if nonfinite.any():
                row, column = np.argwhere(nonfinite)[0]
                raise ValueError(
                    f"Nonfinite inner IQ at flux point {point} (zero-based): "
                    f"{np.count_nonzero(nonfinite)}/{spectrum.size} invalid cells; "
                    f"first at tone {indices[row]}, "
                    f"DC={flux_values[row, point]} (native units), "
                    f"frequency={frequencies[row, column]} MHz: "
                    f"{spectrum[row, column]!r}"
                )
            signals[:, point, :] = spectrum
            update(signals)
        # 3. Keep only complete finite spectra; cancelled coil points remain NaN in the maps.
        return result
