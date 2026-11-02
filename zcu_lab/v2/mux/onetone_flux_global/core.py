"""Global coil DC scan with mux onetone spectra."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from functools import partial
from typing import Annotated, ClassVar, Literal

import numpy as np
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray
from pydantic import Field, StrictInt, field_validator, model_validator
from zcu_tools.cfg_model import ConfigBase, DeviceName, FieldMeta
from zcu_tools.experiment import Experiment
from zcu_tools.experiment.axes_spec import GroupedAxesSpec
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.v2.runtime import RunOptions, Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import Reset, SweepCfg, TimelineBuilder
from zcu_tools.program.v2.modules import MuxReadout

from zcu_lab.v2.support import live_heatmap, live_line
from zcu_lab.v2.support.device import check_flux_unit, set_flux
from zcu_lab.v2.support.mux import averaged_mux_signal
from zcu_lab.v2.support.mux_flux import MuxFluxResult, mux_flux_spec
from zcu_lab.v2.support.mux_sweep_cfg import validate_participants


class OnetoneFluxToneCfg(ConfigBase):
    """One original readout position's unique, nonempty qubit name."""

    name: Annotated[str, FieldMeta(readonly=True)] = Field(min_length=1)


class OnetoneFluxGlobalCfg(ConfigBase):
    """Acquire common-offset readout spectra along one shared DC sweep.

    reset optionally precedes the full mux readout. tones names every original
    readout position, including inactive tones. flux_dev names the connected
    coil; flux is its finite native-unit SweepCfg. df is a finite nominal MHz
    offset SweepCfg shared by all readout tones, not an absolute frequency.
    participants is a nonempty unique list of original tone indices in result
    row order. flux_unit is A, V or native, without conversion. settle is a
    finite nonnegative wait in seconds after each DC write. reps and rounds
    control inner averaging; initial_delay and relax_delay are in microseconds.
    Invalid identities, selections or nonfinite scans raise validation errors.
    """

    reset: Reset.Cfg | None = None
    readout: MuxReadout.Cfg
    tones: Annotated[list[OnetoneFluxToneCfg], FieldMeta(element_label="name")]

    flux_dev: DeviceName = Field(min_length=1)
    flux: SweepCfg
    df: Annotated[SweepCfg, FieldMeta(label="Frequency offset", unit="MHz")]
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

    @field_validator("df")
    @classmethod
    def validate_finite_df(cls, value: SweepCfg) -> SweepCfg:
        """Reject nonfinite nominal offsets before any DC write."""
        if not all(np.isfinite(v) for v in (value.start, value.stop, value.step)):
            raise ValueError("df start, stop and step must be finite MHz values")
        return value

    @model_validator(mode="after")
    def validate_scan(self) -> OnetoneFluxGlobalCfg:
        """Reject mismatched identities, invalid indices and nonfinite scans."""
        names = [tone.name for tone in self.tones]
        if len(names) != len(self.readout.tones) or len(set(names)) != len(names):
            raise ValueError(
                "tones must name every original readout position exactly once"
            )
        validate_participants(names, self.participants)
        if not all(
            np.isfinite(v) for v in (self.flux.start, self.flux.stop, self.flux.step)
        ):
            raise ValueError("flux coordinates must be finite")
        return self


class OnetoneFluxGlobalExp(Experiment[MuxFluxResult, OnetoneFluxGlobalCfg]):
    """Set DC, settle, acquire spectra and preserve completed flux points."""

    AXES_SPEC: ClassVar[GroupedAxesSpec[MuxFluxResult, OnetoneFluxGlobalCfg]] = (
        mux_flux_spec(
            OnetoneFluxGlobalCfg,
            tag="mux/onetone_flux_global",
            cfg_schema_version="5.0",
        )
    )

    def run(
        self, config: OnetoneFluxGlobalCfg, *, context: RunContext
    ) -> MuxFluxResult:
        """Run config without mutation using context's drivers, board and Stop.

        Return original tone indices, float64 DC/frequency coordinates and
        complex128 IQ (participants, flux points, df points). Frequencies are
        each original readout frequency plus nominal df, in absolute MHz.
        Cancelling an inner acquire leaves that point NaN and scans the next.
        Cancelling the Run scope stops the scan and retains completed points.
        Missing devices raise KeyError; unit mismatch raises ValueError and
        unsupported drivers raise TypeError. Driver/acquisition errors propagate.
        Sources remain at the final written value. No output is enabled here.
        """
        # 1. Keep original tone identities and the complete planned axes after Stop.
        cfg = deepcopy(config)
        indices = cfg.participants
        device = context.devices[cfg.flux_dev]
        check_flux_unit(device.get_info(), cfg.flux_unit)
        dfs = sweep2array(cfg.df, "none")
        values = np.linspace(cfg.flux.start, cfg.flux.stop, cfg.flux.expts)
        flux_values = np.tile(values, (len(indices), 1))
        frequencies = np.asarray([cfg.readout.tones[i].freq + dfs for i in indices])
        signals = np.full(
            (len(indices), len(values), len(dfs)),
            np.nan + 1j * np.nan,
            dtype=np.complex128,
        )
        result = MuxFluxResult(
            np.asarray(indices, dtype=np.int64), flux_values, frequencies, signals
        )
        update = _make_flux_update(context.plots, result, cfg.flux_unit)
        active_readout = cfg.readout.with_updates(
            tones=[cfg.readout.tones[i] for i in indices]
        )

        # 2. Set the shared coil, settle, then scan common offsets on the host.
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

            buffer = SignalBuffer((len(dfs), len(indices)))
            with Schedule(buffer, scope=context.scope) as sched:
                for df, step in sched.scan("df", dfs.tolist()):
                    readout = active_readout.with_updates(
                        tones=[
                            tone.with_updates(freq=tone.freq + df)
                            for tone in active_readout.tones
                        ]
                    )
                    # Reset precedes the full selected mux readout at every offset.
                    builder = TimelineBuilder(
                        context.soccfg,
                        reps=cfg.reps,
                        initial_delay=cfg.initial_delay,
                        relax_delay=cfg.relax_delay,
                    )
                    if cfg.reset is not None:
                        builder.emit(Reset("reset", cfg.reset))
                    readout_module = MuxReadout("readout", readout)
                    builder.emit(readout_module)
                    prog = builder.build()
                    _ = step.run_acquire(
                        prog,
                        context.soc,
                        rounds=cfg.rounds,
                        options=RunOptions(
                            converter=partial(
                                averaged_mux_signal, prog, readout_module
                            ),
                            on_stop=on_acquire_stop,
                        ),
                    )

            # 3. Publish only complete spectra; an acquire-only Stop skips this point.
            if stopped or context.scope.stop.is_set():
                continue
            acquired = buffer.array.T
            nonfinite = ~np.isfinite(acquired)
            if nonfinite.any():
                row, column = np.argwhere(nonfinite)[0]
                raise ValueError(
                    f"Nonfinite inner IQ at flux point {point} (zero-based): "
                    f"{np.count_nonzero(nonfinite)}/{acquired.size} invalid cells; "
                    f"first at tone {indices[row]}, "
                    f"DC={flux_values[row, point]} (native units), "
                    f"frequency={frequencies[row, column]} MHz: "
                    f"{acquired[row, column]!r}"
                )
            signals[:, point, :] = acquired
            update(signals)
        return result


def _make_flux_update(
    plots: Plots, result: MuxFluxResult, flux_unit: str
) -> Callable[[NDArray[np.complex128]], None]:
    """Create per-tone magnitude maps and ten scan-history lines.

    Return a plot-thread updater accepting the complete tone-major IQ array.
    Publish the planned axes before the first acquisition; the caller retains
    ownership of Plots and its operation lifetime.
    """
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

    # Publish planned coordinates with no measured cells before the first point.
    update(result.signals)
    for fig in figures:
        fig.show()
    return update
