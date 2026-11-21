"""Global coil DC scan with mux onetone spectra."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
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

    reset: Annotated[
        Reset.Cfg | None,
        FieldMeta(
            label="Reset",
            description="Choose the common preparation reset; leave blank to omit it.",
        ),
    ] = None
    readout: Annotated[
        MuxReadout.Cfg,
        FieldMeta(
            label="Readout",
            description="Enter the multiplexed readout tone and ADC settings for the resonator scan.",
        ),
    ]
    tones: Annotated[list[OnetoneFluxToneCfg], FieldMeta(element_label="name")]

    flux_dev: Annotated[
        DeviceName,
        FieldMeta(
            label="Flux source",
            description="Enter the DC source that sets the shared flux bias.",
        ),
    ] = Field(min_length=1)
    flux: Annotated[
        SweepCfg,
        FieldMeta(
            label="DC bias",
            description="Enter the bias sweep endpoints in the selected Flux unit.",
        ),
    ]
    df: Annotated[
        SweepCfg,
        FieldMeta(
            label="Frequency offset",
            unit="MHz",
            description="Enter the common RF offset from every tone's base readout frequency in MHz.",
        ),
    ]
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
    ]
    rounds: Annotated[
        int,
        FieldMeta(
            label="Rounds",
            unit="count",
            description="Number of software acquisition rounds.",
        ),
    ] = 1
    initial_delay: Annotated[
        float,
        FieldMeta(
            label="Initial delay",
            unit="µs",
            description="Delay before the first repetition.",
        ),
    ] = 1.0
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relax delay",
            unit="µs",
            decimals=3,
            description="Relaxation delay after each repetition.",
        ),
    ]

    @model_validator(mode="after")
    def validate_scan(self) -> OnetoneFluxGlobalCfg:
        """Reject mismatched identities and invalid original indices."""
        names = [tone.name for tone in self.tones]
        if len(names) != len(self.readout.tones) or len(set(names)) != len(names):
            raise ValueError(
                "tones must name every original readout position exactly once"
            )
        validate_participants(names, self.participants)
        return self


def _plot_flux_maps_and_make_updater(
    plots: Plots, result: MuxFluxResult, flux_unit: str, tone_names: tuple[str, ...]
) -> Callable[[NDArray[np.complex128]], None]:
    """Display the per-tone IQ magnitude maps and latest spectra.

    result supplies tone-major (K,F) bias, (K,P) RF and (K,F,P) IQ arrays.
    tone_names maps original tone positions to labels. flux_unit labels bias
    without conversion. The returned plot-thread updater replaces the images
    and current/previous acquired spectra; it does not change data.
    """
    flux_values, frequencies = result.flux_values, result.frequencies
    unit = "device units" if flux_unit == "native" else flux_unit
    rf_label = "Nominal readout RF frequency (MHz)"
    iq_label = "|Mean IQ| (a.u.)"
    panels = []
    for index in result.tone_indices:
        fig, _ = plots.subplots(
            1, 2, num=f"flux_{index}", figsize=(8, 4.5), layout="constrained"
        )
        heat_ax, line_ax = fig.axes
        fig.suptitle(
            f"Mux one-tone · Global DC scan · {tone_names[index]} · Readout amplitude"
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
        panels.append((fig, heatmap, line, heat_ax.title, line_ax.title))

    @plots.on_plot_thread
    def update(data: NDArray[np.complex128]):
        artists = []
        for row, (_, heatmap, line, progress, slice_label) in enumerate(panels):
            real = np.abs(data[row])
            valid = np.flatnonzero(np.all(np.isfinite(real), axis=1))
            history = np.full((2, frequencies.shape[1]), np.nan)
            for offset, bias_index in enumerate(valid[-2:][::-1]):
                history[-offset - 1] = real[bias_index]
            progress.set_text(
                f"Acquired bias points: {valid.size} / {flux_values.shape[1]}"
            )
            latest = (
                f"Bias = {flux_values[row, valid[-1]]:.5g} {unit}"
                if valid.size
                else "No acquired spectrum"
            )
            slice_label.set_text(latest)
            artists.extend(heatmap.set(flux_values[row], frequencies[row], real))
            artists.extend(line.set(frequencies[row], history))
            artists.extend((progress, slice_label))
        return artists

    update(result.signals)
    for fig, *_ in panels:
        fig.show()
    return update


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
        update = _plot_flux_maps_and_make_updater(
            context.plots, result, cfg.flux_unit, tuple(tone.name for tone in cfg.tones)
        )
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
