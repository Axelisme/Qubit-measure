"""Independent local line DC scan with mux onetone spectra."""

import math
from collections.abc import Callable
from copy import deepcopy
from functools import partial
from typing import Annotated, ClassVar, Literal, Self

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
from zcu_tools.program.v2 import Reset, SweepCfg, TimelineBuilder, TimelineProgramV2
from zcu_tools.program.v2.modules import MuxReadout

from zcu_lab.v2.support import live_heatmap, live_line
from zcu_lab.v2.support.device import check_flux_unit, set_flux
from zcu_lab.v2.support.mux import averaged_mux_signal
from zcu_lab.v2.support.mux_flux import MuxFluxResult, mux_flux_spec
from zcu_lab.v2.support.mux_sweep_cfg import (
    ToneRangeCfg,
    validate_participants,
)


class OnetoneFluxLocalToneCfg(ConfigBase):
    """One original qubit's identity and native-unit local DC range.

    name identifies its full readout position. flux_dev names a connected source.
    flux may be None for inactive tones; selected tones require bounds and use
    the shared flux_expts count. Lines advance together, not as a Cartesian grid.
    """

    name: Annotated[str, FieldMeta(readonly=True)] = Field(min_length=1)
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

    def require_flux(self) -> ToneRangeCfg:
        """Return this selected line's DC bounds; raise ValueError if absent."""
        if self.flux is None:
            raise ValueError(f"{self.name}: active tone requires DC sweep")
        return self.flux


class OnetoneFluxLocalCfg(ConfigBase):
    """Full mux readout and zipped independent local DC scans.

    reset optionally precedes readout. tones names every original readout
    position, including inactive tones. Selected flux_dev names must be distinct;
    flux_expts supplies their shared positive point count. df is the common
    nominal MHz offset added to each selected readout tone's base frequency.
    participants selects unique original indices in acquisition/result row order.
    flux_unit is the driver's native A/V unit or native, with no conversion.
    settle waits in seconds after all DC writes. reps and rounds count averaging;
    initial_delay and relax_delay are program delays in us.
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
    tones: Annotated[list[OnetoneFluxLocalToneCfg], FieldMeta(element_label="name")]

    flux_expts: Annotated[int, FieldMeta(label="Flux points", unit="count")] = Field(
        default=101, strict=True, gt=0
    )
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
        FieldMeta(description="Original tone indices in acquisition/result row order."),
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
            label="Reps", unit="count", description="Enter the repetitions per round."
        ),
    ]
    rounds: Annotated[
        int,
        FieldMeta(
            label="Rounds",
            unit="count",
            description="Enter the software averaging rounds.",
        ),
    ] = 1
    initial_delay: Annotated[
        float, FieldMeta(unit="µs", description="Delay before the first repetition.")
    ] = 1.0
    relax_delay: Annotated[
        float,
        FieldMeta(unit="µs", decimals=3, description="Delay after each repetition."),
    ]

    @model_validator(mode="after")
    def validate_lines(self) -> Self:
        """Require full unique identities and distinct valid selected DC ranges.

        Duplicate devices and invalid identities raise before acquisition.
        Selected bounds are consumed by run; inactive ranges stay optional.

        """
        names = [tone.name for tone in self.tones]
        if len(names) != len(self.readout.tones) or len(set(names)) != len(names):
            raise ValueError(
                "tones must name every original readout position exactly once"
            )
        validate_participants(names, self.participants)
        devices = [self.tones[index].flux_dev for index in self.participants]
        if len(set(devices)) != len(devices):
            raise ValueError("local flux lines must use distinct devices")
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
            f"Mux one-tone · Local DC scan · {tone_names[index]} · Readout amplitude"
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


def _require_finite_spectrum(
    acquired: NDArray[np.complex128], point: int, result: MuxFluxResult
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


class OnetoneFluxLocalExp(Experiment[MuxFluxResult, OnetoneFluxLocalCfg]):
    """Set DC, settle, acquire spectra and preserve completed flux points."""

    AXES_SPEC: ClassVar[GroupedAxesSpec[MuxFluxResult, OnetoneFluxLocalCfg]] = (
        mux_flux_spec(
            OnetoneFluxLocalCfg,
            tag="mux/onetone_flux_local",
            cfg_schema_version="5.0",
        )
    )

    def run(self, config: OnetoneFluxLocalCfg, *, context: RunContext) -> MuxFluxResult:
        """Run config without mutation using context's drivers, board and Stop.

        Return tone-major maps in absolute MHz and native DC coordinates.
        Cancelling an inner acquire leaves that point NaN and scans the next.
        Cancelling the Run scope stops the scan and retains completed points.
        Driver and acquisition errors propagate. Invalid inner spectrum settings
        raise before DC writes. Sources retain their last value; no output is enabled.

        """
        # Retain the old inner cfg's finite-offset rejection before any DC write.
        # 1. Check offset settings and DC units before writing the paired local-flux grids.
        if not all(
            math.isfinite(value)
            for value in (config.df.start, config.df.stop, config.df.step)
        ):
            raise ValueError("df start, stop and step must be finite MHz values")
        cfg = deepcopy(config)
        indices = cfg.participants
        flux_values = np.asarray(
            [
                np.linspace(
                    config.tones[i].require_flux().start,
                    config.tones[i].require_flux().stop,
                    config.flux_expts,
                )
                for i in config.participants
            ],
            dtype=np.float64,
        )
        devices = [cfg.tones[i].flux_dev for i in indices]
        for name in devices:
            check_flux_unit(context.devices[name].get_info(), cfg.flux_unit)
        dfs = sweep2array(cfg.df, "none")
        frequencies = np.asarray(
            [cfg.readout.tones[i].freq + dfs for i in indices], dtype=np.float64
        )
        signals = np.full(
            (len(indices), cfg.flux_expts, len(dfs)),
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

        # 2. Settle each DC point and reuse the program for each host frequency offset.
        # Reusing each whole program saves recompilation, but each offset's program
        # retains its latest acquire buffers until its next acquire or this Run ends.
        programs: dict[float, tuple[TimelineProgramV2, MuxReadout]] = {}
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

            # Each flux point owns an offset scan. An acquire-only stop discards
            # this entire point, but still lets the next flux point run.
            stopped = False

            def on_acquire_stop() -> None:
                nonlocal stopped
                stopped = True

            spectrum = SignalBuffer((len(dfs), len(indices)))
            with Schedule(spectrum, scope=context.scope) as sched:
                for df, step in sched.scan("df", dfs.tolist()):
                    # DC affects acquisition, not the compiled reset/readout sequence.
                    if df not in programs:
                        readout = active_readout.with_updates(
                            tones=[
                                tone.with_updates(freq=tone.freq + df)
                                for tone in active_readout.tones
                            ]
                        )
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
                        programs[df] = builder.build(), readout_module
                    prog, readout_module = programs[df]
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
            if stopped or context.scope.stop.is_set():
                continue
            acquired = spectrum.array.T
            _require_finite_spectrum(acquired, point, result)
            signals[:, point, :] = acquired
            update(signals)
        # 3. Keep complete spectra in the flux maps; cancelled points remain NaN on the planned axes.
        return result
