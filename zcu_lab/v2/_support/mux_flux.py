"""DC zip scans around the existing mux spectroscopy cores."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Annotated, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, model_validator
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import (
    GroupedAxesSpec,
    GroupedLoadData,
    VariableAxisSpec,
    VariableSpec,
    VariableZSpec,
)
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.utils.sweep import make_sweep
from zcu_tools.plotting.plots import NonPresentingHost, Plots
from zcu_tools.program.v2 import SweepCfg

from zcu_lab.v2._support.device import check_flux_unit, set_flux
from zcu_lab.v2._support.mux_sweep_cfg import ToneRangeCfg, participant_indices
from zcu_lab.v2.mux.onetone.core import MuxOnetoneCfg, MuxOnetoneExp
from zcu_lab.v2.mux.twotone_spectrum.core import (
    MuxFreqCfg,
    MuxFreqExp,
    prepare_frequency_scan,
)


class FluxFrequencyCfg(MuxFreqCfg):
    """Inner spectrum with full fixed identities; outer participants selects Run.

    participants retains all names for inner cfg validation and is read-only in
    the Form. run_flux_scan replaces it only on the detached acquisition copy.
    """

    participants: Annotated[
        list[str],
        FieldMeta(
            readonly=True,
            description="Full inner identity list; use outer participants to select acquisition",
        ),
    ]


class FluxIdentityCfg(ConfigBase):
    """Original readout position's qubit name; names are unique and nonempty."""

    name: Annotated[str, FieldMeta(readonly=True)] = Field(min_length=1)


class FluxLineCfg(FluxIdentityCfg):
    """One local DC source and its native-unit start/stop range.

    flux_dev names a connected driver. sweep advances with all other selected
    lines by point index, using the shared flux_expts, not a Cartesian product.
    """

    flux_dev: str = Field(min_length=1)
    sweep: ToneRangeCfg


class MuxFluxCfg[S: (MuxOnetoneCfg, MuxFreqCfg), T: FluxIdentityCfg](ExpCfgModel):
    """Full spectrum settings, original identities and ordered participants.

    spectrum retains the complete readout, not an active subset. tones has one
    identity per original readout position. participants is a nonempty unique
    selection in acquisition order. flux_unit is the driver's native unit, with
    no conversion. settle is the cancellable wait in seconds after all writes.
    """

    spectrum: S
    tones: Annotated[list[T], FieldMeta(element_label="name")]
    participants: list[str]
    flux_unit: Literal["A", "V", "native"] = "A"
    settle: Annotated[float, FieldMeta(unit="s")] = Field(
        default=0.1, ge=0, allow_inf_nan=False
    )

    @model_validator(mode="after")
    def validate_identities(self) -> MuxFluxCfg[S, T]:
        """Reject invalid full identities before projecting participant indices."""
        names = [tone.name for tone in self.tones]
        if len(names) != len(self.spectrum.modules.readout.tones) or len(
            set(names)
        ) != len(names):
            raise ValueError(
                "tones must name every original readout position exactly once"
            )
        participant_indices(names, self.participants)
        if isinstance(self.spectrum, MuxFreqCfg) and names != [
            t.name for t in self.spectrum.tones
        ]:
            raise ValueError("flux tone names must match the full spectrum tone order")
        return self


class GlobalFluxCfg[S: (MuxOnetoneCfg, MuxFreqCfg)](MuxFluxCfg[S, FluxIdentityCfg]):
    """One global coil source and one finite native-unit flux sweep."""

    flux_dev: str = Field(min_length=1)
    flux: SweepCfg

    @model_validator(mode="after")
    def validate_flux(self) -> GlobalFluxCfg[S]:
        """Reject nonfinite global device coordinates."""
        if not all(
            np.isfinite(v) for v in (self.flux.start, self.flux.stop, self.flux.step)
        ):
            raise ValueError("flux coordinates must be finite")
        return self


class LocalFluxCfg[S: (MuxOnetoneCfg, MuxFreqCfg)](MuxFluxCfg[S, FluxLineCfg]):
    """Independent local line ranges with one shared positive point count.

    Every full tone has a different source, even when it does not participate.
    Unselected lines are never written. Selected lines advance in a zip scan.
    """

    flux_expts: int = Field(default=101, strict=True, gt=0)

    @model_validator(mode="after")
    def validate_lines(self) -> LocalFluxCfg[S]:
        """Reject shared local drivers and ranges inconsistent with point count."""
        devices = [tone.flux_dev for tone in self.tones]
        if len(set(devices)) != len(devices):
            raise ValueError("local flux lines must use distinct devices")
        for tone in self.tones:
            make_sweep(tone.sweep.start, tone.sweep.stop, self.flux_expts)
        return self


@dataclass(frozen=True)
class MuxFluxResult:
    """Original indices and tone-major maps, including stopped scans.

    tone_indices is int64 (K,) in participant order. flux_values is float64
    (K,F), in native driver units. frequencies is float64 (K,P), absolute MHz.
    signals is complex128 (K,F,P); only completed flux points are published.
    Unfinished flux points remain NaN, including a cancelled inner spectrum.
    """

    tone_indices: NDArray[np.int64]
    flux_values: NDArray[np.float64]
    frequencies: NDArray[np.float64]
    signals: NDArray[np.complex128]


def run_flux_scan(
    cfg: GlobalFluxCfg[MuxOnetoneCfg]
    | LocalFluxCfg[MuxOnetoneCfg]
    | GlobalFluxCfg[MuxFreqCfg]
    | LocalFluxCfg[MuxFreqCfg],
    *,
    context: RunContext,
) -> MuxFluxResult:
    """Set selected DC sources, settle, then reuse the inner mux run per point.

    cfg is not mutated. context supplies connected drivers/board and the one
    Run scope. Validate every selected driver's unit before the first write.
    Global scans write the coil once per point; local scans zip selected lines.
    Stop during ramp, settle or inner acquisition discards that incomplete flux
    point and preserves earlier points. Driver/acquisition errors propagate.
    Sources retain their last setting; no automatic restoration occurs.
    """
    indices = participant_indices([tone.name for tone in cfg.tones], cfg.participants)
    if isinstance(cfg, GlobalFluxCfg):
        values = np.linspace(cfg.flux.start, cfg.flux.stop, cfg.flux.expts)
        flux_values = np.tile(values, (len(indices), 1))
        devices = [cfg.flux_dev]
    else:
        flux_values = np.asarray(
            [
                np.linspace(
                    cfg.tones[i].sweep.start, cfg.tones[i].sweep.stop, cfg.flux_expts
                )
                for i in indices
            ],
            dtype=np.float64,
        )
        devices = [cfg.tones[i].flux_dev for i in indices]
    for name in devices:
        check_flux_unit(context.devices[name].get_info(), cfg.flux_unit)

    spectrum = cfg.spectrum.model_copy(deep=True)
    if isinstance(spectrum, MuxOnetoneCfg):
        spectrum.active_tones = list(indices)
        dfs = np.linspace(
            spectrum.sweep.df.start, spectrum.sweep.df.stop, spectrum.sweep.df.expts
        )
        frequencies = np.asarray(
            [spectrum.modules.readout.tones[i].freq + dfs for i in indices]
        )
    else:
        spectrum.participants = list(cfg.participants)
        frequencies = prepare_frequency_scan(
            spectrum, soccfg=context.soccfg
        ).coordinates
    signals = np.full(
        (len(indices), flux_values.shape[1], frequencies.shape[1]),
        np.nan + 1j * np.nan,
        dtype=np.complex128,
    )
    viewers = [
        context.plots.liveplot_2d_with_line(
            f"flux_{i}",
            f"DC value ({cfg.flux_unit})",
            "Frequency (MHz)",
            line_axis=1,
            num_lines=10,
            uniform=False,
        )
        for i in indices
    ]
    for point in range(flux_values.shape[1]):
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
        acquired = _acquire_spectrum(spectrum, context=context)
        context.scope.stop.raise_if_error()
        if context.scope.stop.partial_result or not np.isfinite(acquired).all():
            break
        signals[:, point, :] = acquired
        for row, viewer in enumerate(viewers):
            viewer.update(flux_values[row], frequencies[row], np.abs(signals[row]))
    return MuxFluxResult(
        np.asarray(indices, dtype=np.int64), flux_values, frequencies, signals
    )


def _acquire_spectrum(
    spectrum: MuxOnetoneCfg | MuxFreqCfg, *, context: RunContext
) -> NDArray[np.complex128]:
    # Outer maps own presentation. Retire each inner producer before the next
    # flux point so repeated spectra do not reuse live axes or open extra tabs.
    plots = Plots(NonPresentingHost())
    inner = replace(context, plots=plots)
    try:
        if isinstance(spectrum, MuxOnetoneCfg):
            return MuxOnetoneExp().run(spectrum, context=inner).signals.T
        return MuxFreqExp().run(spectrum, context=inner).signals
    finally:
        plots.finish(present=False)
        plots.release()


def mux_flux_spec[C: ExpCfgModel](
    cfg_type: type[C], *, tag: str
) -> GroupedAxesSpec[MuxFluxResult, C]:
    """Declare native tone/flux/frequency grids, converting MHz to Hz on disk.

    cfg_type and tag identify one concrete experiment. All arrays retain full
    planned axes and NaN gaps after Stop. Loading validates shapes and axes;
    no current library or alias is consulted.
    """
    tone = VariableAxisSpec("Tone", "", field_name="tone_indices", dtype=np.int64)
    point = VariableAxisSpec.generated_arange("Point", "", dtype=np.int64)
    flux = VariableAxisSpec.generated_arange("Flux point", "", dtype=np.int64)

    def build(data: GroupedLoadData[C]) -> MuxFluxResult:
        values = data.variable("flux_values")
        frequencies = data.variable("frequencies")
        signals = data.variable("signals")
        if not np.array_equal(values.axes[-1], signals.axes[-1]) or not np.array_equal(
            frequencies.axes[-1], signals.axes[-1]
        ):
            raise ValueError("mux flux variables must share original tone indices")
        result = MuxFluxResult(
            signals.axes[-1].astype(np.int64),
            values.z.astype(np.float64),
            frequencies.z.astype(np.float64),
            signals.z.astype(np.complex128),
        )
        validate_flux_result(result)
        return result

    return GroupedAxesSpec(
        variables=(
            VariableSpec(
                variable="flux_values",
                axes=(flux, tone),
                z=VariableZSpec(
                    field_name="flux_values",
                    label="DC value",
                    unit="native",
                    dtype=np.float64,
                ),
            ),
            VariableSpec(
                variable="frequencies",
                axes=(point, tone),
                z=VariableZSpec(
                    field_name="frequencies",
                    label="Frequency",
                    unit="Hz",
                    scale=1e6,
                    dtype=np.float64,
                ),
            ),
            VariableSpec(
                variable="signals",
                axes=(point, flux, tone),
                z=VariableZSpec(
                    field_name="signals",
                    label="Signal",
                    unit="a.u.",
                    dtype=np.complex128,
                ),
            ),
        ),
        result_type=MuxFluxResult,
        cfg_type=cfg_type,
        tag=tag,
        result_builder=build,
        result_validator=validate_flux_result,
        cfg_schema_version="1.0",
        labber_layout="independent",
    )


def validate_flux_result(result: MuxFluxResult) -> None:
    """Reject malformed tone-major map shapes and nonfinite planned axes.

    NaN signals are allowed for incomplete flux points. Indices must be unique
    nonnegative integers. Coordinate axes must have at least one point.
    """
    indices = result.tone_indices
    if (
        indices.ndim != 1
        or indices.dtype.kind not in "iu"
        or len(indices) == 0
        or len(set(indices)) != len(indices)
        or np.any(indices < 0)
    ):
        raise ValueError("tone_indices must be unique nonnegative original indices")
    k = len(indices)
    if (
        result.flux_values.ndim != 2
        or result.frequencies.ndim != 2
        or result.flux_values.shape[0] != k
        or result.frequencies.shape[0] != k
    ):
        raise ValueError("flux_values and frequencies must be tone-major matrices")
    f, p = result.flux_values.shape[1], result.frequencies.shape[1]
    if (
        not f
        or not p
        or result.signals.shape != (k, f, p)
        or not np.isfinite(result.flux_values).all()
        or not np.isfinite(result.frequencies).all()
    ):
        raise ValueError("signals must match finite planned tone/flux/frequency axes")
