"""Native grouped mapping for independent zipped coordinates and averaged IQ."""

import numpy as np
from zcu_tools.cfg_model import ConfigBase
from zcu_tools.datafile import DataVariable
from zcu_tools.experiment import (
    GroupedAxesSpec,
    GroupedLoadData,
    VariableAxisSpec,
    VariableSpec,
    VariableZSpec,
)

from zcu_lab.v2.mux._moments import (
    iq_moments_variable,
    restore_iq_moments,
    validate_iq_moments,
)

from .mux import MuxRabiResult, MuxSweepResult, validate_mux_result

_ZIP_AXES = (
    VariableAxisSpec.generated_arange(label="Point", unit="", dtype=np.int64),
    VariableAxisSpec(label="Tone", unit="", field_name="tone_indices", dtype=np.int64),
)


def _sweep_variables(label: str, unit: str, scale: float) -> tuple[VariableSpec, ...]:
    return (
        VariableSpec(
            "coordinates",
            _ZIP_AXES,
            VariableZSpec("coordinates", label=label, unit=unit, scale=scale),
        ),
        VariableSpec(
            "signals",
            _ZIP_AXES,
            VariableZSpec("signals", label="Signal", unit="a.u.", dtype=np.complex128),
        ),
    )


def _sweep_result[T: ConfigBase](data: GroupedLoadData[T]) -> MuxSweepResult:
    coordinates = data.variable("coordinates")
    signals = data.variable("signals")
    if any(
        not np.array_equal(a, b)
        for a, b in zip(coordinates.axes, signals.axes, strict=True)
    ):
        raise ValueError("Mux coordinate and signal axes must match")
    expected = np.arange(coordinates.z.shape[1], dtype=np.int64)
    if not np.array_equal(coordinates.axes[0], expected):
        raise ValueError("Mux Point axis must preserve original zip indices")
    return MuxSweepResult(
        coordinates.axes[1].astype(np.int64),
        coordinates.z.astype(np.float64),
        signals.z.astype(np.complex128),
    )


def mux_sweep_spec[T: ConfigBase](
    cfg_type: type[T],
    *,
    tag: str,
    coordinate_label: str,
    unit: str,
    scale: float,
) -> GroupedAxesSpec[MuxSweepResult, T]:
    """Map each tone's scan coordinates and mean IQ to matching saved grids.

    cfg_type is the leaf's typed cfg, tag its unique persistence identity.
    coordinate_label/unit are disk metadata; scale converts memory coordinates
    to disk units, e.g. MHz to Hz. Each variable is (K,P), with inner-first Point
    then Tone axes. No I/O occurs. Invalid declarations raise from GroupedAxesSpec.
    The loader checks matching axes and result contracts without consulting cfg.
    """
    return GroupedAxesSpec(
        variables=_sweep_variables(coordinate_label, unit, scale),
        result_type=MuxSweepResult,
        cfg_type=cfg_type,
        tag=tag,
        result_builder=_sweep_result,
        result_validator=validate_mux_result,
        cfg_schema_version="4.0",
    )


def mux_rabi_spec[T: ConfigBase](
    cfg_type: type[T],
    *,
    tag: str,
    coordinate_label: str,
    unit: str,
    scale: float,
) -> GroupedAxesSpec[MuxRabiResult, T]:
    """Map the zipped mux Rabi grids and optional Native-only IQ moments.

    cfg_type and tag identify the experiment. coordinate_label and unit name
    the saved drive coordinate; scale converts it from memory to disk units.
    Labber exports only coordinates/signals on their common grid. Native
    restore accepts absent moments and rejects mismatched or invalid moments.
    """

    def build(data: GroupedLoadData[T]) -> MuxRabiResult:
        mean = _sweep_result(data)
        moments = restore_iq_moments(
            data.variables.get(DataVariable("iq_moments")), data.variable("signals")
        )
        return MuxRabiResult(mean.tone_indices, mean.coordinates, mean.signals, moments)

    def validate(result: MuxRabiResult) -> None:
        validate_mux_result(result)
        validate_iq_moments(result.iq_moments, result.signals)

    return GroupedAxesSpec(
        variables=(
            *_sweep_variables(coordinate_label, unit, scale),
            iq_moments_variable(_ZIP_AXES),
        ),
        result_type=MuxRabiResult,
        cfg_type=cfg_type,
        tag=tag,
        result_builder=build,
        result_validator=validate,
        cfg_schema_version="4.0",
        labber_variables=("coordinates", "signals"),
    )
