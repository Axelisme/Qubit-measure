"""Tone-major mux flux results and native data mapping."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment import (
    GroupedAxesSpec,
    GroupedLoadData,
    VariableAxisSpec,
    VariableSpec,
    VariableZSpec,
)


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


def mux_flux_spec[C: ConfigBase](
    cfg_type: type[C], *, tag: str, cfg_schema_version: str
) -> GroupedAxesSpec[MuxFluxResult, C]:
    """Declare native tone/flux/frequency grids, converting MHz to Hz on disk.

    cfg_type and tag identify one concrete experiment. cfg_schema_version is
    its persisted cfg version (for example "4.0"), passed to GroupedAxesSpec
    without conversion or compatibility handling. All arrays retain full planned
    axes and NaN gaps after Stop. Loading validates shapes and axes; malformed
    data raises ValueError. An incompatible cfg version leaves cfg unavailable
    with a cfg_issue while preserving valid data. No library or alias is read.
    """
    tone = VariableAxisSpec(
        label="Tone", unit="", field_name="tone_indices", dtype=np.int64
    )
    point = VariableAxisSpec.generated_arange(label="Point", unit="", dtype=np.int64)
    flux = VariableAxisSpec.generated_arange(
        label="Flux point", unit="", dtype=np.int64
    )

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
        cfg_schema_version=cfg_schema_version,
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
