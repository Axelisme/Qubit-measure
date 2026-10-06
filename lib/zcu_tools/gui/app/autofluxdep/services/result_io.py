"""Typed Result <-> streaming Labber variable mapping for autofluxdep artifacts."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import ClassVar, Literal, overload

import numpy as np
from numpy.typing import NDArray

from zcu_tools.datafile import (
    Axis,
    DataVariable,
    LabberPayload,
    StreamingGroupedLabberWriter,
    StreamingLabberVariableSpec,
    load_grouped_labber_data,
)
from zcu_tools.gui.app.autofluxdep.results import (
    FrequencySweepProgressSummary,
    FrequencySweepResult,
    FrequencySweepRowSummary,
    ResultProgressSummary,
    ResultRowSummary,
    Sweep1DProgressSummary,
    Sweep1DRowSummary,
    Sweep2DProgressSummary,
    Sweep2DRowSummary,
    SweepResult1D,
    SweepResult2D,
    WorkflowResult,
    require_workflow_result,
)

VARIABLE_SIGNAL = DataVariable("signal")
VARIABLE_FIT_CURVE = DataVariable("fit_curve")
VARIABLE_FIT_FREQ = DataVariable("fit_freq")
VARIABLE_PREDICT_FREQ = DataVariable("predict_freq")
VARIABLE_SNR = DataVariable("snr")
VARIABLE_FIT_VALUE = DataVariable("fit_value")
VARIABLE_BEST_FREQ = DataVariable("best_freq")
VARIABLE_BEST_GAIN = DataVariable("best_gain")

_SpecBuilder = Callable[
    [str, str, object, str], tuple[StreamingLabberVariableSpec, ...]
]
_RowValueBuilder = Callable[
    [object, int], Mapping[DataVariable, NDArray[np.float64] | float]
]
_Loader = Callable[[Mapping[DataVariable, LabberPayload]], WorkflowResult]


@dataclass(frozen=True)
class _LoadedFrequencySweep:
    result_kind: ClassVar[Literal["qubit_freq"]] = "qubit_freq"
    flux: NDArray[np.float64]
    detune: NDArray[np.float64]
    signal: NDArray[np.float64]
    fit_curve: NDArray[np.float64]
    fit_freq: NDArray[np.float64]
    predict_freq: NDArray[np.float64]
    snr: NDArray[np.float64]


@dataclass(frozen=True)
class _LoadedSweep1D:
    result_kind: ClassVar[Literal["sweep1d"]] = "sweep1d"
    flux: NDArray[np.float64]
    x: NDArray[np.float64]
    signal: NDArray[np.float64]
    fit_curve: NDArray[np.float64]
    fit_value: NDArray[np.float64]
    snr: NDArray[np.float64]
    x_label: str


@dataclass(frozen=True)
class _LoadedSweep2D:
    result_kind: ClassVar[Literal["sweep2d"]] = "sweep2d"
    flux: NDArray[np.float64]
    freq: NDArray[np.float64]
    gain: NDArray[np.float64]
    signal: NDArray[np.float64]
    best_freq: NDArray[np.float64]
    best_gain: NDArray[np.float64]


@dataclass(frozen=True)
class _ResultDeclaration:
    kind: str
    variables: frozenset[DataVariable]
    summary_scalar_attrs: tuple[str, ...]
    last_fit_fields: tuple[tuple[str, str], ...]
    spec_builder: _SpecBuilder
    row_values: _RowValueBuilder
    loader: _Loader


def result_variable_specs(
    node_name: str,
    node_type: str,
    result: object,
    *,
    flux_unit: str = "",
) -> tuple[StreamingLabberVariableSpec, ...]:
    """Build variable schemas for one placed node\'s tagged workflow Result.

    node_name/node_type identify the placed node in stored attrs. result must
    satisfy its declared result_kind array contract. flux_unit labels its flux
    axis, or is empty when unspecified. Return ordered full-shape schemas;
    unknown declarations or invalid array fields/dtypes raise TypeError,
    and inconsistent shapes raise ValueError.
    """
    declaration = result_declaration(result)
    return declaration.spec_builder(node_name, node_type, result, flux_unit)


def write_result_row(
    writer: StreamingGroupedLabberWriter,
    node_name: str,
    node_type: str,
    result: object,
    flux_idx: int,
    *,
    timestamp: float | None = None,
) -> tuple[str, ...]:
    """Write all persisted variables for one flux row."""
    del node_name, node_type
    idx = int(flux_idx)
    values = _result_row_values(result, idx)
    for variable, row in values.items():
        writer.write_outer_slice(variable, idx, row, timestamp=timestamp)
    return tuple(str(variable) for variable in values)


def result_row_variable_names(result: object, flux_idx: int) -> tuple[str, ...]:
    """Return variables written for flux_idx without mutating result or storage.

    result must satisfy its tagged workflow Result array contract. flux_idx
    selects a row; unknown declarations or invalid fields/dtypes raise TypeError,
    inconsistent shapes raise ValueError, and an out-of-range row raises IndexError.
    """
    return tuple(
        str(variable) for variable in _result_row_values(result, int(flux_idx))
    )


def load_node_result(path: str, node_type: str) -> WorkflowResult:
    """Load a node HDF5 path into a framework-owned sweep data record.

    node_type is retained for caller context; variable sets determine representation.
    The returned record satisfies the same Protocol as the experiment's Result,
    including NaN rows. It does not allocate or construct an experiment class.
    Raise ValueError for unknown variables, inconsistent axes or data shapes;
    propagate file/IO errors. No archive or input data is mutated.
    """
    del node_type
    grouped = load_grouped_labber_data(path)
    variables = grouped.variables
    declaration = _declaration_for_variables(frozenset(variables))
    return declaration.loader(variables)


def read_result_row(
    path: str,
    node_type: str,
    flux_idx: int,
) -> Mapping[DataVariable, np.ndarray | float]:
    """Read one committed-or-nan Result row from a node HDF5 file."""
    result = load_node_result(path, node_type)
    values = _result_row_values(result, int(flux_idx))
    row: dict[DataVariable, np.ndarray | float] = {}
    for variable, value in values.items():
        array = np.asarray(value)
        if array.ndim == 0:
            row[variable] = float(array)
        else:
            row[variable] = array.copy()
    return row


def result_declaration(result_or_type: object) -> _ResultDeclaration:
    """Return the archive declaration named by an instance/class result_kind.

    Classes need only a supported literal kind; they need not allocate arrays.
    Instances must also satisfy the corresponding Protocol, float64 dtypes and
    axis/array shapes. TypeError rejects missing/unknown kind or invalid fields;
    ValueError rejects inconsistent shapes. No data is modified.
    """
    if isinstance(result_or_type, type):
        kind: object = getattr(result_or_type, "result_kind", None)
    else:
        kind = require_workflow_result(result_or_type).result_kind
    if isinstance(kind, str):
        for declaration in _RESULT_DECLARATIONS:
            if kind == declaration.kind:
                return declaration
    raise TypeError(f"unsupported autofluxdep Result kind {kind!r}")


@overload
def result_row_summary(
    result: FrequencySweepResult, flux_idx: int
) -> FrequencySweepRowSummary: ...


@overload
def result_row_summary(result: SweepResult1D, flux_idx: int) -> Sweep1DRowSummary: ...


@overload
def result_row_summary(result: SweepResult2D, flux_idx: int) -> Sweep2DRowSummary: ...


@overload
def result_row_summary(result: WorkflowResult, flux_idx: int) -> ResultRowSummary: ...


def result_row_summary(result: WorkflowResult, flux_idx: int) -> ResultRowSummary:
    """Return this representation's journal scalars at the zero-based flux_idx.

    Non-finite scalars become None. Raise IndexError for a negative or out-of-range
    index; invalid kind/fields/dtype/shape use result_declaration's error contract.
    """
    checked = require_workflow_result(result)
    idx = int(flux_idx)
    _require_row_index(checked, idx)
    if (
        isinstance(checked, FrequencySweepResult)
        and checked.result_kind == "qubit_freq"
    ):
        return {
            "fit_freq": _finite_scalar_at(checked.fit_freq, idx),
            "predict_freq": _finite_scalar_at(checked.predict_freq, idx),
            "snr": _finite_scalar_at(checked.snr, idx),
        }
    if isinstance(checked, SweepResult1D) and checked.result_kind == "sweep1d":
        return {
            "fit_value": _finite_scalar_at(checked.fit_value, idx),
            "snr": _finite_scalar_at(checked.snr, idx),
        }
    if checked.result_kind == "sweep2d":
        return {
            "best_freq": _finite_scalar_at(checked.best_freq, idx),
            "best_gain": _finite_scalar_at(checked.best_gain, idx),
        }
    raise TypeError("unsupported autofluxdep Result")


@overload
def result_progress_summary(
    result: FrequencySweepResult,
) -> FrequencySweepProgressSummary: ...


@overload
def result_progress_summary(result: SweepResult1D) -> Sweep1DProgressSummary: ...


@overload
def result_progress_summary(result: SweepResult2D) -> Sweep2DProgressSummary: ...


@overload
def result_progress_summary(result: WorkflowResult) -> ResultProgressSummary: ...


def result_progress_summary(result: WorkflowResult) -> ResultProgressSummary:
    """Return this representation's typed remote progress without mutating data.

    Invalid kind/fields/dtype/shape use result_declaration's error contract.

    ``n_measured`` counts rows whose primary raw signal contains finite data.
    ``fit_summary.n_fitted`` counts rows whose declaration's primary fit scalar is
    finite. ADR-0063 distinguishes committed raw measurements from fit/provide outcomes.
    """
    checked = require_workflow_result(result)
    n_flux = checked.flux.size
    n_measured = _count_primary_raw_rows(checked.signal)
    if (
        isinstance(checked, FrequencySweepResult)
        and checked.result_kind == "qubit_freq"
    ):
        return {
            "kind": "qubit_freq",
            "n_flux": n_flux,
            "n_measured": n_measured,
            "fit_summary": {
                "n_fitted": _count_fitted_rows(checked.fit_freq),
                "last_fit_freq": _last_finite(checked.fit_freq),
            },
        }
    if isinstance(checked, SweepResult1D) and checked.result_kind == "sweep1d":
        return {
            "kind": "sweep1d",
            "n_flux": n_flux,
            "n_measured": n_measured,
            "fit_summary": {
                "n_fitted": _count_fitted_rows(checked.fit_value),
                "last_fit_value": _last_finite(checked.fit_value),
                "x_label": checked.x_label,
            },
        }
    if checked.result_kind == "sweep2d":
        return {
            "kind": "sweep2d",
            "n_flux": n_flux,
            "n_measured": n_measured,
            "fit_summary": {
                "n_fitted": _count_fitted_rows(checked.best_freq),
                "last_best_freq": _last_finite(checked.best_freq),
                "last_best_gain": _last_finite(checked.best_gain),
            },
        }
    raise TypeError("unsupported autofluxdep Result")


def _result_row_values(
    result: object, idx: int
) -> Mapping[DataVariable, NDArray[np.float64] | float]:
    checked = require_workflow_result(result)
    _require_row_index(checked, idx)
    declaration = result_declaration(checked)
    return declaration.row_values(checked, idx)


def _require_row_index(result: WorkflowResult, idx: int) -> None:
    if idx < 0 or idx >= result.flux.size:
        raise IndexError(
            f"flux row index {idx} out of range for {result.flux.size} rows"
        )


def _declaration_for_variables(
    variables: frozenset[DataVariable],
) -> _ResultDeclaration:
    for declaration in _RESULT_DECLARATIONS:
        if variables == declaration.variables:
            return declaration
    present = ", ".join(sorted(str(variable) for variable in variables))
    raise ValueError(f"unsupported autofluxdep node result variables: {present}")


def _require_frequency_sweep(result: object) -> FrequencySweepResult:
    checked = require_workflow_result(result)
    if (
        isinstance(checked, FrequencySweepResult)
        and checked.result_kind == "qubit_freq"
    ):
        return checked
    raise TypeError("result declaration requires qubit_freq")


def _require_sweep1d(result: object) -> SweepResult1D:
    checked = require_workflow_result(result)
    if isinstance(checked, SweepResult1D) and checked.result_kind == "sweep1d":
        return checked
    raise TypeError("result declaration requires sweep1d")


def _require_sweep2d(result: object) -> SweepResult2D:
    checked = require_workflow_result(result)
    if isinstance(checked, SweepResult2D) and checked.result_kind == "sweep2d":
        return checked
    raise TypeError("result declaration requires sweep2d")


def _qubit_freq_variable_specs(
    node_name: str, node_type: str, result: object, flux_unit: str
) -> tuple[StreamingLabberVariableSpec, ...]:
    qubit_freq = _require_frequency_sweep(result)
    flux_axis = Axis("Flux device value", flux_unit, qubit_freq.flux)
    detune_axis = Axis("Detune", "MHz", qubit_freq.detune)
    return (
        _spec(
            VARIABLE_SIGNAL,
            "Signal",
            "a.u.",
            (detune_axis, flux_axis),
            qubit_freq.signal.shape,
            node_name,
            node_type,
            "qubit_freq",
        ),
        _spec(
            VARIABLE_FIT_CURVE,
            "Fit curve",
            "a.u.",
            (detune_axis, flux_axis),
            qubit_freq.fit_curve.shape,
            node_name,
            node_type,
            "qubit_freq",
        ),
        _spec(
            VARIABLE_FIT_FREQ,
            "Fit frequency",
            "MHz",
            (flux_axis,),
            qubit_freq.fit_freq.shape,
            node_name,
            node_type,
            "qubit_freq",
        ),
        _spec(
            VARIABLE_PREDICT_FREQ,
            "Predicted frequency",
            "MHz",
            (flux_axis,),
            qubit_freq.predict_freq.shape,
            node_name,
            node_type,
            "qubit_freq",
        ),
        _spec(
            VARIABLE_SNR,
            "SNR",
            "a.u.",
            (flux_axis,),
            qubit_freq.snr.shape,
            node_name,
            node_type,
            "qubit_freq",
        ),
    )


def _sweep1d_variable_specs(
    node_name: str, node_type: str, result: object, flux_unit: str
) -> tuple[StreamingLabberVariableSpec, ...]:
    sweep = _require_sweep1d(result)
    flux_axis = Axis("Flux device value", flux_unit, sweep.flux)
    x_axis = Axis(sweep.x_label, "", sweep.x)
    return (
        _spec(
            VARIABLE_SIGNAL,
            "Signal",
            "a.u.",
            (x_axis, flux_axis),
            sweep.signal.shape,
            node_name,
            node_type,
            "sweep_1d",
        ),
        _spec(
            VARIABLE_FIT_CURVE,
            "Fit curve",
            "a.u.",
            (x_axis, flux_axis),
            sweep.fit_curve.shape,
            node_name,
            node_type,
            "sweep_1d",
        ),
        _spec(
            VARIABLE_FIT_VALUE,
            "Fit value",
            "",
            (flux_axis,),
            sweep.fit_value.shape,
            node_name,
            node_type,
            "sweep_1d",
        ),
        _spec(
            VARIABLE_SNR,
            "SNR",
            "a.u.",
            (flux_axis,),
            sweep.snr.shape,
            node_name,
            node_type,
            "sweep_1d",
        ),
    )


def _sweep2d_variable_specs(
    node_name: str, node_type: str, result: object, flux_unit: str
) -> tuple[StreamingLabberVariableSpec, ...]:
    sweep = _require_sweep2d(result)
    flux_axis = Axis("Flux device value", flux_unit, sweep.flux)
    freq_axis = Axis("Frequency", "MHz", sweep.freq)
    gain_axis = Axis("Gain", "a.u.", sweep.gain)
    return (
        _spec(
            VARIABLE_SIGNAL,
            "Signal",
            "a.u.",
            (gain_axis, freq_axis, flux_axis),
            sweep.signal.shape,
            node_name,
            node_type,
            "sweep_2d",
        ),
        _spec(
            VARIABLE_BEST_FREQ,
            "Best frequency",
            "MHz",
            (flux_axis,),
            sweep.best_freq.shape,
            node_name,
            node_type,
            "sweep_2d",
        ),
        _spec(
            VARIABLE_BEST_GAIN,
            "Best gain",
            "a.u.",
            (flux_axis,),
            sweep.best_gain.shape,
            node_name,
            node_type,
            "sweep_2d",
        ),
    )


def _qubit_freq_row_values(
    result: object, idx: int
) -> Mapping[DataVariable, NDArray[np.float64] | float]:
    qubit_freq = _require_frequency_sweep(result)
    return {
        VARIABLE_SIGNAL: qubit_freq.signal[idx],
        VARIABLE_FIT_CURVE: qubit_freq.fit_curve[idx],
        VARIABLE_FIT_FREQ: float(qubit_freq.fit_freq[idx]),
        VARIABLE_PREDICT_FREQ: float(qubit_freq.predict_freq[idx]),
        VARIABLE_SNR: float(qubit_freq.snr[idx]),
    }


def _sweep1d_row_values(
    result: object, idx: int
) -> Mapping[DataVariable, NDArray[np.float64] | float]:
    sweep = _require_sweep1d(result)
    return {
        VARIABLE_SIGNAL: sweep.signal[idx],
        VARIABLE_FIT_CURVE: sweep.fit_curve[idx],
        VARIABLE_FIT_VALUE: float(sweep.fit_value[idx]),
        VARIABLE_SNR: float(sweep.snr[idx]),
    }


def _sweep2d_row_values(
    result: object, idx: int
) -> Mapping[DataVariable, NDArray[np.float64] | float]:
    sweep = _require_sweep2d(result)
    return {
        VARIABLE_SIGNAL: sweep.signal[idx],
        VARIABLE_BEST_FREQ: float(sweep.best_freq[idx]),
        VARIABLE_BEST_GAIN: float(sweep.best_gain[idx]),
    }


def _load_qubit_freq(
    variables: Mapping[DataVariable, LabberPayload],
) -> FrequencySweepResult:
    signal = variables[VARIABLE_SIGNAL]
    fit_curve = variables[VARIABLE_FIT_CURVE]
    fit_freq = variables[VARIABLE_FIT_FREQ]
    predict_freq = variables[VARIABLE_PREDICT_FREQ]
    snr = variables[VARIABLE_SNR]
    signal_z = _real_data(signal, VARIABLE_SIGNAL)
    _require_ndim(VARIABLE_SIGNAL, signal_z, 2)
    detune = _axis_values(signal, VARIABLE_SIGNAL, 0)
    flux = _axis_values(signal, VARIABLE_SIGNAL, 1)
    fit_curve_z = _matching_data(
        fit_curve, VARIABLE_FIT_CURVE, signal_z.shape, (detune, flux)
    )
    fit_freq_z = _matching_data(fit_freq, VARIABLE_FIT_FREQ, flux.shape, (flux,))
    predict_freq_z = _matching_data(
        predict_freq, VARIABLE_PREDICT_FREQ, flux.shape, (flux,)
    )
    snr_z = _matching_data(snr, VARIABLE_SNR, flux.shape, (flux,))
    return _LoadedFrequencySweep(
        flux=flux,
        detune=detune,
        signal=signal_z,
        fit_curve=fit_curve_z,
        fit_freq=fit_freq_z,
        predict_freq=predict_freq_z,
        snr=snr_z,
    )


def _load_sweep1d(variables: Mapping[DataVariable, LabberPayload]) -> SweepResult1D:
    signal = variables[VARIABLE_SIGNAL]
    fit_curve = variables[VARIABLE_FIT_CURVE]
    fit_value = variables[VARIABLE_FIT_VALUE]
    snr = variables[VARIABLE_SNR]
    signal_z = _real_data(signal, VARIABLE_SIGNAL)
    _require_ndim(VARIABLE_SIGNAL, signal_z, 2)
    x = _axis_values(signal, VARIABLE_SIGNAL, 0)
    flux = _axis_values(signal, VARIABLE_SIGNAL, 1)
    fit_curve_z = _matching_data(
        fit_curve, VARIABLE_FIT_CURVE, signal_z.shape, (x, flux)
    )
    fit_value_z = _matching_data(fit_value, VARIABLE_FIT_VALUE, flux.shape, (flux,))
    snr_z = _matching_data(snr, VARIABLE_SNR, flux.shape, (flux,))
    return _LoadedSweep1D(
        flux=flux,
        x=x,
        signal=signal_z,
        fit_curve=fit_curve_z,
        fit_value=fit_value_z,
        snr=snr_z,
        x_label=str(signal.axes[0].name),
    )


def _load_sweep2d(variables: Mapping[DataVariable, LabberPayload]) -> SweepResult2D:
    signal = variables[VARIABLE_SIGNAL]
    best_freq = variables[VARIABLE_BEST_FREQ]
    best_gain = variables[VARIABLE_BEST_GAIN]
    signal_z = _real_data(signal, VARIABLE_SIGNAL)
    _require_ndim(VARIABLE_SIGNAL, signal_z, 3)
    gain = _axis_values(signal, VARIABLE_SIGNAL, 0)
    freq = _axis_values(signal, VARIABLE_SIGNAL, 1)
    flux = _axis_values(signal, VARIABLE_SIGNAL, 2)
    best_freq_z = _matching_data(best_freq, VARIABLE_BEST_FREQ, flux.shape, (flux,))
    best_gain_z = _matching_data(best_gain, VARIABLE_BEST_GAIN, flux.shape, (flux,))
    return _LoadedSweep2D(
        flux=flux,
        freq=freq,
        gain=gain,
        signal=signal_z,
        best_freq=best_freq_z,
        best_gain=best_gain_z,
    )


_RESULT_DECLARATIONS: tuple[_ResultDeclaration, ...] = (
    _ResultDeclaration(
        kind="qubit_freq",
        variables=frozenset(
            {
                VARIABLE_SIGNAL,
                VARIABLE_FIT_CURVE,
                VARIABLE_FIT_FREQ,
                VARIABLE_PREDICT_FREQ,
                VARIABLE_SNR,
            }
        ),
        summary_scalar_attrs=("fit_freq", "predict_freq", "snr"),
        last_fit_fields=(("last_fit_freq", "fit_freq"),),
        spec_builder=_qubit_freq_variable_specs,
        row_values=_qubit_freq_row_values,
        loader=_load_qubit_freq,
    ),
    _ResultDeclaration(
        kind="sweep1d",
        variables=frozenset(
            {VARIABLE_SIGNAL, VARIABLE_FIT_CURVE, VARIABLE_FIT_VALUE, VARIABLE_SNR}
        ),
        summary_scalar_attrs=("fit_value", "snr"),
        last_fit_fields=(("last_fit_value", "fit_value"),),
        spec_builder=_sweep1d_variable_specs,
        row_values=_sweep1d_row_values,
        loader=_load_sweep1d,
    ),
    _ResultDeclaration(
        kind="sweep2d",
        variables=frozenset({VARIABLE_SIGNAL, VARIABLE_BEST_FREQ, VARIABLE_BEST_GAIN}),
        summary_scalar_attrs=("best_freq", "best_gain"),
        last_fit_fields=(
            ("last_best_freq", "best_freq"),
            ("last_best_gain", "best_gain"),
        ),
        spec_builder=_sweep2d_variable_specs,
        row_values=_sweep2d_row_values,
        loader=_load_sweep2d,
    ),
)


def _count_primary_raw_rows(raw: NDArray[np.float64]) -> int:
    # Reduce all trailing axes, including empty grids, without reshaping zero rows.
    return int(np.count_nonzero(np.isfinite(raw).any(axis=tuple(range(1, raw.ndim)))))


def _count_fitted_rows(values: NDArray[np.float64]) -> int:
    return int(np.count_nonzero(np.isfinite(values)))


def _last_finite(values: NDArray[np.float64]) -> float | None:
    finite = values[np.isfinite(values)]
    return float(finite[-1]) if finite.size else None


def _finite_scalar_at(values: NDArray[np.float64], idx: int) -> float | None:
    value = values[idx]
    return None if not np.isfinite(value) else float(value)


def _spec(
    variable: DataVariable,
    data_name: str,
    data_unit: str,
    axes: tuple[Axis, ...],
    shape: tuple[int, ...],
    node_name: str,
    node_type: str,
    result_kind: str,
) -> StreamingLabberVariableSpec:
    return StreamingLabberVariableSpec(
        variable,
        data_name,
        data_unit,
        axes,
        shape,
        attrs={
            "zcu_tools.autofluxdep.node_name": node_name,
            "zcu_tools.autofluxdep.node_type": node_type,
            "zcu_tools.autofluxdep.result_kind": result_kind,
            "zcu_tools.autofluxdep.result_role": str(variable),
            "zcu_tools.autofluxdep.role_label": data_name,
            "zcu_tools.autofluxdep.role_unit": data_unit,
        },
    )


def _real_axis(axis: Axis) -> NDArray[np.float64]:
    return np.asarray(axis.values, dtype=np.float64)


def _real_data(payload: LabberPayload, variable: DataVariable) -> NDArray[np.float64]:
    values = np.asarray(payload.z.real, dtype=np.float64)
    expected_shape = tuple(
        int(np.asarray(axis.values, dtype=np.float64).reshape(-1).shape[0])
        for axis in reversed(payload.axes)
    )
    if values.shape != expected_shape:
        raise ValueError(
            f"variable {variable!r} data shape {values.shape} does not match axes "
            f"{expected_shape}"
        )
    return values


def _require_ndim(
    variable: DataVariable, values: NDArray[np.float64], ndim: int
) -> None:
    if values.ndim != ndim:
        raise ValueError(
            f"variable {variable!r} must be {ndim}D, got shape {values.shape}"
        )


def _axis_values(
    payload: LabberPayload, variable: DataVariable, axis_index: int
) -> NDArray[np.float64]:
    if len(payload.axes) <= axis_index:
        raise ValueError(
            f"variable {variable!r} is missing axis {axis_index}; "
            f"only {len(payload.axes)} axis/axes present"
        )
    return _real_axis(payload.axes[axis_index])


def _matching_data(
    payload: LabberPayload,
    variable: DataVariable,
    expected_shape: tuple[int, ...],
    expected_axes: tuple[NDArray[np.float64], ...],
) -> NDArray[np.float64]:
    values = _real_data(payload, variable)
    if values.shape != expected_shape:
        raise ValueError(
            f"variable {variable!r} shape {values.shape} does not match expected "
            f"{expected_shape}"
        )
    if len(payload.axes) != len(expected_axes):
        raise ValueError(
            f"variable {variable!r} axis count {len(payload.axes)} does not match "
            f"expected {len(expected_axes)}"
        )
    for axis_index, expected in enumerate(expected_axes):
        actual = _real_axis(payload.axes[axis_index])
        if actual.shape != expected.shape or not np.array_equal(actual, expected):
            raise ValueError(
                f"variable {variable!r} axis {axis_index} does not match the signal variable"
            )
    return values


__all__ = [
    "VARIABLE_BEST_FREQ",
    "VARIABLE_BEST_GAIN",
    "VARIABLE_FIT_CURVE",
    "VARIABLE_FIT_FREQ",
    "VARIABLE_FIT_VALUE",
    "VARIABLE_PREDICT_FREQ",
    "VARIABLE_SIGNAL",
    "VARIABLE_SNR",
    "load_node_result",
    "read_result_row",
    "result_declaration",
    "result_progress_summary",
    "result_row_variable_names",
    "result_row_summary",
    "result_variable_specs",
    "write_result_row",
]
