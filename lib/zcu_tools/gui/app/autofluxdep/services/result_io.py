"""Typed Result <-> streaming Labber role mapping for autofluxdep artifacts."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np

from zcu_tools.gui.app.autofluxdep.nodes.result import (
    QubitFreqResult,
    Sweep1DResult,
    Sweep2DResult,
)
from zcu_tools.utils.datasaver import (
    Axis,
    DatasetRole,
    LabberPayload,
    StreamingGroupedLabberWriter,
    StreamingLabberRoleSpec,
    load_grouped_labber_data,
)

ROLE_SIGNAL = DatasetRole("signal")
ROLE_FIT_CURVE = DatasetRole("fit_curve")
ROLE_FIT_FREQ = DatasetRole("fit_freq")
ROLE_PREDICT_FREQ = DatasetRole("predict_freq")
ROLE_SNR = DatasetRole("snr")
ROLE_FIT_VALUE = DatasetRole("fit_value")
ROLE_BEST_FREQ = DatasetRole("best_freq")
ROLE_BEST_GAIN = DatasetRole("best_gain")

_ResultT = TypeVar("_ResultT")
_ResultObject = QubitFreqResult | Sweep1DResult | Sweep2DResult
_SpecBuilder = Callable[[str, str, object, str], tuple[StreamingLabberRoleSpec, ...]]
_RowValueBuilder = Callable[[object, int], Mapping[DatasetRole, Any]]
_Loader = Callable[[Mapping[DatasetRole, LabberPayload]], _ResultObject]
_ExtraFitSummary = Callable[[object], Mapping[str, Any]]


def _empty_fit_summary(_result: object) -> Mapping[str, Any]:
    return {}


@dataclass(frozen=True)
class _ResultDeclaration:
    result_type: type[object]
    kind: str
    roles: frozenset[DatasetRole]
    primary_raw_role: DatasetRole
    primary_raw_attr: str
    fit_scalar_attrs: tuple[str, ...]
    summary_scalar_attrs: tuple[str, ...]
    last_fit_fields: tuple[tuple[str, str], ...]
    spec_builder: _SpecBuilder
    row_values: _RowValueBuilder
    loader: _Loader
    extra_fit_summary: _ExtraFitSummary = _empty_fit_summary


def result_role_specs(
    node_name: str,
    node_type: str,
    result: object,
    *,
    flux_unit: str = "",
) -> tuple[StreamingLabberRoleSpec, ...]:
    """Build the stream schema for one placed node Result."""
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
    """Write all persisted roles for one flux row."""
    del node_name, node_type
    idx = int(flux_idx)
    values = _result_row_values(result, idx)
    for role, row in values.items():
        writer.write_outer_slice(role, idx, row, timestamp=timestamp)
    return tuple(str(role) for role in values)


def result_row_role_names(result: object, flux_idx: int) -> tuple[str, ...]:
    """Return the role names a row write would touch, without mutating storage."""
    return tuple(str(role) for role in _result_row_values(result, int(flux_idx)))


def load_node_result(
    path: str, node_type: str
) -> QubitFreqResult | Sweep1DResult | Sweep2DResult:
    """Load a node HDF5 file back into its typed sweep Result."""
    del node_type
    grouped = load_grouped_labber_data(path)
    roles = grouped.roles
    declaration = _declaration_for_roles(frozenset(roles))
    return declaration.loader(roles)


def read_result_row(
    path: str,
    node_type: str,
    flux_idx: int,
) -> Mapping[DatasetRole, np.ndarray | float]:
    """Read one committed-or-nan Result row from a node HDF5 file."""
    result = load_node_result(path, node_type)
    values = _result_row_values(result, int(flux_idx))
    row: dict[DatasetRole, np.ndarray | float] = {}
    for role, value in values.items():
        array = np.asarray(value)
        if array.ndim == 0:
            row[role] = float(array)
        else:
            row[role] = array.copy()
    return row


def result_declaration(result_or_type: object) -> _ResultDeclaration:
    """Return the single Result declaration for a Result instance or class."""
    if isinstance(result_or_type, type):
        for declaration in _RESULT_DECLARATIONS:
            if issubclass(result_or_type, declaration.result_type):
                return declaration
        type_name = result_or_type.__name__
    else:
        for declaration in _RESULT_DECLARATIONS:
            if isinstance(result_or_type, declaration.result_type):
                return declaration
        type_name = type(result_or_type).__name__
    raise TypeError(f"unsupported autofluxdep Result type {type_name}")


def result_row_summary(result: object, flux_idx: int) -> dict[str, float | None]:
    """Return the small per-row scalar summary stored in the journal."""
    declaration = result_declaration(result)
    idx = int(flux_idx)
    return {
        attr: _finite_scalar_at(getattr(result, attr), idx)
        for attr in declaration.summary_scalar_attrs
    }


def result_progress_summary(result: object) -> dict[str, Any]:
    """Return the remote progress summary for one node Result.

    ``n_measured`` counts rows whose primary raw signal contains finite data.
    ``fit_summary.n_fitted`` counts rows whose declaration's primary fit scalar is
    finite. ADR-0040 treats raw-present/fit-nan rows as committed measurements.
    """
    declaration = result_declaration(result)
    fit_summary = dict(declaration.extra_fit_summary(result))
    fit_summary["n_fitted"] = _count_fitted_rows(result, declaration)
    for output_key, attr in declaration.last_fit_fields:
        fit_summary[output_key] = _last_finite(getattr(result, attr))
    return {
        "kind": declaration.kind,
        "n_flux": _n_flux(result),
        "n_measured": _count_primary_raw_rows(result, declaration),
        "fit_summary": fit_summary,
    }


def _result_row_values(result: object, idx: int) -> Mapping[DatasetRole, Any]:
    declaration = result_declaration(result)
    return declaration.row_values(result, idx)


def _declaration_for_roles(roles: frozenset[DatasetRole]) -> _ResultDeclaration:
    for declaration in _RESULT_DECLARATIONS:
        if roles == declaration.roles:
            return declaration
    present = ", ".join(sorted(str(role) for role in roles))
    raise ValueError(f"unsupported autofluxdep node result roles: {present}")


def _require_result(result: object, expected: type[_ResultT]) -> _ResultT:
    if not isinstance(result, expected):
        raise TypeError(
            f"result declaration for {expected.__name__} received "
            f"{type(result).__name__}"
        )
    return result


def _qubit_freq_role_specs(
    node_name: str, node_type: str, result: object, flux_unit: str
) -> tuple[StreamingLabberRoleSpec, ...]:
    qubit_freq = _require_result(result, QubitFreqResult)
    flux_axis = Axis("Flux device value", flux_unit, qubit_freq.flux)
    detune_axis = Axis("Detune", "MHz", qubit_freq.detune)
    return (
        _spec(
            ROLE_SIGNAL,
            "Signal",
            "a.u.",
            (detune_axis, flux_axis),
            qubit_freq.signal.shape,
            node_name,
            node_type,
            "qubit_freq",
        ),
        _spec(
            ROLE_FIT_CURVE,
            "Fit curve",
            "a.u.",
            (detune_axis, flux_axis),
            qubit_freq.fit_curve.shape,
            node_name,
            node_type,
            "qubit_freq",
        ),
        _spec(
            ROLE_FIT_FREQ,
            "Fit frequency",
            "MHz",
            (flux_axis,),
            qubit_freq.fit_freq.shape,
            node_name,
            node_type,
            "qubit_freq",
        ),
        _spec(
            ROLE_PREDICT_FREQ,
            "Predicted frequency",
            "MHz",
            (flux_axis,),
            qubit_freq.predict_freq.shape,
            node_name,
            node_type,
            "qubit_freq",
        ),
        _spec(
            ROLE_SNR,
            "SNR",
            "a.u.",
            (flux_axis,),
            qubit_freq.snr.shape,
            node_name,
            node_type,
            "qubit_freq",
        ),
    )


def _sweep1d_role_specs(
    node_name: str, node_type: str, result: object, flux_unit: str
) -> tuple[StreamingLabberRoleSpec, ...]:
    sweep = _require_result(result, Sweep1DResult)
    flux_axis = Axis("Flux device value", flux_unit, sweep.flux)
    x_axis = Axis(sweep.x_label, "", sweep.x)
    return (
        _spec(
            ROLE_SIGNAL,
            "Signal",
            "a.u.",
            (x_axis, flux_axis),
            sweep.signal.shape,
            node_name,
            node_type,
            "sweep_1d",
        ),
        _spec(
            ROLE_FIT_CURVE,
            "Fit curve",
            "a.u.",
            (x_axis, flux_axis),
            sweep.fit_curve.shape,
            node_name,
            node_type,
            "sweep_1d",
        ),
        _spec(
            ROLE_FIT_VALUE,
            "Fit value",
            "",
            (flux_axis,),
            sweep.fit_value.shape,
            node_name,
            node_type,
            "sweep_1d",
        ),
        _spec(
            ROLE_SNR,
            "SNR",
            "a.u.",
            (flux_axis,),
            sweep.snr.shape,
            node_name,
            node_type,
            "sweep_1d",
        ),
    )


def _sweep2d_role_specs(
    node_name: str, node_type: str, result: object, flux_unit: str
) -> tuple[StreamingLabberRoleSpec, ...]:
    sweep = _require_result(result, Sweep2DResult)
    flux_axis = Axis("Flux device value", flux_unit, sweep.flux)
    freq_axis = Axis("Frequency", "MHz", sweep.freq)
    gain_axis = Axis("Gain", "a.u.", sweep.gain)
    return (
        _spec(
            ROLE_SIGNAL,
            "Signal",
            "a.u.",
            (gain_axis, freq_axis, flux_axis),
            sweep.signal.shape,
            node_name,
            node_type,
            "sweep_2d",
        ),
        _spec(
            ROLE_BEST_FREQ,
            "Best frequency",
            "MHz",
            (flux_axis,),
            sweep.best_freq.shape,
            node_name,
            node_type,
            "sweep_2d",
        ),
        _spec(
            ROLE_BEST_GAIN,
            "Best gain",
            "a.u.",
            (flux_axis,),
            sweep.best_gain.shape,
            node_name,
            node_type,
            "sweep_2d",
        ),
    )


def _qubit_freq_row_values(result: object, idx: int) -> Mapping[DatasetRole, Any]:
    qubit_freq = _require_result(result, QubitFreqResult)
    return {
        ROLE_SIGNAL: qubit_freq.signal[idx],
        ROLE_FIT_CURVE: qubit_freq.fit_curve[idx],
        ROLE_FIT_FREQ: qubit_freq.fit_freq[idx],
        ROLE_PREDICT_FREQ: qubit_freq.predict_freq[idx],
        ROLE_SNR: qubit_freq.snr[idx],
    }


def _sweep1d_row_values(result: object, idx: int) -> Mapping[DatasetRole, Any]:
    sweep = _require_result(result, Sweep1DResult)
    return {
        ROLE_SIGNAL: sweep.signal[idx],
        ROLE_FIT_CURVE: sweep.fit_curve[idx],
        ROLE_FIT_VALUE: sweep.fit_value[idx],
        ROLE_SNR: sweep.snr[idx],
    }


def _sweep2d_row_values(result: object, idx: int) -> Mapping[DatasetRole, Any]:
    sweep = _require_result(result, Sweep2DResult)
    return {
        ROLE_SIGNAL: sweep.signal[idx],
        ROLE_BEST_FREQ: sweep.best_freq[idx],
        ROLE_BEST_GAIN: sweep.best_gain[idx],
    }


def _load_qubit_freq(
    roles: Mapping[DatasetRole, LabberPayload],
) -> QubitFreqResult:
    signal = roles[ROLE_SIGNAL]
    fit_curve = roles[ROLE_FIT_CURVE]
    fit_freq = roles[ROLE_FIT_FREQ]
    predict_freq = roles[ROLE_PREDICT_FREQ]
    snr = roles[ROLE_SNR]
    signal_z = _real_data(signal, ROLE_SIGNAL)
    _require_ndim(ROLE_SIGNAL, signal_z, 2)
    detune = _axis_values(signal, ROLE_SIGNAL, 0)
    flux = _axis_values(signal, ROLE_SIGNAL, 1)
    fit_curve_z = _matching_data(
        fit_curve, ROLE_FIT_CURVE, signal_z.shape, (detune, flux)
    )
    fit_freq_z = _matching_data(fit_freq, ROLE_FIT_FREQ, flux.shape, (flux,))
    predict_freq_z = _matching_data(
        predict_freq, ROLE_PREDICT_FREQ, flux.shape, (flux,)
    )
    snr_z = _matching_data(snr, ROLE_SNR, flux.shape, (flux,))
    return QubitFreqResult(
        flux=flux,
        detune=detune,
        signal=signal_z,
        fit_curve=fit_curve_z,
        fit_freq=fit_freq_z,
        predict_freq=predict_freq_z,
        snr=snr_z,
    )


def _load_sweep1d(roles: Mapping[DatasetRole, LabberPayload]) -> Sweep1DResult:
    signal = roles[ROLE_SIGNAL]
    fit_curve = roles[ROLE_FIT_CURVE]
    fit_value = roles[ROLE_FIT_VALUE]
    snr = roles[ROLE_SNR]
    signal_z = _real_data(signal, ROLE_SIGNAL)
    _require_ndim(ROLE_SIGNAL, signal_z, 2)
    x = _axis_values(signal, ROLE_SIGNAL, 0)
    flux = _axis_values(signal, ROLE_SIGNAL, 1)
    fit_curve_z = _matching_data(fit_curve, ROLE_FIT_CURVE, signal_z.shape, (x, flux))
    fit_value_z = _matching_data(fit_value, ROLE_FIT_VALUE, flux.shape, (flux,))
    snr_z = _matching_data(snr, ROLE_SNR, flux.shape, (flux,))
    return Sweep1DResult(
        flux=flux,
        x=x,
        signal=signal_z,
        fit_curve=fit_curve_z,
        fit_value=fit_value_z,
        snr=snr_z,
        x_label=str(signal.axes[0].name),
    )


def _load_sweep2d(roles: Mapping[DatasetRole, LabberPayload]) -> Sweep2DResult:
    signal = roles[ROLE_SIGNAL]
    best_freq = roles[ROLE_BEST_FREQ]
    best_gain = roles[ROLE_BEST_GAIN]
    signal_z = _real_data(signal, ROLE_SIGNAL)
    _require_ndim(ROLE_SIGNAL, signal_z, 3)
    gain = _axis_values(signal, ROLE_SIGNAL, 0)
    freq = _axis_values(signal, ROLE_SIGNAL, 1)
    flux = _axis_values(signal, ROLE_SIGNAL, 2)
    best_freq_z = _matching_data(best_freq, ROLE_BEST_FREQ, flux.shape, (flux,))
    best_gain_z = _matching_data(best_gain, ROLE_BEST_GAIN, flux.shape, (flux,))
    return Sweep2DResult(
        flux=flux,
        freq=freq,
        gain=gain,
        signal=signal_z,
        best_freq=best_freq_z,
        best_gain=best_gain_z,
    )


def _sweep1d_extra_fit_summary(result: object) -> Mapping[str, Any]:
    sweep = _require_result(result, Sweep1DResult)
    return {"x_label": sweep.x_label}


_RESULT_DECLARATIONS: tuple[_ResultDeclaration, ...] = (
    _ResultDeclaration(
        result_type=QubitFreqResult,
        kind="qubit_freq",
        roles=frozenset(
            {
                ROLE_SIGNAL,
                ROLE_FIT_CURVE,
                ROLE_FIT_FREQ,
                ROLE_PREDICT_FREQ,
                ROLE_SNR,
            }
        ),
        primary_raw_role=ROLE_SIGNAL,
        primary_raw_attr="signal",
        fit_scalar_attrs=("fit_freq",),
        summary_scalar_attrs=("fit_freq", "predict_freq", "snr"),
        last_fit_fields=(("last_fit_freq", "fit_freq"),),
        spec_builder=_qubit_freq_role_specs,
        row_values=_qubit_freq_row_values,
        loader=_load_qubit_freq,
    ),
    _ResultDeclaration(
        result_type=Sweep1DResult,
        kind="sweep1d",
        roles=frozenset({ROLE_SIGNAL, ROLE_FIT_CURVE, ROLE_FIT_VALUE, ROLE_SNR}),
        primary_raw_role=ROLE_SIGNAL,
        primary_raw_attr="signal",
        fit_scalar_attrs=("fit_value",),
        summary_scalar_attrs=("fit_value", "snr"),
        last_fit_fields=(("last_fit_value", "fit_value"),),
        spec_builder=_sweep1d_role_specs,
        row_values=_sweep1d_row_values,
        loader=_load_sweep1d,
        extra_fit_summary=_sweep1d_extra_fit_summary,
    ),
    _ResultDeclaration(
        result_type=Sweep2DResult,
        kind="sweep2d",
        roles=frozenset({ROLE_SIGNAL, ROLE_BEST_FREQ, ROLE_BEST_GAIN}),
        primary_raw_role=ROLE_SIGNAL,
        primary_raw_attr="signal",
        fit_scalar_attrs=("best_freq",),
        summary_scalar_attrs=("best_freq", "best_gain"),
        last_fit_fields=(
            ("last_best_freq", "best_freq"),
            ("last_best_gain", "best_gain"),
        ),
        spec_builder=_sweep2d_role_specs,
        row_values=_sweep2d_row_values,
        loader=_load_sweep2d,
    ),
)


def _count_primary_raw_rows(result: object, declaration: _ResultDeclaration) -> int:
    raw = np.asarray(getattr(result, declaration.primary_raw_attr), dtype=np.float64)
    if raw.ndim == 0:
        raise ValueError(
            f"primary raw role {declaration.primary_raw_role!r} must be at least 1D"
        )
    rows = raw.reshape(raw.shape[0], -1)
    return int(np.count_nonzero(np.isfinite(rows).any(axis=1)))


def _count_fitted_rows(result: object, declaration: _ResultDeclaration) -> int:
    if not declaration.fit_scalar_attrs:
        return 0
    values = np.asarray(
        getattr(result, declaration.fit_scalar_attrs[0]), dtype=np.float64
    )
    return int(np.count_nonzero(np.isfinite(values)))


def _last_finite(values: Any) -> float | None:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float(finite[-1]) if finite.size else None


def _finite_scalar_at(values: Any, idx: int) -> float | None:
    value = np.asarray(values, dtype=np.float64).reshape(-1)[idx]
    return None if not np.isfinite(value) else float(value)


def _n_flux(result: object) -> int:
    flux = np.asarray(getattr(result, "flux"), dtype=np.float64)
    if flux.ndim != 1:
        raise ValueError(f"Result flux axis must be 1D, got shape {flux.shape}")
    return int(flux.shape[0])


def _spec(
    role: DatasetRole,
    data_name: str,
    data_unit: str,
    axes: tuple[Axis, ...],
    shape: tuple[int, ...],
    node_name: str,
    node_type: str,
    result_kind: str,
) -> StreamingLabberRoleSpec:
    return StreamingLabberRoleSpec(
        role,
        data_name,
        data_unit,
        axes,
        shape,
        attrs={
            "zcu_tools.autofluxdep.node_name": node_name,
            "zcu_tools.autofluxdep.node_type": node_type,
            "zcu_tools.autofluxdep.result_kind": result_kind,
            "zcu_tools.autofluxdep.result_role": str(role),
            "zcu_tools.autofluxdep.role_label": data_name,
            "zcu_tools.autofluxdep.role_unit": data_unit,
        },
    )


def _real_axis(axis: Axis) -> np.ndarray:
    return np.asarray(axis.values, dtype=np.float64)


def _real_data(payload: LabberPayload, role: DatasetRole) -> np.ndarray:
    values = np.asarray(payload.z.real, dtype=np.float64)
    expected_shape = tuple(
        int(np.asarray(axis.values, dtype=np.float64).reshape(-1).shape[0])
        for axis in reversed(payload.axes)
    )
    if values.shape != expected_shape:
        raise ValueError(
            f"role {role!r} data shape {values.shape} does not match axes "
            f"{expected_shape}"
        )
    return values


def _require_ndim(role: DatasetRole, values: np.ndarray, ndim: int) -> None:
    if values.ndim != ndim:
        raise ValueError(f"role {role!r} must be {ndim}D, got shape {values.shape}")


def _axis_values(
    payload: LabberPayload, role: DatasetRole, axis_index: int
) -> np.ndarray:
    if len(payload.axes) <= axis_index:
        raise ValueError(
            f"role {role!r} is missing axis {axis_index}; "
            f"only {len(payload.axes)} axis/axes present"
        )
    return _real_axis(payload.axes[axis_index])


def _matching_data(
    payload: LabberPayload,
    role: DatasetRole,
    expected_shape: tuple[int, ...],
    expected_axes: tuple[np.ndarray, ...],
) -> np.ndarray:
    values = _real_data(payload, role)
    if values.shape != expected_shape:
        raise ValueError(
            f"role {role!r} shape {values.shape} does not match expected "
            f"{expected_shape}"
        )
    if len(payload.axes) != len(expected_axes):
        raise ValueError(
            f"role {role!r} axis count {len(payload.axes)} does not match "
            f"expected {len(expected_axes)}"
        )
    for axis_index, expected in enumerate(expected_axes):
        actual = _real_axis(payload.axes[axis_index])
        if actual.shape != expected.shape or not np.array_equal(actual, expected):
            raise ValueError(
                f"role {role!r} axis {axis_index} does not match the signal role"
            )
    return values


__all__ = [
    "ROLE_BEST_FREQ",
    "ROLE_BEST_GAIN",
    "ROLE_FIT_CURVE",
    "ROLE_FIT_FREQ",
    "ROLE_FIT_VALUE",
    "ROLE_PREDICT_FREQ",
    "ROLE_SIGNAL",
    "ROLE_SNR",
    "load_node_result",
    "read_result_row",
    "result_declaration",
    "result_progress_summary",
    "result_row_role_names",
    "result_row_summary",
    "result_role_specs",
    "write_result_row",
]
