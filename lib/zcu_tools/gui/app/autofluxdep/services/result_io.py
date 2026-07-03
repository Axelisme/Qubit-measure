"""Typed Result <-> streaming Labber role mapping for autofluxdep artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

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
_QUBIT_FREQ_ROLES = frozenset(
    {ROLE_SIGNAL, ROLE_FIT_CURVE, ROLE_FIT_FREQ, ROLE_PREDICT_FREQ, ROLE_SNR}
)
_SWEEP1D_ROLES = frozenset({ROLE_SIGNAL, ROLE_FIT_CURVE, ROLE_FIT_VALUE, ROLE_SNR})
_SWEEP2D_ROLES = frozenset({ROLE_SIGNAL, ROLE_BEST_FREQ, ROLE_BEST_GAIN})


def result_role_specs(
    node_name: str,
    node_type: str,
    result: object,
    *,
    flux_unit: str = "",
) -> tuple[StreamingLabberRoleSpec, ...]:
    """Build the stream schema for one placed node Result."""
    if isinstance(result, QubitFreqResult):
        flux_axis = Axis("Flux device value", flux_unit, result.flux)
        detune_axis = Axis("Detune", "MHz", result.detune)
        return (
            _spec(
                ROLE_SIGNAL,
                "Signal",
                "a.u.",
                (detune_axis, flux_axis),
                result.signal.shape,
                node_name,
                node_type,
                "qubit_freq",
            ),
            _spec(
                ROLE_FIT_CURVE,
                "Fit curve",
                "a.u.",
                (detune_axis, flux_axis),
                result.fit_curve.shape,
                node_name,
                node_type,
                "qubit_freq",
            ),
            _spec(
                ROLE_FIT_FREQ,
                "Fit frequency",
                "MHz",
                (flux_axis,),
                result.fit_freq.shape,
                node_name,
                node_type,
                "qubit_freq",
            ),
            _spec(
                ROLE_PREDICT_FREQ,
                "Predicted frequency",
                "MHz",
                (flux_axis,),
                result.predict_freq.shape,
                node_name,
                node_type,
                "qubit_freq",
            ),
            _spec(
                ROLE_SNR,
                "SNR",
                "a.u.",
                (flux_axis,),
                result.snr.shape,
                node_name,
                node_type,
                "qubit_freq",
            ),
        )
    if isinstance(result, Sweep1DResult):
        flux_axis = Axis("Flux device value", flux_unit, result.flux)
        x_axis = Axis(result.x_label, "", result.x)
        return (
            _spec(
                ROLE_SIGNAL,
                "Signal",
                "a.u.",
                (x_axis, flux_axis),
                result.signal.shape,
                node_name,
                node_type,
                "sweep_1d",
            ),
            _spec(
                ROLE_FIT_CURVE,
                "Fit curve",
                "a.u.",
                (x_axis, flux_axis),
                result.fit_curve.shape,
                node_name,
                node_type,
                "sweep_1d",
            ),
            _spec(
                ROLE_FIT_VALUE,
                "Fit value",
                "",
                (flux_axis,),
                result.fit_value.shape,
                node_name,
                node_type,
                "sweep_1d",
            ),
            _spec(
                ROLE_SNR,
                "SNR",
                "a.u.",
                (flux_axis,),
                result.snr.shape,
                node_name,
                node_type,
                "sweep_1d",
            ),
        )
    if isinstance(result, Sweep2DResult):
        flux_axis = Axis("Flux device value", flux_unit, result.flux)
        freq_axis = Axis("Frequency", "MHz", result.freq)
        gain_axis = Axis("Gain", "a.u.", result.gain)
        return (
            _spec(
                ROLE_SIGNAL,
                "Signal",
                "a.u.",
                (gain_axis, freq_axis, flux_axis),
                result.signal.shape,
                node_name,
                node_type,
                "sweep_2d",
            ),
            _spec(
                ROLE_BEST_FREQ,
                "Best frequency",
                "MHz",
                (flux_axis,),
                result.best_freq.shape,
                node_name,
                node_type,
                "sweep_2d",
            ),
            _spec(
                ROLE_BEST_GAIN,
                "Best gain",
                "a.u.",
                (flux_axis,),
                result.best_gain.shape,
                node_name,
                node_type,
                "sweep_2d",
            ),
        )
    raise TypeError(f"unsupported autofluxdep Result type {type(result).__name__}")


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


def _result_row_values(result: object, idx: int) -> Mapping[DatasetRole, Any]:
    if isinstance(result, QubitFreqResult):
        return {
            ROLE_SIGNAL: result.signal[idx],
            ROLE_FIT_CURVE: result.fit_curve[idx],
            ROLE_FIT_FREQ: result.fit_freq[idx],
            ROLE_PREDICT_FREQ: result.predict_freq[idx],
            ROLE_SNR: result.snr[idx],
        }
    if isinstance(result, Sweep1DResult):
        return {
            ROLE_SIGNAL: result.signal[idx],
            ROLE_FIT_CURVE: result.fit_curve[idx],
            ROLE_FIT_VALUE: result.fit_value[idx],
            ROLE_SNR: result.snr[idx],
        }
    if isinstance(result, Sweep2DResult):
        return {
            ROLE_SIGNAL: result.signal[idx],
            ROLE_BEST_FREQ: result.best_freq[idx],
            ROLE_BEST_GAIN: result.best_gain[idx],
        }
    raise TypeError(f"unsupported autofluxdep Result type {type(result).__name__}")


def load_node_result(
    path: str, node_type: str
) -> QubitFreqResult | Sweep1DResult | Sweep2DResult:
    """Load a node HDF5 file back into its typed sweep Result."""
    del node_type
    grouped = load_grouped_labber_data(path)
    roles = grouped.roles
    role_set = set(roles)
    if role_set == _QUBIT_FREQ_ROLES:
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
    if role_set == _SWEEP1D_ROLES:
        signal = roles[ROLE_SIGNAL]
        fit_curve = roles[ROLE_FIT_CURVE]
        fit_value = roles[ROLE_FIT_VALUE]
        snr = roles[ROLE_SNR]
        signal_z = _real_data(signal, ROLE_SIGNAL)
        _require_ndim(ROLE_SIGNAL, signal_z, 2)
        x = _axis_values(signal, ROLE_SIGNAL, 0)
        flux = _axis_values(signal, ROLE_SIGNAL, 1)
        fit_curve_z = _matching_data(
            fit_curve, ROLE_FIT_CURVE, signal_z.shape, (x, flux)
        )
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
    if role_set == _SWEEP2D_ROLES:
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
    present = ", ".join(sorted(str(role) for role in role_set))
    raise ValueError(f"unsupported autofluxdep node result roles: {present}")


def read_result_row(
    path: str,
    node_type: str,
    flux_idx: int,
) -> Mapping[DatasetRole, np.ndarray | float]:
    """Read one committed-or-nan Result row from a node HDF5 file."""
    result = load_node_result(path, node_type)
    idx = int(flux_idx)
    if isinstance(result, QubitFreqResult):
        return {
            ROLE_SIGNAL: result.signal[idx].copy(),
            ROLE_FIT_CURVE: result.fit_curve[idx].copy(),
            ROLE_FIT_FREQ: float(result.fit_freq[idx]),
            ROLE_PREDICT_FREQ: float(result.predict_freq[idx]),
            ROLE_SNR: float(result.snr[idx]),
        }
    if isinstance(result, Sweep1DResult):
        return {
            ROLE_SIGNAL: result.signal[idx].copy(),
            ROLE_FIT_CURVE: result.fit_curve[idx].copy(),
            ROLE_FIT_VALUE: float(result.fit_value[idx]),
            ROLE_SNR: float(result.snr[idx]),
        }
    return {
        ROLE_SIGNAL: result.signal[idx].copy(),
        ROLE_BEST_FREQ: float(result.best_freq[idx]),
        ROLE_BEST_GAIN: float(result.best_gain[idx]),
    }


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
    "result_row_role_names",
    "result_role_specs",
    "write_result_row",
]
