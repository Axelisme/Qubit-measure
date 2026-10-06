"""Autofluxdep Result variable/schema Labber IO tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal, cast

import h5py
import numpy as np
import pytest
from numpy.typing import NDArray
from zcu_tools.datafile import (
    Axis,
    LabberPayload,
    StreamingGroupedLabberWriter,
    StreamingLabberVariableSpec,
    load_labber_data,
    open_streaming_grouped_labber_data,
)
from zcu_tools.gui.app.autofluxdep.results import (
    FrequencySweepResult,
    SweepResult1D,
    SweepResult2D,
)
from zcu_tools.gui.app.autofluxdep.services.fluxdep_export import (
    export_qubit_freq_fluxdep_spectrum,
)
from zcu_tools.gui.app.autofluxdep.services.labber_browser_export import (
    export_qubit_freq_labber_browser_sidecar,
)
from zcu_tools.gui.app.autofluxdep.services.result_io import (
    VARIABLE_BEST_FREQ,
    VARIABLE_FIT_CURVE,
    VARIABLE_FIT_FREQ,
    VARIABLE_FIT_VALUE,
    VARIABLE_PREDICT_FREQ,
    VARIABLE_SIGNAL,
    VARIABLE_SNR,
    load_node_result,
    read_result_row,
    result_declaration,
    result_progress_summary,
    result_row_summary,
    result_row_variable_names,
    result_variable_specs,
    write_result_row,
)

from zcu_lab.v2._support.autofluxdep.result import (
    QubitFreqResult,
    Sweep1DResult,
    Sweep2DResult,
)


@dataclass
class _ForeignFrequencySweep:
    result_kind: ClassVar[Literal["qubit_freq"]] = "qubit_freq"
    flux: NDArray[np.float64]
    detune: NDArray[np.float64]
    signal: NDArray[np.float64]
    fit_curve: NDArray[np.float64]
    fit_freq: NDArray[np.float64]
    predict_freq: NDArray[np.float64]
    snr: NDArray[np.float64]


class _FrequencyDeclaration:
    result_kind: ClassVar[Literal["qubit_freq"]] = "qubit_freq"


class _Sweep1DDeclaration:
    result_kind: ClassVar[Literal["sweep1d"]] = "sweep1d"


class _Sweep2DDeclaration:
    result_kind: ClassVar[Literal["sweep2d"]] = "sweep2d"


class _UnknownDeclaration:
    result_kind: ClassVar[Literal["unknown"]] = "unknown"


def _filled_qubit_freq_result() -> QubitFreqResult:
    result = QubitFreqResult.allocate(
        np.array([0.0, 0.5], dtype=float),
        np.array([-5.0, 0.0, 5.0], dtype=float),
    )
    result.signal[1] = [1.0, 2.0, 3.0]
    result.fit_curve[1] = [1.5, 2.5, 3.5]
    result.fit_freq[1] = 5001.0
    result.predict_freq[1] = 5000.0
    result.snr[1] = 42.0
    return result


def _filled_sweep1d_result() -> Sweep1DResult:
    result = Sweep1DResult.allocate(
        np.array([0.0, 1.0], dtype=float),
        np.array([10.0, 20.0], dtype=float),
        x_label="delay time (us)",
    )
    result.signal[1] = [0.1, 0.2]
    result.fit_curve[1] = [0.3, 0.4]
    result.fit_value[1] = 12.0
    result.snr[1] = 5.0
    return result


def _filled_sweep2d_result() -> Sweep2DResult:
    result = Sweep2DResult.allocate(
        np.array([0.0, 1.0], dtype=float),
        np.array([6000.0, 6001.0], dtype=float),
        np.array([0.1, 0.2, 0.3], dtype=float),
    )
    result.signal[1] = np.arange(6, dtype=float).reshape(2, 3)
    result.best_freq[1] = 6001.0
    result.best_gain[1] = 0.2
    return result


@pytest.mark.parametrize(
    ("declaration", "kind"),
    (
        (_FrequencyDeclaration, "qubit_freq"),
        (_Sweep1DDeclaration, "sweep1d"),
        (_Sweep2DDeclaration, "sweep2d"),
    ),
)
def test_class_declaration_lookup_needs_no_arrays(declaration, kind):
    assert result_declaration(declaration).kind == kind


def test_external_tagged_result_uses_same_archive_contract(tmp_path):
    source = _filled_qubit_freq_result()
    result = _ForeignFrequencySweep(
        source.flux,
        source.detune,
        source.signal,
        source.fit_curve,
        source.fit_freq,
        source.predict_freq,
        source.snr,
    )
    assert isinstance(result, FrequencySweepResult)

    path = str(tmp_path / "foreign")
    specs = result_variable_specs("foreign", "user_measurement", result)
    with open_streaming_grouped_labber_data(path, specs) as writer:
        write_result_row(writer, "foreign", "user_measurement", result, 1)
        writer.flush()
    loaded = load_node_result(path, "user_measurement")

    assert isinstance(loaded, FrequencySweepResult)
    np.testing.assert_allclose(loaded.signal, result.signal, equal_nan=True)
    np.testing.assert_allclose(loaded.fit_freq, result.fit_freq, equal_nan=True)
    assert result_row_summary(loaded, 1) == result_row_summary(result, 1)
    assert result_progress_summary(loaded) == result_progress_summary(result)

    exported = export_qubit_freq_fluxdep_spectrum(
        loaded,
        tmp_path / "foreign-spectrum",
        committed_mask=np.array([False, True]),
    )
    spectrum = load_labber_data(exported)
    np.testing.assert_allclose(spectrum.axes[0].values, result.flux)
    np.testing.assert_allclose(spectrum.z.real[:, 1], result.signal[1])
    assert np.isnan(spectrum.z.real[:, 0]).all()

    root = tmp_path / "20260705-223908_foreign"
    sidecar = export_qubit_freq_labber_browser_sidecar(
        data_root=root,
        index=0,
        node_name="foreign",
        node_type="user_measurement",
        result=loaded,
        committed_mask=np.array([False, True]),
    )
    browser_spectrum = load_labber_data(str(root / sidecar.path))
    np.testing.assert_allclose(browser_spectrum.z, spectrum.z, equal_nan=True)
    np.testing.assert_allclose(browser_spectrum.axes[0].values, result.flux)


@pytest.mark.parametrize("declaration", [object, _UnknownDeclaration])
def test_unsupported_class_kind_fast_fails(declaration):
    with pytest.raises(TypeError, match="unsupported autofluxdep Result"):
        result_declaration(declaration)


def test_tagged_instance_without_arrays_fast_fails():
    with pytest.raises(TypeError, match="Result|Protocol|field"):
        result_variable_specs("missing", "custom", _FrequencyDeclaration())


@pytest.mark.parametrize(
    "result_factory",
    [_filled_qubit_freq_result, _filled_sweep1d_result, _filled_sweep2d_result],
)
def test_result_io_rejects_inconsistent_signal_shape(result_factory):
    result = result_factory()
    result.signal = result.signal[:1]
    with pytest.raises(ValueError, match="shape"):
        result_variable_specs("bad", "custom", result)


@pytest.mark.parametrize(
    "result_factory",
    [_filled_qubit_freq_result, _filled_sweep1d_result, _filled_sweep2d_result],
)
def test_result_io_rejects_non_float64_signal(result_factory):
    result = result_factory()
    result.signal = result.signal.astype(np.float32)
    with pytest.raises(TypeError, match="float64|dtype"):
        result_variable_specs("bad", "custom", result)


@pytest.mark.parametrize("idx", [-1, 2])
def test_result_row_rejects_out_of_range_index(idx):
    result = _filled_qubit_freq_result()
    with pytest.raises(IndexError):
        result_row_summary(result, idx)


def test_qubit_freq_result_variable_specs_and_row_roundtrip(tmp_path):
    result = QubitFreqResult.allocate(
        np.array([0.0, 0.5], dtype=float),
        np.array([-5.0, 0.0, 5.0], dtype=float),
    )
    result.signal[1] = [1.0, 2.0, 3.0]
    result.fit_curve[1] = [1.5, 2.5, 3.5]
    result.fit_freq[1] = 5001.0
    result.predict_freq[1] = 5000.0
    result.snr[1] = 42.0

    specs = result_variable_specs("qf", "qubit_freq", result, flux_unit="A")
    assert [str(spec.variable) for spec in specs] == [
        "signal",
        "fit_curve",
        "fit_freq",
        "predict_freq",
        "snr",
    ]
    assert specs[0].axes[0].name == "Detune"
    assert specs[0].axes[1].unit == "A"

    path = str(tmp_path / "qf")
    with open_streaming_grouped_labber_data(path, specs) as writer:
        variables = write_result_row(writer, "qf", "qubit_freq", result, 1)
        writer.flush()

    assert VARIABLE_SIGNAL in {
        type(VARIABLE_SIGNAL)(variable) for variable in variables
    }
    row = read_result_row(path, "qubit_freq", 1)
    np.testing.assert_allclose(row[VARIABLE_SIGNAL], [1.0, 2.0, 3.0])
    assert row[VARIABLE_FIT_FREQ] == 5001.0
    assert row[VARIABLE_PREDICT_FREQ] == 5000.0

    loaded = load_node_result(path, "qubit_freq")
    assert isinstance(loaded, FrequencySweepResult)
    np.testing.assert_allclose(loaded.detune, result.detune)
    np.testing.assert_allclose(loaded.signal[1], result.signal[1])

    with h5py.File(path + ".hdf5", "r") as handle:
        assert handle.attrs["zcu_tools.autofluxdep.node_name"] == "qf"
        assert handle.attrs["zcu_tools.autofluxdep.result_role"] == "signal"


def test_sweep1d_result_row_roundtrip(tmp_path):
    result = Sweep1DResult.allocate(
        np.array([0.0, 1.0], dtype=float),
        np.array([10.0, 20.0], dtype=float),
        x_label="delay time (us)",
    )
    result.signal[0] = [0.1, 0.2]
    result.fit_curve[0] = [0.3, 0.4]
    result.fit_value[0] = 12.0
    result.snr[0] = 5.0

    path = str(tmp_path / "sweep1d")
    specs = result_variable_specs("t1", "t1", result)
    with open_streaming_grouped_labber_data(path, specs) as writer:
        variables = write_result_row(writer, "t1", "t1", result, 0)
        writer.flush()

    assert "fit_value" in variables
    row = read_result_row(path, "t1", 0)
    np.testing.assert_allclose(row[VARIABLE_SIGNAL], [0.1, 0.2])
    assert row[VARIABLE_FIT_VALUE] == 12.0
    loaded = load_node_result(path, "t1")
    assert isinstance(loaded, SweepResult1D)
    assert loaded.x_label == "delay time (us)"
    with h5py.File(path + ".hdf5", "r") as handle:
        assert handle.attrs["zcu_tools.autofluxdep.result_kind"] == "sweep_1d"


def test_sweep2d_result_row_roundtrip(tmp_path):
    result = Sweep2DResult.allocate(
        np.array([0.0, 1.0], dtype=float),
        np.array([6000.0, 6001.0], dtype=float),
        np.array([0.1, 0.2, 0.3], dtype=float),
    )
    result.signal[1] = np.arange(6, dtype=float).reshape(2, 3)
    result.best_freq[1] = 6001.0
    result.best_gain[1] = 0.2

    path = str(tmp_path / "sweep2d")
    specs = result_variable_specs("ro", "ro_optimize", result)
    with open_streaming_grouped_labber_data(path, specs) as writer:
        variables = write_result_row(writer, "ro", "ro_optimize", result, 1)
        writer.flush()

    assert set(variables) == {"signal", "best_freq", "best_gain"}
    row = read_result_row(path, "ro_optimize", 1)
    np.testing.assert_allclose(row[VARIABLE_SIGNAL], result.signal[1])
    assert row[VARIABLE_BEST_FREQ] == 6001.0
    loaded = load_node_result(path, "ro_optimize")
    assert isinstance(loaded, SweepResult2D)
    np.testing.assert_allclose(loaded.gain, result.gain)
    with h5py.File(path + ".hdf5", "r") as handle:
        assert handle.attrs["zcu_tools.autofluxdep.result_kind"] == "sweep_2d"


@pytest.mark.parametrize(
    ("node_type", "result_factory", "expected_type"),
    (
        ("qubit_freq", _filled_qubit_freq_result, FrequencySweepResult),
        ("t1", _filled_sweep1d_result, SweepResult1D),
        ("ro_optimize", _filled_sweep2d_result, SweepResult2D),
    ),
)
def test_result_declaration_contract_converges_all_public_paths(
    tmp_path, node_type, result_factory, expected_type
):
    result = result_factory()
    declaration = result_declaration(result)
    row_idx = 1

    specs = result_variable_specs("node", node_type, result)
    declared_variables = {str(variable) for variable in declaration.variables}

    assert {str(spec.variable) for spec in specs} == declared_variables
    assert set(result_row_variable_names(result, row_idx)) == declared_variables
    assert set(result_row_summary(result, row_idx)) == set(
        declaration.summary_scalar_attrs
    )

    progress = result_progress_summary(result)
    assert progress["kind"] == declaration.kind
    assert {output_key for output_key, _attr in declaration.last_fit_fields}.issubset(
        progress["fit_summary"]
    )

    path = str(tmp_path / f"{node_type}_contract")
    with open_streaming_grouped_labber_data(path, specs) as writer:
        variables = write_result_row(writer, "node", node_type, result, row_idx)
        writer.flush()

    assert set(variables) == declared_variables
    loaded = load_node_result(path, node_type)
    assert isinstance(loaded, expected_type)
    assert result_declaration(loaded).kind == declaration.kind
    assert {
        str(variable) for variable in read_result_row(path, node_type, row_idx)
    } == (declared_variables)


def test_result_variable_specs_unknown_result_fast_fails():
    with pytest.raises(TypeError, match="unsupported autofluxdep Result"):
        result_variable_specs("bad", "bad", object())
    with pytest.raises(TypeError, match="unsupported autofluxdep Result"):
        result_row_variable_names(object(), 0)
    writer = cast(StreamingGroupedLabberWriter, object())
    with pytest.raises(TypeError, match="unsupported autofluxdep Result"):
        write_result_row(writer, "bad", "bad", object(), 0)


def test_result_progress_summary_counts_raw_rows_separately_from_fits():
    result = QubitFreqResult.allocate(
        np.array([0.0, 0.5], dtype=float),
        np.array([-5.0, 0.0, 5.0], dtype=float),
    )
    result.signal[0] = [1.0, np.nan, np.nan]

    progress = result_progress_summary(result)

    assert progress["kind"] == "qubit_freq"
    assert progress["n_flux"] == 2
    assert progress["n_measured"] == 1
    assert progress["fit_summary"]["n_fitted"] == 0
    assert progress["fit_summary"]["last_fit_freq"] is None
    assert result_row_summary(result, 0)["fit_freq"] is None

    result.fit_freq[0] = 5000.0
    progress = result_progress_summary(result)
    assert progress["n_measured"] == 1
    assert progress["fit_summary"]["n_fitted"] == 1
    assert progress["fit_summary"]["last_fit_freq"] == pytest.approx(5000.0)


def _save_streaming_payloads(path: str, variables: dict[str, LabberPayload]) -> str:
    specs = [
        StreamingLabberVariableSpec(
            variable,
            payload.data.name,
            payload.data.unit,
            payload.axes,
            np.asarray(payload.z).shape,
        )
        for variable, payload in variables.items()
    ]
    with open_streaming_grouped_labber_data(path, specs):
        pass
    return path + ".hdf5"


def test_load_node_result_rejects_mismatched_variable_shape(tmp_path):
    flux = Axis("Flux device value", "", np.array([0.0, 1.0], dtype=float))
    detune = Axis("Detune", "MHz", np.array([-1.0, 0.0, 1.0], dtype=float))
    short_detune = Axis("Detune", "MHz", np.array([-1.0, 0.0], dtype=float))
    path = _save_streaming_payloads(
        str(tmp_path / "bad_shape"),
        {
            VARIABLE_SIGNAL: LabberPayload(
                Axis("Signal", "a.u.", np.zeros((2, 3))), [detune, flux]
            ),
            VARIABLE_FIT_CURVE: LabberPayload(
                Axis("Fit curve", "a.u.", np.zeros((2, 2))), [short_detune, flux]
            ),
            VARIABLE_FIT_FREQ: LabberPayload(
                Axis("Fit frequency", "MHz", np.zeros(2)), [flux]
            ),
            VARIABLE_PREDICT_FREQ: LabberPayload(
                Axis("Predicted frequency", "MHz", np.zeros(2)), [flux]
            ),
            VARIABLE_SNR: LabberPayload(Axis("SNR", "a.u.", np.zeros(2)), [flux]),
        },
    )

    with pytest.raises(ValueError, match="fit_curve"):
        load_node_result(path, "qubit_freq")


def test_load_node_result_rejects_mismatched_variable_axis(tmp_path):
    flux = Axis("Flux device value", "", np.array([0.0, 1.0], dtype=float))
    shifted_flux = Axis("Flux device value", "", np.array([0.0, 2.0], dtype=float))
    detune = Axis("Detune", "MHz", np.array([-1.0, 0.0, 1.0], dtype=float))
    path = _save_streaming_payloads(
        str(tmp_path / "bad_axis"),
        {
            VARIABLE_SIGNAL: LabberPayload(
                Axis("Signal", "a.u.", np.zeros((2, 3))), [detune, flux]
            ),
            VARIABLE_FIT_CURVE: LabberPayload(
                Axis("Fit curve", "a.u.", np.zeros((2, 3))), [detune, flux]
            ),
            VARIABLE_FIT_FREQ: LabberPayload(
                Axis("Fit frequency", "MHz", np.zeros(2)), [shifted_flux]
            ),
            VARIABLE_PREDICT_FREQ: LabberPayload(
                Axis("Predicted frequency", "MHz", np.zeros(2)), [flux]
            ),
            VARIABLE_SNR: LabberPayload(Axis("SNR", "a.u.", np.zeros(2)), [flux]),
        },
    )

    with pytest.raises(ValueError, match="axis"):
        load_node_result(path, "qubit_freq")
