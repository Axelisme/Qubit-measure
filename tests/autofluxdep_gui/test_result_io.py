"""Autofluxdep Result role/schema Labber IO tests."""

from __future__ import annotations

from typing import cast

import h5py
import numpy as np
import pytest
from zcu_tools.gui.app.autofluxdep.nodes.result import (
    QubitFreqResult,
    Sweep1DResult,
    Sweep2DResult,
)
from zcu_tools.gui.app.autofluxdep.services.result_io import (
    ROLE_BEST_FREQ,
    ROLE_FIT_CURVE,
    ROLE_FIT_FREQ,
    ROLE_FIT_VALUE,
    ROLE_PREDICT_FREQ,
    ROLE_SIGNAL,
    ROLE_SNR,
    load_node_result,
    read_result_row,
    result_declaration,
    result_progress_summary,
    result_role_specs,
    result_row_role_names,
    result_row_summary,
    write_result_row,
)
from zcu_tools.utils.datasaver import (
    Axis,
    LabberPayload,
    StreamingGroupedLabberWriter,
    open_streaming_grouped_labber_data,
    save_grouped_labber_data,
)


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


def test_qubit_freq_result_role_specs_and_row_roundtrip(tmp_path):
    result = QubitFreqResult.allocate(
        np.array([0.0, 0.5], dtype=float),
        np.array([-5.0, 0.0, 5.0], dtype=float),
    )
    result.signal[1] = [1.0, 2.0, 3.0]
    result.fit_curve[1] = [1.5, 2.5, 3.5]
    result.fit_freq[1] = 5001.0
    result.predict_freq[1] = 5000.0
    result.snr[1] = 42.0

    specs = result_role_specs("qf", "qubit_freq", result, flux_unit="A")
    assert [str(spec.role) for spec in specs] == [
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
        roles = write_result_row(writer, "qf", "qubit_freq", result, 1)
        writer.flush()

    assert ROLE_SIGNAL in {type(ROLE_SIGNAL)(role) for role in roles}
    row = read_result_row(path, "qubit_freq", 1)
    np.testing.assert_allclose(row[ROLE_SIGNAL], [1.0, 2.0, 3.0])
    assert row[ROLE_FIT_FREQ] == 5001.0
    assert row[ROLE_PREDICT_FREQ] == 5000.0

    loaded = load_node_result(path, "qubit_freq")
    assert isinstance(loaded, QubitFreqResult)
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
    specs = result_role_specs("t1", "t1", result)
    with open_streaming_grouped_labber_data(path, specs) as writer:
        roles = write_result_row(writer, "t1", "t1", result, 0)
        writer.flush()

    assert "fit_value" in roles
    row = read_result_row(path, "t1", 0)
    np.testing.assert_allclose(row[ROLE_SIGNAL], [0.1, 0.2])
    assert row[ROLE_FIT_VALUE] == 12.0
    loaded = load_node_result(path, "t1")
    assert isinstance(loaded, Sweep1DResult)
    assert loaded.x_label == "delay time (us)"


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
    specs = result_role_specs("ro", "ro_optimize", result)
    with open_streaming_grouped_labber_data(path, specs) as writer:
        roles = write_result_row(writer, "ro", "ro_optimize", result, 1)
        writer.flush()

    assert set(roles) == {"signal", "best_freq", "best_gain"}
    row = read_result_row(path, "ro_optimize", 1)
    np.testing.assert_allclose(row[ROLE_SIGNAL], result.signal[1])
    assert row[ROLE_BEST_FREQ] == 6001.0
    loaded = load_node_result(path, "ro_optimize")
    assert isinstance(loaded, Sweep2DResult)
    np.testing.assert_allclose(loaded.gain, result.gain)


@pytest.mark.parametrize(
    ("node_type", "result_factory", "expected_type"),
    (
        ("qubit_freq", _filled_qubit_freq_result, QubitFreqResult),
        ("t1", _filled_sweep1d_result, Sweep1DResult),
        ("ro_optimize", _filled_sweep2d_result, Sweep2DResult),
    ),
)
def test_result_declaration_contract_converges_all_public_paths(
    tmp_path, node_type, result_factory, expected_type
):
    result = result_factory()
    declaration = result_declaration(result)
    row_idx = 1

    specs = result_role_specs("node", node_type, result)
    declared_roles = {str(role) for role in declaration.roles}

    assert {str(spec.role) for spec in specs} == declared_roles
    assert set(result_row_role_names(result, row_idx)) == declared_roles
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
        roles = write_result_row(writer, "node", node_type, result, row_idx)
        writer.flush()

    assert set(roles) == declared_roles
    loaded = load_node_result(path, node_type)
    assert isinstance(loaded, expected_type)
    assert result_declaration(loaded).result_type is declaration.result_type
    assert {str(role) for role in read_result_row(path, node_type, row_idx)} == (
        declared_roles
    )


def test_result_role_specs_unknown_result_fast_fails():
    with pytest.raises(TypeError, match="unsupported autofluxdep Result"):
        result_role_specs("bad", "bad", object())
    with pytest.raises(TypeError, match="unsupported autofluxdep Result"):
        result_row_role_names(object(), 0)
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


def test_load_node_result_rejects_mismatched_role_shape(tmp_path):
    flux = Axis("Flux device value", "", np.array([0.0, 1.0], dtype=float))
    detune = Axis("Detune", "MHz", np.array([-1.0, 0.0, 1.0], dtype=float))
    short_detune = Axis("Detune", "MHz", np.array([-1.0, 0.0], dtype=float))
    path = save_grouped_labber_data(
        str(tmp_path / "bad_shape"),
        {
            ROLE_SIGNAL: LabberPayload(
                Axis("Signal", "a.u.", np.zeros((2, 3))), [detune, flux]
            ),
            ROLE_FIT_CURVE: LabberPayload(
                Axis("Fit curve", "a.u.", np.zeros((2, 2))), [short_detune, flux]
            ),
            ROLE_FIT_FREQ: LabberPayload(
                Axis("Fit frequency", "MHz", np.zeros(2)), [flux]
            ),
            ROLE_PREDICT_FREQ: LabberPayload(
                Axis("Predicted frequency", "MHz", np.zeros(2)), [flux]
            ),
            ROLE_SNR: LabberPayload(Axis("SNR", "a.u.", np.zeros(2)), [flux]),
        },
    )

    with pytest.raises(ValueError, match="fit_curve"):
        load_node_result(path, "qubit_freq")


def test_load_node_result_rejects_mismatched_role_axis(tmp_path):
    flux = Axis("Flux device value", "", np.array([0.0, 1.0], dtype=float))
    shifted_flux = Axis("Flux device value", "", np.array([0.0, 2.0], dtype=float))
    detune = Axis("Detune", "MHz", np.array([-1.0, 0.0, 1.0], dtype=float))
    path = save_grouped_labber_data(
        str(tmp_path / "bad_axis"),
        {
            ROLE_SIGNAL: LabberPayload(
                Axis("Signal", "a.u.", np.zeros((2, 3))), [detune, flux]
            ),
            ROLE_FIT_CURVE: LabberPayload(
                Axis("Fit curve", "a.u.", np.zeros((2, 3))), [detune, flux]
            ),
            ROLE_FIT_FREQ: LabberPayload(
                Axis("Fit frequency", "MHz", np.zeros(2)), [shifted_flux]
            ),
            ROLE_PREDICT_FREQ: LabberPayload(
                Axis("Predicted frequency", "MHz", np.zeros(2)), [flux]
            ),
            ROLE_SNR: LabberPayload(Axis("SNR", "a.u.", np.zeros(2)), [flux]),
        },
    )

    with pytest.raises(ValueError, match="axis"):
        load_node_result(path, "qubit_freq")
