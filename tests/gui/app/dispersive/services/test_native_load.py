"""One-tone axes use the selected predictor record and same-source metadata."""

from dataclasses import replace

import numpy as np
import pytest
from zcu_tools.datafile import load, save
from zcu_tools.gui.app.analysis_project.native import NativeSpectrumSelection
from zcu_tools.gui.app.dispersive.services.load import LoadService
from zcu_tools.gui.app.dispersive.state import DispersiveState, FluxoniumInputs
from zcu_tools.resources.predictor_file import PredictorFlux

from tests.gui.app.analysis_project._native import native_spectrum_file


@pytest.fixture
def native_state():
    state = DispersiveState()
    state.set_fit_inputs(
        FluxoniumInputs(
            params=(4.0, 1.0, 0.5),
            flux=PredictorFlux(
                1.25, {"bias": 0.002, "coil": -0.01}, {"bias": "A", "coil": "A"}
            ),
            bare_rf_seed=5.3,
        )
    )
    return state


def test_native_projection_uses_recorded_frame_and_ignores_scanned_snapshot(
    tmp_path, native_state
):
    path = native_spectrum_file(
        tmp_path / "sweep.h5",
        {
            "bias": {"value": "not a scalar"},
            "coil": {"value": 0.002, "value_unit": "A"},
            "rf": {"freq_Hz": 1e9},
        },
    )
    state = native_state
    LoadService(state).load_onetone(
        str(path), selection=NativeSpectrumSelection("signal", "bias", "bias")
    )
    assert state.onetone is not None
    assert (state.onetone.device, state.onetone.unit) == ("bias", "A")
    np.testing.assert_allclose(state.onetone.raw["fluxs"], [1.55, 2.05])
    np.testing.assert_allclose(state.onetone.raw["freqs"], [4.0, 5.0, 6.0])
    np.testing.assert_array_equal(
        state.onetone.raw["signals"], [[1j, 2j, 3j], [4j, 5j, 6j]]
    )


@pytest.mark.parametrize(
    "scalar",
    [
        {"value": True},
        {},
        {"value": "invalid"},
        {"value": 0.002, "value_unit": "V"},
    ],
)
def test_invalid_other_device_is_named_before_publication(
    tmp_path, native_state, scalar
):
    path = native_spectrum_file(tmp_path / "invalid.h5", {"coil": scalar})
    with pytest.raises(ValueError, match="coil"):
        LoadService(native_state).load_onetone(
            str(path), selection=NativeSpectrumSelection("signal", "bias", "bias")
        )
    assert native_state.onetone is None


def test_snapshot_absent_devices_contribute_zero(tmp_path, native_state):
    path = native_spectrum_file(tmp_path / "empty.h5", {})
    LoadService(native_state).load_onetone(
        str(path), selection=NativeSpectrumSelection("signal", "bias", "bias")
    )
    assert native_state.onetone is not None
    np.testing.assert_allclose(native_state.onetone.raw["fluxs"], [1.75, 2.25])


@pytest.mark.parametrize("snapshot", [None, {}])
def test_unknown_source_device_is_named(tmp_path, native_state, snapshot):
    path = native_spectrum_file(tmp_path / "unknown.h5", snapshot)
    with pytest.raises(ValueError, match="unknown"):
        LoadService(native_state).load_onetone(
            str(path), selection=NativeSpectrumSelection("signal", "bias", "unknown")
        )
    assert native_state.onetone is None


@pytest.mark.parametrize("metadata", [None, {"run_id": "invalid"}])
def test_metadata_absence_is_distinct_from_invalid_metadata(
    tmp_path, native_state, metadata
):
    path = native_spectrum_file(tmp_path / "metadata.h5", None)
    source = load(path)
    save(
        path,
        replace(source, docs={} if metadata is None else {"metadata": metadata}),
        replace=True,
    )
    message = "snapshot.*bias.*coil" if metadata is None else "/docs/metadata"
    with pytest.raises(ValueError, match=message):
        LoadService(native_state).load_onetone(
            str(path), selection=NativeSpectrumSelection("signal", "bias", "bias")
        )
    assert native_state.onetone is None


def test_historical_native_single_device_keeps_unnamed_source(tmp_path):
    path = native_spectrum_file(tmp_path / "history.h5", None)
    state = DispersiveState()
    state.set_fit_inputs(
        FluxoniumInputs(
            (4.0, 1.0, 0.5), PredictorFlux(1.25, {"bias": -0.002}, {"bias": "A"}), 5.3
        )
    )
    LoadService(state).load_onetone(
        str(path), selection=NativeSpectrumSelection("signal", "bias", "bias")
    )
    assert state.onetone is not None
    assert state.onetone.device == "unnamed_device"
    np.testing.assert_allclose(state.onetone.raw["fluxs"], [0.75, 0.25])


def test_native_axis_unit_is_checked_for_the_unique_unrecorded_device(tmp_path):
    path = native_spectrum_file(tmp_path / "wrong-unit.h5", None)
    state = DispersiveState()
    state.set_fit_inputs(
        FluxoniumInputs(
            (4.0, 1.0, 0.5), PredictorFlux(0.0, {"bias": 2.0}, {"bias": "V"}), 5.3
        )
    )
    with pytest.raises(ValueError, match="bias.*unit.*A.*V"):
        LoadService(state).load_onetone(
            str(path), selection=NativeSpectrumSelection("signal", "bias")
        )
    assert state.onetone is None


def test_native_requires_explicit_variable_axis_and_instrument(tmp_path, native_state):
    path = native_spectrum_file(tmp_path / "explicit.h5", {})
    service = LoadService(native_state)
    with pytest.raises(ValueError, match="variable.*device_axis.*device"):
        service.load_onetone(str(path))
    with pytest.raises(ValueError, match="missing.*signal.*bias.*frequency"):
        service.load_onetone(
            str(path), selection=NativeSpectrumSelection("missing", "bias", "bias")
        )
    assert native_state.onetone is None


def test_explicit_variable_and_reversed_axes_reach_the_shared_projection(
    tmp_path, native_state
):
    path = native_spectrum_file(tmp_path / "multiple.h5", {})
    container = load(path)
    variable = container.vars["signal"]
    save(
        path,
        replace(
            container,
            vars={
                "signal": variable,
                "second": replace(
                    variable,
                    values=variable.values.T * 2,
                    axes=tuple(reversed(variable.axes)),
                ),
            },
        ),
        replace=True,
    )
    LoadService(native_state).load_onetone(
        str(path),
        selection=NativeSpectrumSelection("second", "bias", "bias"),
        transpose_axes=True,
    )
    assert native_state.onetone is not None
    np.testing.assert_array_equal(
        native_state.onetone.raw["signals"], [[2j, 4j, 6j], [8j, 10j, 12j]]
    )
