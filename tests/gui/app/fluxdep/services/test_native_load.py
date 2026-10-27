"""Native variable/axis/device selections enter through the app load service."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from zcu_tools.analysis.flux import ComponentFluxContext, DevicePeriod, FluxFrame
from zcu_tools.analysis.fluxdep.io import load_spectrums
from zcu_tools.datafile import JsonObject, load, save
from zcu_tools.gui.app.analysis_project.native import NativeSpectrumSelection
from zcu_tools.gui.app.fluxdep.services.export import ExportService
from zcu_tools.gui.app.fluxdep.services.load import LoadService
from zcu_tools.gui.app.fluxdep.state import FluxDepState

from tests.gui.app.analysis_project._native import native_spectrum_file


@pytest.fixture
def native_target(tmp_path, fluxonium_entry):
    from zcu_tools.gui.app.analysis_project.entry import AnalysisTarget

    state = FluxDepState(AnalysisTarget.open(tmp_path, "synthetic", "Q1"))
    context = ComponentFluxContext(
        FluxFrame(
            0.25, {"bias": DevicePeriod(2.0, "A"), "coil": DevicePeriod(0.01, "A")}
        ),
        "bias",
        {"bias": "A", "coil": "A"},
    )
    return state, LoadService(state, flux_context_provider=lambda _component: context)


@pytest.mark.parametrize("snapshot_unit", ["mA", "native", None])
def test_other_snapshot_unit_mismatch_refuses_publication(
    tmp_path, native_target, snapshot_unit
):
    scalar: JsonObject = {"value": 0.002}
    if snapshot_unit is not None:
        scalar["value_unit"] = snapshot_unit
    path = native_spectrum_file(tmp_path / "mismatch.h5", {"coil": scalar})
    state, service = native_target
    with pytest.raises(ValueError, match=f"coil.*{snapshot_unit or 'native'}.*A"):
        service.load_spectrum(
            str(path),
            "OneTone",
            selection=NativeSpectrumSelection("signal", "bias", "bias"),
        )
    assert state.spectrums == {}


@pytest.mark.parametrize("scalar", [{"value": True}, {}, {"value": "invalid"}])
def test_scanned_snapshot_scalar_must_be_valid_for_storage(
    tmp_path, native_target, scalar
):
    path = native_spectrum_file(tmp_path / "invalid-scanned.h5", {"bias": scalar})
    state, service = native_target
    with pytest.raises(ValueError, match="bias"):
        service.load_spectrum(
            str(path),
            "OneTone",
            selection=NativeSpectrumSelection("signal", "bias", "bias"),
        )
    assert state.spectrums == {}


@pytest.mark.parametrize("snapshot", [None, {}])
def test_named_context_preserves_absent_snapshot_scalars(
    tmp_path, native_target, snapshot
):
    path = native_spectrum_file(tmp_path / "absent.h5", snapshot)
    state, service = native_target
    name = service.load_spectrum(
        str(path),
        "OneTone",
        selection=NativeSpectrumSelection("signal", "bias", "bias"),
    )
    assert state.get_spectrum(name).device_snapshot == snapshot


def test_uncalibrated_context_preserves_unused_snapshot_units(tmp_path, native_target):
    path = native_spectrum_file(
        tmp_path / "uncalibrated.h5",
        {"coil": {"value": 0.002, "value_unit": "mA"}},
    )
    state, _service = native_target
    context = ComponentFluxContext(None, "bias", {"bias": "A", "coil": "A"})
    service = LoadService(state, flux_context_provider=lambda _component: context)
    name = service.load_spectrum(
        str(path), "OneTone", selection=NativeSpectrumSelection("signal", "bias")
    )
    entry = state.get_spectrum(name)
    assert entry.device == "unnamed_device"
    assert entry.device_snapshot == {"coil": {"value": 0.002, "unit": "mA"}}


def test_native_target_uses_injected_context_and_device_owner_snapshot(
    tmp_path, native_target
):
    path = native_spectrum_file(
        tmp_path / "data.h5",
        {
            "bias": {"value": 99.0, "value_unit": "mA"},
            "coil": {"value": 0.002, "value_unit": "A"},
            "rf": {"freq_Hz": 1e9},
        },
    )
    state, service = native_target
    name = service.load_spectrum(
        str(path),
        "OneTone",
        selection=NativeSpectrumSelection("signal", "bias", "bias"),
    )
    entry = state.get_spectrum(name)
    assert entry.device == "bias"
    assert entry.unit == "A"
    assert entry.device_snapshot == {
        "bias": {"value": 99.0, "unit": "mA"},
        "coil": {"value": 0.002, "unit": "A"},
    }
    output = ExportService(state).export_spectrums(str(tmp_path / "processed.hdf5"))
    restored = FluxDepState()
    LoadService(restored).load_processed_spectrums(output)
    assert restored.get_spectrum(name).device == "bias"
    assert restored.get_spectrum(name).unit == "A"
    assert restored.get_spectrum(name).device_snapshot == entry.device_snapshot


@pytest.mark.parametrize("snapshot", [None, {}])
def test_native_no_target_preserves_unrecorded_vs_empty_snapshot(tmp_path, snapshot):
    path = native_spectrum_file(tmp_path / "data.h5", snapshot)
    state = FluxDepState()
    name = LoadService(state).load_spectrum(
        str(path),
        "OneTone",
        selection=NativeSpectrumSelection("signal", "bias"),
    )
    entry = state.get_spectrum(name)
    assert entry.device == "unnamed_device"
    assert entry.unit == "A"
    assert entry.device_snapshot == snapshot
    np.testing.assert_allclose(entry.raw["freqs"], [4, 5, 6])
    np.testing.assert_array_equal(entry.raw["signals"], [[1j, 2j, 3j], [4j, 5j, 6j]])
    output = ExportService(state).export_spectrums(str(tmp_path / "processed.hdf5"))
    restored = load_spectrums(output)[name]
    assert ("device_snapshot" in restored) == (snapshot is not None)


@pytest.mark.parametrize("metadata", [None, {"run_id": "invalid"}])
def test_missing_metadata_is_unrecorded_but_invalid_metadata_refuses_publication(
    tmp_path, metadata
):
    path = native_spectrum_file(tmp_path / "data.h5", None)
    source = load(path)
    save(
        path,
        replace(source, docs={} if metadata is None else {"metadata": metadata}),
        replace=True,
    )
    state = FluxDepState()
    service = LoadService(state)
    if metadata is None:
        name = service.load_spectrum(
            str(path), "OneTone", selection=NativeSpectrumSelection("signal", "bias")
        )
        assert state.get_spectrum(name).device_snapshot is None
    else:
        with pytest.raises(ValueError, match="/docs/metadata"):
            service.load_spectrum(
                str(path),
                "OneTone",
                selection=NativeSpectrumSelection("signal", "bias"),
            )
        assert state.spectrums == {}


@pytest.mark.parametrize("reject", [False, True])
def test_context_errors_name_component_and_kind_without_schema_fallback(
    tmp_path, fluxonium_entry, reject
):
    from zcu_tools.gui.app.analysis_project.entry import AnalysisTarget

    def reject_kind(_component):
        raise ValueError("unsupported schema")

    state = FluxDepState(AnalysisTarget.open(tmp_path, "synthetic", "Q1"))
    service = LoadService(state, flux_context_provider=reject_kind if reject else None)
    with pytest.raises(ValueError, match="Q1.*qubit/fluxonium"):
        service.flux_context()
