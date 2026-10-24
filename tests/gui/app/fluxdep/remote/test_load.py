"""Native spectrum loading through the guarded public Fluxdep RPC route."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TypeAlias

import h5py
import numpy as np
import pytest
from numpy.typing import NDArray
from zcu_tools.datafile import LabberData
from zcu_tools.gui.app.fluxdep.controller import Controller
from zcu_tools.gui.app.fluxdep.event_bus import SpectrumAddedPayload
from zcu_tools.gui.app.fluxdep.state import FluxDepState, SpectrumEntry
from zcu_tools.gui.event_bus import EventMeta
from zcu_tools.gui.remote.errors import ErrorCode, RemoteError
from zcu_tools.gui.remote.rpc_endpoint import ClientLink

from tests.gui.app.fluxdep.remote._route_harness import RouteHarness, RouteReply

# Native fixture: path, device values, GHz frequencies, device-major signals.
SpectrumFile: TypeAlias = tuple[
    str, NDArray[np.float64], NDArray[np.float64], NDArray[np.complex128]
]


def _observe_sources(harness: RouteHarness, client: ClientLink) -> None:
    assert harness.request(client, "project.info")["ok"]
    assert harness.request(client, "spectrum.list")["ok"]
    for name in harness.ctrl.state.spectrums:
        assert harness.request(client, "spectrum.snapshot", name=name)["ok"]


def _load_request(
    harness: RouteHarness, client: ClientLink, method: str, filepath: str
) -> RouteReply:
    params: dict[str, object] = {"filepath": filepath}
    if method == "spectrum.load":
        params["spec_type"] = "OneTone"
    return harness.request(client, method, **params)


@pytest.fixture
def processed_hdf5(spectrum_hdf5: SpectrumFile, tmp_path: Path) -> str:
    """Export native OneTone points and an empty completed TwoTone spectrum."""
    filepath, *_ = spectrum_hdf5
    ctrl = Controller(FluxDepState())
    try:
        name = ctrl.load_spectrum(filepath, "OneTone")
        ctrl.set_alignment(name, 0.0, 1.0)
        ctrl.set_points(name, np.array([0.0, 2.0]), np.array([5.0, 5.5]))
        empty_path = tmp_path / "empty.hdf5"
        shutil.copyfile(filepath, empty_path)
        empty = ctrl.load_spectrum(str(empty_path), "TwoTone")
        ctrl.set_alignment(empty, 1.0, 2.0)
        ctrl.set_points(empty, np.empty(0), np.empty(0))
        return ctrl.export_spectrums(str(tmp_path / "processed.hdf5"))
    finally:
        ctrl.interactive.dispose()


def test_raw_load_preserves_basename_replacement_and_inherited_calibration(
    route_harness: RouteHarness,
    spectrum_hdf5: SpectrumFile,
    tmp_path: Path,
) -> None:
    filepath, dev_values, freqs, signals = spectrum_hdf5
    ctrl = route_harness.ctrl
    name = ctrl.load_spectrum(filepath, "OneTone")
    ctrl.set_alignment(name, 1.0, 2.0)
    ctrl.set_points(name, np.array([0.0]), np.array([5.0]))
    ctrl.set_active_spectrum(name)
    same_basename = tmp_path / "other" / name
    same_basename.parent.mkdir()
    shutil.copyfile(filepath, same_basename)
    client = route_harness.client()
    _observe_sources(route_harness, client)
    set_version = ctrl.state.version.get("spectrums:__set__")
    facts: list[tuple[str, EventMeta]] = []

    def record(payload: SpectrumAddedPayload, meta: EventMeta) -> None:
        facts.append((payload.name, meta))

    subscription = ctrl.bus.subscribe_with_meta(SpectrumAddedPayload, record)
    try:
        loaded = route_harness.request(
            client,
            "spectrum.load",
            filepath=str(same_basename),
            spec_type="TwoTone",
            inherit_from=name,
        )
    finally:
        subscription.unsubscribe()
    assert loaded["ok"] is True
    assert loaded["result"] == {"name": name}
    assert len(facts) == 1
    assert facts[0][0] == name
    assert facts[0][1].origin.kind == "agent"
    assert facts[0][1].origin.client_id is not None
    entry = ctrl.state.get_spectrum(name)
    assert list(ctrl.state.spectrums) == [name]
    assert ctrl.state.active_spectrum == name
    assert ctrl.state.version.get("spectrums:__set__") == set_version
    assert entry.spec_type == "TwoTone"
    assert entry.alignment_seeded and not entry.aligned
    assert not entry.points_completed and entry.point_count == 0
    assert (entry.flux_half, entry.flux_int, entry.flux_period) == (1.0, 2.0, 2.0)
    np.testing.assert_allclose(entry.raw["dev_values"], dev_values)
    np.testing.assert_allclose(entry.raw["freqs"], freqs)
    np.testing.assert_allclose(entry.raw["signals"], signals)
    np.testing.assert_allclose(entry.raw["fluxs"], (dev_values - 1.0) / 2.0 + 0.5)


@pytest.mark.parametrize(
    "optional", [{}, {"inherit_from": None, "transpose_axes": None}]
)
def test_raw_load_omission_and_null_use_native_identity_defaults(
    route_harness: RouteHarness,
    spectrum_hdf5: SpectrumFile,
    optional: dict[str, object],
) -> None:
    client = route_harness.client()
    _observe_sources(route_harness, client)
    loaded = route_harness.request(
        client,
        "spectrum.load",
        filepath=spectrum_hdf5[0],
        spec_type="OneTone",
        **optional,
    )
    assert loaded["ok"] is True
    entry = route_harness.ctrl.state.get_spectrum("Q1_flux_1.hdf5")
    assert entry.spec_type == "OneTone"
    assert not entry.alignment_seeded
    assert (entry.flux_half, entry.flux_int, entry.flux_period) == (0.0, 0.0, 1.0)
    np.testing.assert_allclose(entry.raw["fluxs"], spectrum_hdf5[1] + 0.5)


def test_raw_load_transpose_recovers_native_axes_and_signals(
    route_harness: RouteHarness, transposed_spectrum_hdf5: SpectrumFile
) -> None:
    filepath, dev_values, freqs, signals = transposed_spectrum_hdf5
    client = route_harness.client()
    _observe_sources(route_harness, client)
    loaded = route_harness.request(
        client,
        "spectrum.load",
        filepath=filepath,
        spec_type="OneTone",
        transpose_axes=True,
    )
    assert loaded["ok"] is True
    entry = route_harness.ctrl.state.get_spectrum("legacy_flux_1.hdf5")
    np.testing.assert_allclose(entry.raw["dev_values"], dev_values)
    np.testing.assert_allclose(entry.raw["freqs"], freqs)
    np.testing.assert_allclose(entry.raw["signals"], signals)


def test_processed_load_restores_empty_completion_points_and_legacy_type(
    route_harness: RouteHarness, processed_hdf5: str
) -> None:
    # Legacy processed groups have no type attribute; native restore uses TwoTone.
    with h5py.File(processed_hdf5, "r+") as stored:
        del stored["empty.hdf5"].attrs["type"]
    client = route_harness.client()
    _observe_sources(route_harness, client)
    loaded = route_harness.request(
        client, "spectrum.load_processed", filepath=processed_hdf5
    )
    assert loaded["ok"] is True
    assert loaded["result"] == {"names": ["Q1_flux_1.hdf5", "empty.hdf5"]}
    for name, spec_type, frequencies in (
        ("Q1_flux_1.hdf5", "OneTone", [5.0, 5.5]),
        ("empty.hdf5", "TwoTone", []),
    ):
        entry = route_harness.ctrl.state.get_spectrum(name)
        assert entry.aligned and entry.points_completed and entry.alignment_seeded
        assert entry.spec_type == spec_type
        assert entry.flux_period == 2.0
        np.testing.assert_allclose(entry.points["freqs"], frequencies)
        snapshot = route_harness.request(client, "spectrum.snapshot", name=name)
        assert snapshot["ok"] is True
        assert snapshot["result"]["points_completed"] is True
    assert route_harness.ctrl.state.active_spectrum is None


@pytest.mark.parametrize("failure", ["unknown_inheritance", "non_2d", "missing_file"])
def test_raw_load_keeps_nominal_owner_errors_and_unexpected_io_failures(
    route_harness: RouteHarness,
    spectrum_hdf5: SpectrumFile,
    tmp_path: Path,
    failure: str,
) -> None:
    filepath = spectrum_hdf5[0]
    params: dict[str, object] = {}
    if failure == "unknown_inheritance":
        params["inherit_from"] = "unknown"
    elif failure == "non_2d":
        filepath = LabberData(
            ("Signal", "a.u.", np.ones(3, dtype=np.complex128)),
            [("Device", "native", np.arange(3))],
        ).save(str(tmp_path / "trace"))
    else:
        filepath = str(tmp_path / "missing.hdf5")
    client = route_harness.client()
    _observe_sources(route_harness, client)
    before = route_harness.ctrl.state.version.snapshot()
    rejected = route_harness.request(
        client, "spectrum.load", filepath=filepath, spec_type="OneTone", **params
    )
    assert rejected["ok"] is False
    error = rejected["error"]
    if failure == "missing_file":
        assert error["code"] == "controller_error"
        assert error.get("reason") is None
    else:
        assert error["code"] == "invalid_params"
        assert error.get("reason") == (
            "unknown_spectrum"
            if failure == "unknown_inheritance"
            else "spectrum_not_2d"
        )
    assert route_harness.ctrl.state.version.snapshot() == before
    assert route_harness.ctrl.state.spectrums == {}
    # No observation was lost on failed admission.
    assert route_harness.request(
        client, "spectrum.load", filepath=spectrum_hdf5[0], spec_type="OneTone"
    )["ok"]


@pytest.mark.parametrize("failure", ["missing_file", "malformed_processed"])
def test_processed_io_and_parse_failures_remain_unexpected_without_publication(
    route_harness: RouteHarness, processed_hdf5: str, tmp_path: Path, failure: str
) -> None:
    filepath = str(tmp_path / "missing.hdf5")
    if failure == "malformed_processed":
        filepath = processed_hdf5
        with h5py.File(filepath, "r+") as stored:
            del stored["empty.hdf5/points"]
    client = route_harness.client()
    _observe_sources(route_harness, client)
    rejected = route_harness.request(
        client, "spectrum.load_processed", filepath=filepath
    )
    assert rejected["ok"] is False
    assert rejected["error"]["code"] == "controller_error"
    assert rejected["error"].get("reason") is None
    assert route_harness.ctrl.state.spectrums == {}
    assert route_harness.ctrl.state.version.snapshot() == {}


@pytest.mark.parametrize("method", ["spectrum.load", "spectrum.load_processed"])
@pytest.mark.parametrize("filepath", [None, "", 42, False, []])
def test_malformed_load_path_cannot_publish_or_consume_observations(
    route_harness: RouteHarness,
    spectrum_hdf5: SpectrumFile,
    processed_hdf5: str,
    method: str,
    filepath: object,
) -> None:
    client = route_harness.client()
    _observe_sources(route_harness, client)
    params: dict[str, object] = {"filepath": filepath}
    if method == "spectrum.load":
        params["spec_type"] = "OneTone"
    with pytest.raises(RemoteError) as failure:
        route_harness.request(client, method, **params)
    assert failure.value.code is ErrorCode.INVALID_PARAMS
    assert route_harness.ctrl.state.version.snapshot() == {}
    assert route_harness.ctrl.state.spectrums == {}
    valid_path = spectrum_hdf5[0] if method == "spectrum.load" else processed_hdf5
    assert _load_request(route_harness, client, method, valid_path)["ok"]


@pytest.mark.parametrize(
    "extra",
    [
        {"spec_type": "invalid"},
        {"spec_type": ""},
        {"spec_type": None},
        {"inherit_from": 1},
        {"transpose_axes": 1},
        {"transpose_axes": "false"},
    ],
)
def test_malformed_raw_options_fail_before_native_load(
    route_harness: RouteHarness, spectrum_hdf5: SpectrumFile, extra: dict[str, object]
) -> None:
    client = route_harness.client()
    _observe_sources(route_harness, client)
    params: dict[str, object] = {"filepath": spectrum_hdf5[0], "spec_type": "OneTone"}
    params.update(extra)
    with pytest.raises(RemoteError) as failure:
        route_harness.request(client, "spectrum.load", **params)
    assert failure.value.code is ErrorCode.INVALID_PARAMS
    assert route_harness.ctrl.state.version.snapshot() == {}
    assert route_harness.ctrl.state.spectrums == {}


def test_processed_partial_publication_is_not_rolled_back_or_self_refreshed(
    route_harness: RouteHarness,
    spectrum_hdf5: SpectrumFile,
    processed_hdf5: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = route_harness.ctrl.state
    name = route_harness.ctrl.load_spectrum(spectrum_hdf5[0], "TwoTone")
    client = route_harness.client()
    _observe_sources(route_harness, client)
    before = state.version.get(f"spectrum:{name}")
    route_harness.ctrl.remove_spectrum(name)
    put_spectrum = FluxDepState.put_spectrum

    def fail_second_publication(target: FluxDepState, entry: SpectrumEntry) -> None:
        if entry.name == "empty.hdf5":
            raise RuntimeError("injected publication failure")
        put_spectrum(target, entry)

    # Keep native loading and the first State publication. Inject only the
    # second publication failure to expose the route's partial-write contract.
    with monkeypatch.context() as patch:
        patch.setattr(FluxDepState, "put_spectrum", fail_second_publication)
        rejected = route_harness.request(
            client, "spectrum.load_processed", filepath=processed_hdf5
        )
    assert rejected["ok"] is False
    assert rejected["error"]["code"] == "controller_error"
    assert list(state.spectrums) == [name]
    assert state.get_spectrum(name).points_completed
    assert state.get_spectrum(name).spec_type == "OneTone"
    assert state.version.get(f"spectrum:{name}") == before + 1
    # The failed handler retained the committed prefix, not a rollback.
    _observe_sources(route_harness, client)
    route_harness.ctrl.remove_spectrum(name)
    assert route_harness.request(
        client, "spectrum.load_processed", filepath=processed_hdf5
    )["ok"]


def test_rpc_load_updates_the_list_without_opening_a_picker(
    route_harness: RouteHarness, spectrum_hdf5: SpectrumFile, qapp
) -> None:
    from zcu_tools.gui.app.fluxdep.ui.interactive.result_preview import (
        ResultPreviewWidget,
    )
    from zcu_tools.gui.app.fluxdep.ui.main_window import MainWindow

    from tests.gui._dialog_fakes import RecordingDialogPresenter
    from tests.gui.app.fluxdep._window import editor, list_labels, stage_status

    ctrl = route_harness.ctrl
    window = MainWindow(
        ctrl, dialog_presenter=RecordingDialogPresenter(confirm_answers=[True])
    )
    try:
        client = route_harness.client()
        _observe_sources(route_harness, client)
        reply = _load_request(route_harness, client, "spectrum.load", spectrum_hdf5[0])
        assert reply["ok"] is True, reply
        name = reply["result"]["name"]
        assert isinstance(name, str)
        assert any(label.startswith(name) for label in list_labels(window))
        assert ctrl.interactive.inspect(name) is None
        ctrl.set_active_spectrum(name)
        # Active only displays: the agent's spectrum shows its published preview.
        preview = editor(window)
        assert isinstance(preview, ResultPreviewWidget)
        assert stage_status(window, "align") == "open"
        assert stage_status(window, "sample") == "locked"
        assert ctrl.interactive.inspect(name) is None
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()
