"""Native spectrum loading through the guarded public Fluxdep RPC route."""

from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path
from typing import TypeAlias

import h5py
import numpy as np
import pytest
from numpy.typing import NDArray
from zcu_tools.datafile import save_labber_data
from zcu_tools.gui.app.fluxdep.controller import Controller
from zcu_tools.gui.app.fluxdep.event_bus import SpectrumAddedPayload
from zcu_tools.gui.app.fluxdep.state import FluxDepState, SpectrumEntry
from zcu_tools.gui.event_bus import EventMeta
from zcu_tools.gui.project import ProjectInfo
from zcu_tools.gui.remote.errors import ErrorCode, RemoteError
from zcu_tools.gui.remote.rpc_endpoint import ClientLink

from tests.gui.app.fluxdep.remote._route_harness import RouteHarness, RouteReply

# Native fixture: path, device values, GHz frequencies, device-major signals.
SpectrumFile: TypeAlias = tuple[
    str, NDArray[np.float64], NDArray[np.float64], NDArray[np.complex128]
]


def _assert_stale(reply: RouteReply, keys: list[str]) -> None:
    assert reply["ok"] is False
    error = reply["error"]
    assert error["code"] == "precondition_failed"
    assert error.get("reason") == "stale_version"
    assert error.get("data") == {"stale": sorted(keys)}


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


@pytest.mark.parametrize("method", ["spectrum.load", "spectrum.load_processed"])
@pytest.mark.parametrize(
    "missing", ["project", "spectrums:__set__", "spectrum:Q1_flux_1.hdf5"]
)
def test_load_requires_project_collection_and_every_full_leaf_read(
    route_harness: RouteHarness,
    spectrum_hdf5: SpectrumFile,
    method: str,
    missing: str,
) -> None:
    filepath, *_ = spectrum_hdf5
    name = route_harness.ctrl.load_spectrum(filepath, "TwoTone")
    # The all-source guard includes an empty completed spectrum, not just points.
    route_harness.ctrl.set_alignment(name, 0.0, 1.0)
    route_harness.ctrl.set_points(name, np.empty(0), np.empty(0))
    client = route_harness.client()
    for key, read, params in (
        ("project", "project.info", {}),
        ("spectrums:__set__", "spectrum.list", {}),
        (f"spectrum:{name}", "spectrum.snapshot", {"name": name}),
    ):
        if key != missing:
            assert route_harness.request(client, read, **params)["ok"]
    # Bookkeeping and derived reads cannot fill a missing observation.
    for read in ("resources.versions", "state.check", "selection.pointcloud"):
        assert route_harness.request(client, read)["ok"]
    before = route_harness.ctrl.state.version.snapshot()
    _assert_stale(
        _load_request(route_harness, client, method, "missing.hdf5"), [missing]
    )
    assert route_harness.ctrl.state.version.snapshot() == before
    assert route_harness.ctrl.state.get_spectrum(name).points_completed


@pytest.mark.parametrize("method", ["spectrum.load", "spectrum.load_processed"])
def test_first_load_requires_zero_version_reads_and_does_not_observe_new_leaves(
    route_harness: RouteHarness,
    spectrum_hdf5: SpectrumFile,
    processed_hdf5: str,
    method: str,
) -> None:
    client = route_harness.client()
    filepath = spectrum_hdf5[0] if method == "spectrum.load" else processed_hdf5
    _assert_stale(
        _load_request(route_harness, client, method, filepath),
        ["project", "spectrums:__set__"],
    )
    _observe_sources(route_harness, client)
    loaded = _load_request(route_harness, client, method, filepath)
    assert loaded["ok"] is True
    names = list(route_harness.ctrl.state.spectrums)
    assert loaded["result"] == (
        {"name": names[0]} if method == "spectrum.load" else {"names": names}
    )
    # Collection self-write advanced, but receipts did not reveal new leaves.
    _assert_stale(
        _load_request(route_harness, client, method, filepath),
        [f"spectrum:{name}" for name in names],
    )
    for name in names:
        _assert_stale(
            route_harness.request(client, "spectrum.reset_alignment", name=name),
            [f"spectrum:{name}"],
        )
        assert route_harness.request(client, "spectrum.snapshot", name=name)["ok"]
    assert _load_request(route_harness, client, method, filepath)["ok"]
    # A successful replacement advances matching observations, unlike creation.
    assert _load_request(route_harness, client, method, filepath)["ok"]


@pytest.mark.parametrize("method", ["spectrum.load", "spectrum.load_processed"])
def test_all_source_guards_detect_other_clients_gui_edits_and_retired_leaves(
    route_harness: RouteHarness,
    spectrum_hdf5: SpectrumFile,
    processed_hdf5: str,
    method: str,
) -> None:
    ctrl = route_harness.ctrl
    name = ctrl.load_spectrum(spectrum_hdf5[0], "TwoTone")
    filepath = spectrum_hdf5[0] if method == "spectrum.load" else processed_hdf5
    first, second = route_harness.client(), route_harness.client()
    _observe_sources(route_harness, first)
    _assert_stale(
        _load_request(route_harness, second, method, filepath),
        ["project", "spectrums:__set__", f"spectrum:{name}"],
    )
    _observe_sources(route_harness, second)
    assert _load_request(route_harness, first, method, filepath)["ok"]
    stale = [f"spectrum:{name}"]
    if method == "spectrum.load_processed":
        stale += ["spectrums:__set__", "spectrum:empty.hdf5"]
    _assert_stale(_load_request(route_harness, second, method, filepath), stale)

    _observe_sources(route_harness, first)
    ctrl.reset_alignment(name)
    _assert_stale(
        _load_request(route_harness, first, method, filepath), [f"spectrum:{name}"]
    )
    _observe_sources(route_harness, first)
    ctrl.setup_project(ProjectInfo(chip_name="GUI", qub_name="Q2"))
    _assert_stale(_load_request(route_harness, first, method, filepath), ["project"])
    _observe_sources(route_harness, first)
    ctrl.remove_spectrum(name)
    # Wildcard includes previously seen leaves even after the owner retires them.
    _assert_stale(
        _load_request(route_harness, first, method, filepath),
        ["spectrums:__set__", f"spectrum:{name}"],
    )
    assert route_harness.request(first, "spectrum.list")["ok"]
    _assert_stale(
        _load_request(route_harness, first, method, filepath), [f"spectrum:{name}"]
    )
    absent = route_harness.request(first, "spectrum.snapshot", name=name)
    assert absent["ok"] is True
    assert absent["result"] == {"name": name, "exists": False}
    assert _load_request(route_harness, first, method, filepath)["ok"]


@pytest.mark.parametrize("method", ["spectrum.load", "spectrum.load_processed"])
def test_all_source_guard_includes_literal_star_names_and_their_neighbors(
    route_harness: RouteHarness,
    spectrum_hdf5: SpectrumFile,
    processed_hdf5: str,
    tmp_path: Path,
    method: str,
) -> None:
    path = tmp_path / "測量:one*.hdf5"
    shutil.copyfile(spectrum_hdf5[0], path)
    ctrl = route_harness.ctrl
    name = ctrl.load_spectrum(str(path), "TwoTone")
    neighbor = name + ":neighbor"
    ctrl.state.put_spectrum(replace(ctrl.state.get_spectrum(name), name=neighbor))
    client = route_harness.client()
    assert route_harness.request(client, "project.info")["ok"]
    assert route_harness.request(client, "spectrum.list")["ok"]
    assert route_harness.request(client, "spectrum.snapshot", name=name)["ok"]
    filepath = str(path) if method == "spectrum.load" else processed_hdf5
    _assert_stale(
        _load_request(route_harness, client, method, filepath),
        [f"spectrum:{neighbor}"],
    )
    assert route_harness.request(client, "spectrum.snapshot", name=neighbor)["ok"]
    assert _load_request(route_harness, client, method, filepath)["ok"]


def test_successful_load_advances_a_previously_read_full_absence(
    route_harness: RouteHarness, spectrum_hdf5: SpectrumFile
) -> None:
    client = route_harness.client()
    _observe_sources(route_harness, client)
    name = "Q1_flux_1.hdf5"
    absent = route_harness.request(client, "spectrum.snapshot", name=name)
    assert absent["ok"] is True
    assert absent["result"] == {"name": name, "exists": False}
    assert _load_request(route_harness, client, "spectrum.load", spectrum_hdf5[0])["ok"]
    # This is normal self-write tracking of the full absence, not a creation
    # receipt granting an observation to an unread new resource.
    assert route_harness.request(client, "spectrum.reset_alignment", name=name)["ok"]


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
        filepath = save_labber_data(
            str(tmp_path / "trace"),
            z=("Signal", "a.u.", np.ones(3, dtype=np.complex128)),
            axes=[("Device", "native", np.arange(3))],
        )
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
    # Failed handlers do not advance even previously matching observations.
    _assert_stale(
        route_harness.request(
            client, "spectrum.load_processed", filepath=processed_hdf5
        ),
        [f"spectrum:{name}"],
    )
    _observe_sources(route_harness, client)
    assert route_harness.request(
        client, "spectrum.load_processed", filepath=processed_hdf5
    )["ok"]
