"""A1/A4: AutoFlux snapshots the authoritative A/V unit once at run creation.

Run creation resolves the selected flux device's strict unit exactly once and
persists it immutably in the manifest/workflow snapshot and report; pause,
resume and later lifecycle changes never re-resolve or re-read live device
state. Runs whose device cannot supply an authoritative unit (bare sweeps,
missing devices, unsupported units) still run under existing policy but carry
an empty unit and cannot produce a flat-v2 sample table export.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest
from zcu_tools.gui.app.autofluxdep.app import build_core
from zcu_tools.gui.app.autofluxdep.nodes.io import Patch
from zcu_tools.gui.app.autofluxdep.orchestrator import InfoStore
from zcu_tools.gui.app.autofluxdep.services.run_setup import resolve_flux_device_unit
from zcu_tools.gui.app.autofluxdep.services.run_store import load_manifest
from zcu_tools.gui.app.autofluxdep.services.sample_table_export import (
    export_sample_table_from_artifact,
)
from zcu_tools.gui.app.autofluxdep.state import ProjectInfo

from ._helpers import connect_mock, make_measurement_builder


def _project(tmp_path: Path) -> ProjectInfo:
    return ProjectInfo(
        chip_name="chip",
        qub_name="q1",
        result_dir=str(tmp_path),
        database_path=str(tmp_path / "Database" / "chip" / "q1"),
        params_path=str(tmp_path / "params.json"),
    )


def _fresh_controller(tmp_path: Path):
    ctrl = build_core(project=_project(tmp_path))
    ctrl.add_node(make_measurement_builder("probe"))
    ctrl.set_flux_values([0.0])
    return ctrl


def _resolver(unit: str) -> tuple[list[str], Callable[[str], str]]:
    calls: list[str] = []

    def resolver(name: str) -> str:
        calls.append(name)
        return unit

    return calls, resolver


def test_resolve_flux_device_unit_bare_never_calls_resolver() -> None:
    calls, resolver = _resolver("A")
    assert resolve_flux_device_unit(None, resolver) == ""
    assert resolve_flux_device_unit("", resolver) == ""
    assert resolve_flux_device_unit("", None) == ""
    assert calls == []


def test_resolve_flux_device_unit_accepts_only_authoritative_a_v() -> None:
    calls, resolver = _resolver("V")
    assert resolve_flux_device_unit("yoko", resolver) == "V"
    assert calls == ["yoko"]


@pytest.mark.parametrize("unit", ["none", "mA", "X", ""])
def test_resolve_flux_device_unit_rejects_unsupported_units(unit: str) -> None:
    calls, resolver = _resolver(unit)
    assert resolve_flux_device_unit("yoko", resolver) == ""
    assert calls == ["yoko"]


def test_resolve_flux_device_unit_missing_device_fails_open() -> None:
    def missing(name: str) -> str:
        raise RuntimeError(f"No such device: {name}")

    assert resolve_flux_device_unit("ghost", missing) == ""


def test_create_run_session_resolves_strict_unit_exactly_once(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctrl = _fresh_controller(tmp_path)
    ctrl.set_flux_device("fake_flux")
    calls, resolver = _resolver("A")
    monkeypatch.setattr(ctrl._dev_svc, "get_device_unit_strict", resolver)
    session = ctrl._create_run_session(None)
    try:
        assert calls == ["fake_flux"]
        manifest = load_manifest(session.store.manifest_path)
        assert manifest["workflow"]["flux"]["unit"] == "A"
        assert manifest["workflow"]["flux"]["device_name"] == "fake_flux"
    finally:
        session.store.close_writers()


def test_snapshot_immutable_and_export_never_rereads_device(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctrl = _fresh_controller(tmp_path)
    connect_mock(ctrl)
    ctrl.set_flux_device("fake_flux")
    mode = {"value": "current"}
    calls: list[str] = []

    def resolver(name: str) -> str:
        calls.append(name)
        return "V" if mode["value"] == "voltage" else "A"

    monkeypatch.setattr(ctrl._dev_svc, "get_device_unit_strict", resolver)
    session = ctrl._create_run_session(None)
    store = session.store
    try:
        store.write_node_row("probe", 0, Patch({"qubit_freq": 5001.25}), InfoStore())
        store.commit_flux(0, 0.25, InfoStore())
        mode["value"] = "voltage"  # live mode flip after the snapshot
        store.flush_live()
        store.finalize("finished")
        manifest = load_manifest(store.manifest_path)
        assert manifest["workflow"]["flux"]["unit"] == "A"
        report = (store.run_dir / "report.md").read_text(encoding="utf-8")
        assert "unit: A" in report
        result = export_sample_table_from_artifact(store.run_dir)
        df = pd.read_csv(result.path)
        assert len(df) == 1
        assert df.loc[0, "dev_value"] == 0.25
        assert df.loc[0, "dev_unit"] == "A"
        assert df.loc[0, "Freq (MHz)"] == 5001.25
        assert calls == ["fake_flux"]  # never re-resolved, even for export
    finally:
        store.close_writers()


def test_pause_resume_finalize_keep_snapshot_unit_unchanged(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctrl = _fresh_controller(tmp_path)
    ctrl.set_flux_device("fake_flux")
    calls, resolver = _resolver("A")
    monkeypatch.setattr(ctrl._dev_svc, "get_device_unit_strict", resolver)
    session = ctrl._create_run_session(None)
    store = session.store
    try:
        assert calls == ["fake_flux"]
        store.mark_paused(0)
        store.mark_running(0)
        store.finalize("stopped")
        assert load_manifest(store.manifest_path)["workflow"]["flux"]["unit"] == "A"
        assert calls == ["fake_flux"]
    finally:
        store.close_writers()


def test_fake_flux_unsupported_snapshots_empty_unit(tmp_path) -> None:
    ctrl = _fresh_controller(tmp_path)
    connect_mock(ctrl)
    ctrl.set_flux_device("fake_flux")
    session = ctrl._create_run_session(None)
    try:
        manifest = load_manifest(session.store.manifest_path)
        assert manifest["workflow"]["flux"]["unit"] == ""
        assert manifest["workflow"]["flux"]["device_name"] == "fake_flux"
    finally:
        session.store.close_writers()


def test_missing_device_snapshots_empty_unit(tmp_path) -> None:
    ctrl = _fresh_controller(tmp_path)
    ctrl.set_flux_device("ghost_device")
    session = ctrl._create_run_session(None)
    try:
        assert (
            load_manifest(session.store.manifest_path)["workflow"]["flux"]["unit"] == ""
        )
    finally:
        session.store.close_writers()


def test_bare_sweep_snapshots_empty_unit_without_calling_resolver(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctrl = _fresh_controller(tmp_path)
    calls, resolver = _resolver("A")
    monkeypatch.setattr(ctrl._dev_svc, "get_device_unit_strict", resolver)
    session = ctrl._create_run_session(None)
    try:
        assert calls == []
        assert (
            load_manifest(session.store.manifest_path)["workflow"]["flux"]["unit"] == ""
        )
    finally:
        session.store.close_writers()
