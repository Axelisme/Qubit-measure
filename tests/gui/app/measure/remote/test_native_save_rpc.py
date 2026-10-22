"""Native Save RPC reports through the real connection-owned dispatcher."""

from tests.gui.app.measure._record_support import require_record
import socket
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from qtpy.QtWidgets import QApplication
from zcu_tools.datafile import load_run_data
from zcu_tools.experiment.metadata import SoftwareProvenance
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import SessionEnv
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.remote import ControlOptions, RemoteControlAdapter
from zcu_tools.gui.app.measure.remote.save_types import SaveSnapshotWire
from zcu_tools.gui.session.context_types import EntryRequest, PointUseRequest
from zcu_tools.gui.session.parameters import ParameterContext
from zcu_tools.resources.entry import AcquiredPayload

from tests.experiment._saving_support import SavingCfg, SavingResult
from tests.gui.app.measure._native_save_support import (
    AnalysisSavingAdapter,
    NativeSaveSession,
    declare_panes,
    native_save_session,
)

from ._helpers import call, open_client, reset_inbox

pytestmark = pytest.mark.uses_wall_clock


@contextmanager
def native_save_rpc(
    tmp_path: Path, qapp: QApplication
) -> Generator[tuple[NativeSaveSession, socket.socket]]:
    """Serve the native session over loopback with production dispatch and guards.

    The session already owns a fixed cfg capture and real SaveControl. Bind these
    existing resources to the adapter's public facets; Controller supplies the
    remaining real controls, owner scheduler and current resource versions.
    Close the socket inbox and server before releasing the native session.
    """
    with native_save_session(tmp_path, qapp) as session:
        tab = session.state.get_tab("tab")
        session.state.update_tab_result("tab", replace(tab.run.result, metadata=session.metadata), source_operation_id=101, parameters=tab.run.parameters)
        declare_panes(session)
        post = tab.post_analysis
        session.state.swap_analysis_pane(
            "tab", replace(tab.analysis, source_operation_id=201)
        )
        session.state.swap_post_analysis_pane(
            "tab", replace(post, source_operation_id=301)
        )
        registry = Registry()
        registry.register("saving", AnalysisSavingAdapter)
        controller = Controller(
            session.state, registry, None, session.bus, project_root=str(tmp_path)
        )
        controller.setup_control.open_entry(EntryRequest("original"))
        controller.setup_control.use_point(PointUseRequest("p"))
        remote = RemoteControlAdapter(
            controller, ControlOptions(port=0), controller.owner_scheduler
        )
        remote.cfg_lookup = lambda tab_id: session.state.get_tab(tab_id).cfg
        remote.save_control = session.control
        remote.start()
        try:
            sock = open_client(remote.port)
            try:
                yield session, sock
            finally:
                reset_inbox(sock)
                sock.close()
        finally:
            remote.stop()
            closed: list[bool] = []
            controller.begin_shutdown(lambda: closed.append(True))
            assert closed == [True]
            qapp.processEvents()


def test_analysis_reports_raw_identity_without_saving_measurement_data(
    tmp_path: Path, qapp: QApplication
) -> None:
    with native_save_rpc(tmp_path, qapp) as (session, sock):
        unsaved = {"run_id": session.metadata.run_id, "saved": False}
        before = call(sock, "tab.snapshot", {"tab_id": "tab"})["result"]["tabs"][0]
        assert before.get("raw_data") == unsaved
        started = call(sock, "tab.analyze", {"tab_id": "tab"})
        assert started["ok"], started
        assert started["result"]["raw_data"] == unsaved
        operation = started["result"]["operation_id"]
        # The accepted synthetic analysis always completes; timeout only guards deadlock.
        finished = call(
            sock, "operation.await", {"operation_id": operation, "timeout": 30}
        )
        assert finished["result"]["status"] == "finished", finished
        result = call(
            sock, "tab.get_analyze_result", {"tab_id": "tab", "operation_id": operation}
        )
        assert result["result"]["operation_state"]["raw_data"] == unsaved
        saved_analysis = call(
            sock,
            "tab.save_artifacts",
            {
                "tab_id": "tab",
                "artifacts": [{"section": "analysis", "name": "results"}],
            },
        )
        assert saved_analysis["ok"], saved_analysis
        after_analysis = call(sock, "tab.snapshot", {"tab_id": "tab"})["result"][
            "tabs"
        ][0]
        assert after_analysis["run_id"] == session.metadata.run_id
        assert after_analysis["raw_data"] == unsaved
        assert not session.native_data_path.exists()
        saved_raw = call(sock, "tab.save_data", {"tab_id": "tab"})
        assert saved_raw["ok"], saved_raw
        after_raw = call(sock, "tab.snapshot", {"tab_id": "tab"})["result"]["tabs"][0]
        assert after_raw["raw_data"] == {
            "run_id": session.metadata.run_id,
            "saved": True,
        }
        assert session.native_data_path.is_file()


def install_unbound_saved_capture(session: NativeSaveSession) -> None:
    """Keep the selected point's fixed names and values without storage capability."""
    captured = session.state.session_env.parameters
    assert captured is not None
    unbound = ParameterContext(
        point=captured.document(),
        library=captured.library,
        target=captured.target,
        roles=captured.roles,
    )
    session.state.set_context(replace(session.state.session_env, parameters=unbound))


@pytest.mark.parametrize("unbound_capture", [False, True])
@pytest.mark.parametrize("analysis_failure", [False, True])
def test_native_load_rpc_retains_history_on_analysis_partial_failure(
    tmp_path: Path,
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    analysis_failure: bool,
    unbound_capture: bool,
) -> None:
    with native_save_rpc(tmp_path, qapp) as (session, sock):
        session.save()
        path = session.native_data_path
        stored = load_run_data(path)
        events = session.entry.ledger.events()
        tab = session.state.get_tab("tab")
        if unbound_capture:
            install_unbound_saved_capture(session)
            captured = session.state.session_env.parameters
            assert captured is not None and captured.storage is None
        if analysis_failure:

            def unavailable(
                self: AnalysisSavingAdapter, result: object, ctx: SessionEnv
            ) -> None:
                raise ValueError("analysis parameters unavailable")

            monkeypatch.setattr(
                AnalysisSavingAdapter, "get_analyze_params", unavailable
            )
        assert call(sock, "tab.snapshot", {"tab_id": "tab"})["ok"] is True

        if not unbound_capture:
            session.state.set_context(
                replace(session.state.session_env, parameters=None)
            )
        reply = call(sock, "tab.load_data", {"tab_id": "tab", "data_path": str(path)})

        assert reply["ok"] is True, reply
        assert reply["result"]["has_run_result"] is True
        assert reply["result"]["has_analyze_params"] is not analysis_failure
        assert reply["result"]["analysis_error"] == (
            "analysis parameters unavailable" if analysis_failure else None
        )
        tab = session.state.get_tab("tab")
        assert require_record(tab.run).metadata == stored.metadata
        assert tab.run.source_path == str(path)
        assert tab.run.confirmed_run_id is None
        assert tab.post_analysis.result is None
        before = call(sock, "tab.snapshot", {"tab_id": "tab"})["result"]
        result_state = before["tabs"][0]["result_state"]
        assert result_state["available"] is True
        assert result_state["source_path"] == str(path)
        assert result_state["source_operation_id"] is None
        for method in ("tab.get_figure", "tab.analyze"):
            params = {"tab_id": "tab", "run_operation_id": 101}
            if method == "tab.get_figure":
                params["subtab_id"] = "run"
            rejected_source = call(sock, method, params)
            assert rejected_source["ok"] is False
            assert rejected_source["error"]["reason"] == "result_superseded"
            assert call(sock, "tab.snapshot", {"tab_id": "tab"})["result"] == before
        rejected = call(
            sock,
            "tab.save_data",
            {
                "tab_id": "tab",
                "run_operation_id": 101,
                "comment": "must not publish",
            },
        )
        assert rejected["ok"] is False
        assert rejected["error"]["reason"] == "result_superseded"
        assert call(sock, "tab.snapshot", {"tab_id": "tab"})["result"] == before
        if unbound_capture:
            assert tab.run.parameters is not None
            assert tab.run.storage is None
            assert tab.artifacts.snapshot() == ()
        assert session.entry.ledger.events() == events


@pytest.mark.parametrize("unbound_capture", [False, True])
@pytest.mark.parametrize("analysis_failure", [False, True])
def test_native_open_file_rpc_retains_history_after_explicit_context_read(
    tmp_path: Path,
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    unbound_capture: bool,
    analysis_failure: bool,
) -> None:
    with native_save_rpc(tmp_path, qapp) as (session, sock):
        session.save()
        path = session.native_data_path
        history = load_run_data(path).metadata
        events = session.entry.ledger.events()
        session.state.set_active_tab("tab")
        if unbound_capture:
            install_unbound_saved_capture(session)
        if analysis_failure:

            def unavailable(
                self: AnalysisSavingAdapter, result: object, ctx: SessionEnv
            ) -> None:
                raise ValueError("analysis parameters unavailable")

            monkeypatch.setattr(
                AnalysisSavingAdapter, "get_analyze_params", unavailable
            )
        request = {"adapter_name": "saving", "data_path": str(path)}
        assert call(sock, "context.snapshot", {})["ok"] is True

        reply = call(sock, "tab.open_file", request)

        assert reply["ok"] is True, reply
        outcome = reply["result"]
        new_id = outcome["tab_id"]
        assert new_id != "tab"
        tab = session.state.get_tab(new_id)
        record = tab.run.result
        assert isinstance(record, RunRecord)
        assert isinstance(record.cfg, SavingCfg)
        assert isinstance(record.result, SavingResult)
        np.testing.assert_array_equal(record.result.signal, [1 + 2j, 3 - 4j, 5 + 6j])
        assert outcome["has_analyze_params"] is not analysis_failure
        assert outcome["analysis_error"] == (
            "analysis parameters unavailable" if analysis_failure else None
        )
        assert require_record(tab.run).metadata == history
        assert tab.run.source_path == str(path)
        assert tab.run.confirmed_run_id is None
        assert tab.run.parameters is not None
        assert (tab.run.storage is None) is unbound_capture
        if unbound_capture:
            assert tab.artifacts.snapshot() == ()
        assert session.state.active_tab_id == new_id
        assert session.entry.ledger.events() == events


@pytest.mark.parametrize("unbound_capture", [False, True])
def test_unreadable_native_open_file_restores_previous_tab_and_focus(
    tmp_path: Path, qapp: QApplication, unbound_capture: bool
) -> None:
    with native_save_rpc(tmp_path, qapp) as (session, sock):
        session.state.set_active_tab("tab")
        previous = session.state.get_tab("tab").run
        before = session.state.list_tab_ids()
        events = session.entry.ledger.events()
        if unbound_capture:
            install_unbound_saved_capture(session)
        assert call(sock, "context.snapshot", {})["ok"] is True

        reply = call(
            sock,
            "tab.open_file",
            {"adapter_name": "saving", "data_path": str(tmp_path / "missing.h5")},
        )

        assert reply["ok"] is False
        assert reply["error"]["reason"] == "data_file_read_failed"
        assert session.state.list_tab_ids() == before
        assert session.state.active_tab_id == "tab"
        assert session.state.get_tab("tab").run is previous
        assert session.entry.ledger.events() == events


def test_native_run_rpc_records_mcp_acquired_without_save(
    tmp_path: Path, qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    def offline_soc_info(
        self: Controller, include_cfg: bool = False
    ) -> dict[str, object]:
        # Hardware observation is not this offline Run/ledger contract's target.
        return {"soccfg": {}}

    monkeypatch.setattr(Controller, "get_soc_info", offline_soc_info)
    with native_save_rpc(tmp_path, qapp) as (session, sock):

        def provenance(*, context: RunContext) -> SoftwareProvenance:
            return session.metadata.provenance

        monkeypatch.setattr(
            "zcu_tools.gui.app.measure.services.run.capture_software_provenance",
            provenance,
        )
        soc_info = call(sock, "soc.info", {"include_cfg": True})
        assert soc_info["ok"] is True, soc_info
        assert call(sock, "device.list", {})["ok"] is True
        assert call(sock, "tab.snapshot", {"tab_id": "tab"})["ok"] is True
        started = call(
            sock,
            "tab.run_start",
            {
                "tab_id": "tab",
            },
        )
        assert started["ok"] is True, started
        completed = call(
            sock,
            "operation.await",
            {"operation_id": started["result"]["operation_id"], "timeout": 2},
        )
        assert completed["result"]["status"] == "finished"
        events = session.entry.ledger.events()
        assert len(events) == 1 and events[0].kind == "acquired"
        assert events[0].origin == "mcp"
        assert isinstance(events[0].payload, AcquiredPayload)
        metadata = require_record(session.state.get_tab("tab").run).metadata
        assert metadata is not None and events[0].payload.run_id == metadata.run_id
        assert not (session.entry.database_path / "runs").exists()


@pytest.mark.parametrize("selection", ["data", "all", "analysis", "post_analysis"])
def test_save_rpc_returns_the_current_complete_report(
    tmp_path: Path, qapp: QApplication, selection: str
) -> None:
    with native_save_rpc(tmp_path, qapp) as (session, sock):
        assert call(sock, "tab.snapshot", {"tab_id": "tab"})["ok"] is True
        if selection == "data":
            reply = call(
                sock, "tab.save_data", {"tab_id": "tab", "run_operation_id": 101}
            )
            expected = {"run": {"measurement_data": 2}}
        else:
            refs = (
                "all"
                if selection == "all"
                else [{"section": selection, "name": "results"}]
            )
            reply = call(
                sock,
                "tab.save_artifacts",
                {
                    "tab_id": "tab",
                    "run_operation_id": 101,
                    "operation_id": (
                        None
                        if selection == "all"
                        else 201
                        if selection == "analysis"
                        else 301
                    ),
                    "artifacts": refs,
                },
            )
            expected = (
                {
                    "run": {"measurement_data": 2},
                    "analysis": {"results": 5},
                    "post_analysis": {"results": 1},
                }
                if selection == "all"
                else {selection: {"results": 5 if selection == "analysis" else 1}}
            )
        assert reply["ok"] is True
        report = SaveSnapshotWire.model_validate(reply["result"])
        assert report.run_id == session.metadata.run_id
        assert session.state.get_tab("tab").run.confirmed_run_id == report.run_id
        assert {
            section: {
                name: sum(len(outputs) for outputs in artifact.members.values())
                for name, artifact in artifacts.items()
            }
            for section, artifacts in report.artifacts.items()
        } == expected
        for artifacts in report.artifacts.values():
            for artifact in artifacts.values():
                assert artifact.status == "Saved"
                for outputs in artifact.members.values():
                    for output in outputs:
                        assert output.status == "saved"
                        assert Path(output.path).is_file()
        assert len(session.entry.ledger.events()) == 1


@pytest.mark.parametrize("block_native", [True, False])
@pytest.mark.parametrize("method", ["tab.save_data", "tab.save_artifacts"])
def test_save_rpc_reports_partial_and_initial_all_failed_identity(
    tmp_path: Path, qapp: QApplication, block_native: bool, method: str
) -> None:
    with native_save_rpc(tmp_path, qapp) as (session, sock):
        session.entry.database_path.mkdir(parents=True, exist_ok=True)
        (session.entry.database_path / "Labber").write_text("blocked")
        if block_native:
            (session.native_data_path).mkdir(parents=True)
        assert call(sock, "tab.snapshot", {"tab_id": "tab"})["ok"] is True
        params: dict[str, object] = {"tab_id": "tab"}
        if method == "tab.save_artifacts":
            params["artifacts"] = [{"section": "run", "name": "measurement_data"}]
        reply = call(sock, method, params)
        assert reply["ok"] is True
        report = SaveSnapshotWire.model_validate(reply["result"])
        assert report.run_id == session.metadata.run_id
        snapshot = call(sock, "tab.snapshot", {"tab_id": "tab"})
        assert snapshot["ok"] is True
        assert snapshot["result"]["tabs"][0]["run_id"] == (
            None if block_native else session.metadata.run_id
        )
        artifact = report.artifacts["run"]["measurement_data"]
        outputs = artifact.members["data"]
        assert [output.status for output in outputs] == (
            ["failed", "failed"] if block_native else ["saved", "failed"]
        )
        assert artifact.status == ("Not saved" if block_native else "Partially saved")
        assert all(output.error for output in outputs if output.status == "failed")


@pytest.mark.parametrize("selection", ["data", "all", "analysis", "post_analysis"])
def test_save_rpc_rejects_replaced_sources_even_after_a_fresh_snapshot(
    tmp_path: Path, qapp: QApplication, selection: str
) -> None:
    with native_save_rpc(tmp_path, qapp) as (session, sock):
        assert call(sock, "tab.snapshot", {"tab_id": "tab"})["ok"] is True
        tab = session.state.get_tab("tab")
        if selection in ("data", "all"):
            session.state.update_tab_result("tab", replace(tab.run.result, metadata=session.metadata), source_operation_id=102, parameters=tab.run.parameters)
            params = {
                "tab_id": "tab",
                "comment": "must not publish",
                "run_operation_id": 101,
            }
        else:
            if selection == "analysis":
                session.state.swap_analysis_pane(
                    "tab", replace(tab.analysis, source_operation_id=202)
                )
            else:
                session.state.swap_post_analysis_pane(
                    "tab", replace(tab.post_analysis, source_operation_id=302)
                )
            params = {
                "tab_id": "tab",
                "comment": "must not publish",
                "artifacts": [{"section": selection, "name": "results"}],
                "operation_id": 201 if selection == "analysis" else 301,
            }
        assert call(sock, "tab.snapshot", {"tab_id": "tab"})["ok"] is True
        method = "tab.save_data" if selection == "data" else "tab.save_artifacts"
        reply = call(sock, method, params)
        assert reply["ok"] is False
        assert reply["error"]["code"] == "precondition_failed"
        assert reply["error"]["reason"] == "result_superseded"
        assert session.state.get_tab("tab").save.comment == ""
        assert session.entry.ledger.events() == ()
        assert not (session.entry.database_path / "runs").exists()
        assert not (session.entry.database_path / "Labber").exists()


def test_save_rpc_selects_primary_and_post_results_without_name_collision(
    tmp_path: Path, qapp: QApplication
) -> None:
    with native_save_rpc(tmp_path, qapp) as (session, sock):
        assert call(sock, "tab.snapshot", {"tab_id": "tab"})["ok"] is True
        reply = call(
            sock,
            "tab.save_artifacts",
            {
                "tab_id": "tab",
                "artifacts": [
                    {"section": "analysis", "name": "results"},
                    {"section": "post_analysis", "name": "results"},
                ],
            },
        )
        assert reply["ok"] is True
        report = SaveSnapshotWire.model_validate(reply["result"])
        assert report.run_id == session.metadata.run_id
        assert set(report.artifacts) == {"analysis", "post_analysis"}
        paths: set[str] = set()
        snapshot = call(sock, "tab.snapshot", {"tab_id": "tab"})["result"]["tabs"][0]
        for section, count in (("analysis", 5), ("post_analysis", 1)):
            assert set(report.artifacts[section]) == {"results"}
            artifact = report.artifacts[section]["results"]
            assert artifact.status == "Saved"
            assert snapshot["artifacts"][section]["results"] == artifact.model_dump(
                mode="json"
            )
            outputs = [
                output for members in artifact.members.values() for output in members
            ]
            assert len(outputs) == count
            for output in outputs:
                assert output.status == "saved" and output.error is None
                assert Path(output.path).is_file()
                paths.add(output.path)
        assert len(paths) == 6
        assert len(session.entry.ledger.events()) == 1


@pytest.mark.parametrize(
    "artifacts",
    [
        "other",
        2,
        [],
        [{"section": "analysis"}],
        [{"section": "analysis", "name": ""}],
        [{"section": "../analysis", "name": "results"}],
        [{"section": "analysis", "name": "results", "extra": True}],
        [{"section": "analysis", "name": 2}],
        [
            {"section": "analysis", "name": "results"},
            {"section": "analysis", "name": "results"},
        ],
    ],
)
def test_save_rpc_rejects_invalid_artifact_refs_without_changing_outputs(
    tmp_path: Path, qapp: QApplication, artifacts: object
) -> None:
    with native_save_rpc(tmp_path, qapp) as (session, sock):
        before = call(sock, "tab.snapshot", {"tab_id": "tab"})["result"]
        reply = call(
            sock,
            "tab.save_artifacts",
            {
                "tab_id": "tab",
                "artifacts": artifacts,
                "comment": "must not publish",
            },
        )
        assert reply["ok"] is False
        assert reply["error"]["code"] == "invalid_params"
        assert call(sock, "tab.snapshot", {"tab_id": "tab"})["result"] == before
        assert session.state.get_tab("tab").save.comment == ""
        assert session.entry.ledger.events() == ()
        assert not (session.entry.database_path / "runs").exists()


def test_save_rpc_keeps_its_own_save_changes_observed_without_another_snapshot(
    tmp_path: Path, qapp: QApplication
) -> None:
    with native_save_rpc(tmp_path, qapp) as (session, sock):
        assert call(sock, "tab.snapshot", {"tab_id": "tab"})["ok"] is True
        reports: list[SaveSnapshotWire] = []
        for comment in ("first", "second"):
            reply = call(
                sock,
                "tab.save_artifacts",
                {
                    "tab_id": "tab",
                    "artifacts": [{"section": "analysis", "name": "results"}],
                    "comment": comment,
                },
            )
            assert reply["ok"] is True, reply
            report = SaveSnapshotWire.model_validate(reply["result"])
            assert report.run_id == session.metadata.run_id
            assert report.artifacts["analysis"]["results"].status == "Saved"
            assert session.state.get_tab("tab").save.comment == comment
            reports.append(report)
        assert reports[0].artifacts == reports[1].artifacts
        assert len(session.entry.ledger.events()) == 2


@pytest.mark.parametrize(
    ("source_token", "code", "reason"),
    [
        ("bad", "invalid_params", None),
        (999, "precondition_failed", "result_superseded"),
    ],
)
def test_save_rpc_rejects_invalid_or_unknown_source_token(
    tmp_path: Path,
    qapp: QApplication,
    source_token: object,
    code: str,
    reason: str | None,
) -> None:
    with native_save_rpc(tmp_path, qapp) as (session, sock):
        assert call(sock, "tab.snapshot", {"tab_id": "tab"})["ok"] is True
        reply = call(
            sock,
            "tab.save_data",
            {
                "tab_id": "tab",
                "run_operation_id": source_token,
                "comment": "must not publish",
            },
        )
        assert reply["ok"] is False
        assert reply["error"]["code"] == code
        if reason is not None:
            assert reply["error"]["reason"] == reason
        assert session.state.get_tab("tab").save.comment == ""
        assert session.entry.ledger.events() == ()
        assert not (session.entry.database_path / "runs").exists()
