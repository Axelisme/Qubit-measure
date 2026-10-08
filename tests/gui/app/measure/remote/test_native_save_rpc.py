"""Native Save RPC reports through the real connection-owned dispatcher."""

import socket
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import pytest
from qtpy.QtWidgets import QApplication
from zcu_tools.datafile import SoftwareProvenance, load_run_data
from zcu_tools.experiment.context import RunContext
from zcu_tools.gui.app.measure.adapter import SessionEnv
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.remote import ControlOptions, RemoteControlAdapter
from zcu_tools.gui.app.measure.remote.save_types import SaveSnapshotWire
from zcu_tools.gui.cfg.edit_codec import encode_ref
from zcu_tools.gui.session.context_control import ContextControlFacet
from zcu_tools.resources.entry import AcquiredPayload

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
        session.state.update_tab_result(
            "tab", tab.run.result, source_operation_id=101, metadata=session.metadata
        )
        declare_panes(session)
        post = tab.post_analysis
        session.state.swap_analysis_pane(
            "tab", replace(tab.analysis, source_operation_id=201)
        )
        session.state.swap_post_analysis_pane(
            "tab", replace(post, source_operation_id=301)
        )
        controller = Controller(
            session.state, Registry(), None, session.bus, project_root=str(tmp_path)
        )
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


@pytest.mark.parametrize("draft_capture", [False, True])
@pytest.mark.parametrize("analysis_failure", [False, True])
def test_native_load_rpc_retains_history_on_analysis_partial_failure(
    tmp_path: Path,
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    analysis_failure: bool,
    draft_capture: bool,
) -> None:
    with native_save_rpc(tmp_path, qapp) as (session, sock):
        session.save()
        path = (
            session.entry.database_path / "runs" / session.metadata.run_id / "data.h5"
        )
        stored = load_run_data(path)
        events = session.entry.ledger.events()
        tab = session.state.get_tab("tab")
        if draft_capture:
            session.context.install_entry(session.entry)
            captured = ContextControlFacet(context=session.context).read_context()
            assert captured.storage is None
            tab.parameters = captured
            tab.save_context = captured
        if analysis_failure:

            def unavailable(
                self: AnalysisSavingAdapter, result: object, ctx: SessionEnv
            ) -> None:
                raise ValueError("analysis parameters unavailable")

            monkeypatch.setattr(
                AnalysisSavingAdapter, "get_analyze_params", unavailable
            )
        assert call(sock, "tab.snapshot", {"tab_id": "tab"})["ok"] is True

        if not draft_capture:
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
        assert tab.run.metadata == stored.metadata
        assert tab.run.source_path == str(path)
        assert tab.run.confirmed_run_id is None
        assert tab.post_analysis.result is None
        if draft_capture:
            assert tab.save_context is not None
            assert tab.save_context.storage is None
            assert tab.artifacts.snapshot() == ()
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
                "expected": encode_ref(session.state.get_tab("tab").cfg.observe().ref),
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
        metadata = session.state.get_tab("tab").run.metadata
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
def test_save_rpc_reports_partial_and_initial_all_failed_identity(
    tmp_path: Path, qapp: QApplication, block_native: bool
) -> None:
    with native_save_rpc(tmp_path, qapp) as (session, sock):
        (session.entry.database_path / "Labber").write_text("blocked")
        if block_native:
            (
                session.entry.database_path
                / "runs"
                / session.metadata.run_id
                / "data.h5"
            ).mkdir(parents=True)
        assert call(sock, "tab.snapshot", {"tab_id": "tab"})["ok"] is True
        reply = call(sock, "tab.save_data", {"tab_id": "tab"})
        assert reply["ok"] is True
        report = SaveSnapshotWire.model_validate(reply["result"])
        assert report.run_id == (None if block_native else session.metadata.run_id)
        artifact = report.artifacts["run"]["measurement_data"]
        outputs = artifact.members["data"]
        assert [output.status for output in outputs] == (
            ["failed", "failed"] if block_native else ["saved", "failed"]
        )
        assert artifact.status == ("Not saved" if block_native else "Partially saved")
        assert all(output.error for output in outputs if output.status == "failed")


@pytest.mark.parametrize("method", ["tab.save_data", "tab.save_artifacts"])
def test_save_rpc_requires_each_connections_own_snapshot(
    tmp_path: Path, qapp: QApplication, method: str
) -> None:
    with native_save_rpc(tmp_path, qapp) as (session, sock):
        session.state.update_tab_comment("tab", "unchanged")
        params = {
            "tab_id": "tab",
            "comment": "must not publish",
            "run_operation_id": 101,
        }
        unseen = call(sock, method, params)
        assert unseen["ok"] is False
        assert unseen["error"]["code"] == "precondition_failed"
        assert unseen["error"]["reason"] == "stale_version"
        assert "tab:tab:result" in unseen["error"]["data"]["stale"]
        assert "tab:tab:save" in unseen["error"]["data"]["stale"]
        assert call(sock, "tab.snapshot", {"tab_id": "tab"})["ok"] is True
        fresh = open_client(sock.getpeername()[1])
        try:
            rejected = call(fresh, method, params)
            assert rejected["ok"] is False
            assert rejected["error"]["reason"] == "stale_version"
        finally:
            reset_inbox(fresh)
            fresh.close()
        assert session.state.get_tab("tab").save.comment == "unchanged"
        assert session.entry.ledger.events() == ()
        assert not (session.entry.database_path / "runs").exists()
        assert not (session.entry.database_path / "Labber").exists()
        assert call(sock, method, params)["ok"] is True


@pytest.mark.parametrize(
    ("method", "resource"),
    [
        ("tab.save_data", "save"),
        ("tab.save_data", "result"),
        ("tab.save_artifacts", "save"),
        ("tab.save_artifacts", "result"),
        ("tab.save_artifacts", "analyze"),
        ("tab.save_artifacts", "post_analyze"),
    ],
)
def test_save_rpc_rejects_stale_snapshot_without_publishing_comment_or_outputs(
    tmp_path: Path, qapp: QApplication, method: str, resource: str
) -> None:
    with native_save_rpc(tmp_path, qapp) as (session, sock):
        session.state.update_tab_comment("tab", "local note")
        assert call(sock, "tab.snapshot", {"tab_id": "tab"})["ok"] is True
        tab = session.state.get_tab("tab")
        if resource == "save":
            session.state.update_tab_comment("tab", "new local note")
        elif resource == "result":
            session.state.update_tab_result(
                "tab",
                tab.run.result,
                source_operation_id=101,
                metadata=session.metadata,
            )
        elif resource == "analyze":
            session.state.swap_analysis_pane("tab", replace(tab.analysis))
        else:
            session.state.swap_post_analysis_pane("tab", replace(tab.post_analysis))
        reply = call(sock, method, {"tab_id": "tab", "comment": "must not publish"})
        assert reply["ok"] is False
        assert reply["error"]["code"] == "precondition_failed"
        assert reply["error"]["reason"] == "stale_version"
        assert f"tab:tab:{resource}" in reply["error"]["data"]["stale"]
        assert session.state.get_tab("tab").save.comment == (
            "new local note" if resource == "save" else "local note"
        )
        assert session.entry.ledger.events() == ()
        assert not (session.entry.database_path / "runs").exists()
        assert not (session.entry.database_path / "Labber").exists()
        assert call(sock, "tab.snapshot", {"tab_id": "tab"})["ok"] is True
        assert call(sock, method, {"tab_id": "tab"})["ok"] is True


@pytest.mark.parametrize("selection", ["data", "all", "analysis", "post_analysis"])
def test_save_rpc_rejects_replaced_sources_even_after_a_fresh_snapshot(
    tmp_path: Path, qapp: QApplication, selection: str
) -> None:
    with native_save_rpc(tmp_path, qapp) as (session, sock):
        assert call(sock, "tab.snapshot", {"tab_id": "tab"})["ok"] is True
        tab = session.state.get_tab("tab")
        if selection in ("data", "all"):
            session.state.update_tab_result(
                "tab",
                tab.run.result,
                source_operation_id=102,
                metadata=session.metadata,
            )
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


def test_save_rpc_rejects_invalid_source_token_at_request_admission(
    tmp_path: Path, qapp: QApplication
) -> None:
    with native_save_rpc(tmp_path, qapp) as (session, sock):
        assert call(sock, "tab.snapshot", {"tab_id": "tab"})["ok"] is True
        reply = call(
            sock, "tab.save_data", {"tab_id": "tab", "run_operation_id": "bad"}
        )
        assert reply["ok"] is False
        assert reply["error"]["code"] == "invalid_params"
        assert session.entry.ledger.events() == ()
        assert not (session.entry.database_path / "runs").exists()
