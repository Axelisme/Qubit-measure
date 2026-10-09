"""Entry RPC binding through real socket, setup and context owners."""

from pathlib import Path

import pytest
from zcu_tools.gui.session.context_types import EntryDraft, EntryInfo, EntryRequest
from zcu_tools.gui.session.types import ContextReadiness
from zcu_tools.resources.entry import ResultEntry

from ._point_application import PointApplication


def test_empty_entry_queries_and_explicit_new_publish_draft(
    application: PointApplication, tmp_path: Path
) -> None:
    assert application.result("entry.info", {}) == {"entry": None}
    (tmp_path / "result").mkdir()
    assert application.result("entry.list", {}) == {"entries": []}
    reply = application.result("entry.new", {"name": "cooldown-A"})
    info = application.controller.setup_control.get_entry_info()
    assert info is not None
    assert reply == {
        "entry_id": info.entry_id,
        "name": "cooldown-A",
        "result_path": str(tmp_path / "result" / "cooldown-A"),
        "database_path": str(tmp_path / "Database" / "cooldown-A"),
    }
    assert application.result("entry.info", {}) == {"entry": reply}
    assert application.result("entry.list", {}) == {"entries": ["cooldown-A"]}
    assert application.state.session_env.readiness == ContextReadiness.DRAFT
    capture = application.controller.context_control.read_context()
    assert capture.target == EntryDraft(entry_id=info.entry_id)
    assert capture.document().components == {}
    assert application.result("point.list", {}) == {"entry": reply, "points": []}
    assert application.state.session_env.soc is None
    assert application.state.session_env.soccfg is None
    assert application.controller.device_control.list_devices() == []


def test_open_existing_entry_clears_point_and_new_point_uses_current_entry(
    application: PointApplication, tmp_path: Path
) -> None:
    first = application.result("entry.new", {"name": "first"})
    application.result("point.new", {"label": "bias"})
    second = ResultEntry.create(
        "arbitrary-name",
        result_root=tmp_path / "result",
        database_root=tmp_path / "Database",
    )
    second.new_point("existing")
    assert application.result("entry.list", {}) == {
        "entries": ["arbitrary-name", "first"]
    }
    opened = application.result("entry.open", {"name": "arbitrary-name"})
    assert opened["entry_id"] == second.entry_id
    assert application.state.session_env.readiness == ContextReadiness.DRAFT
    assert application.controller.context_control.read_context().target == EntryDraft(
        entry_id=second.entry_id
    )
    selected = application.result("point.use", {"label": "existing"})
    assert selected == {
        "kind": "saved_point",
        "entry_id": second.entry_id,
        "point": "existing",
    }
    created = application.result("point.new", {"label": "next"})
    assert created["entry_id"] == second.entry_id
    application.result("entry.open", {"name": "first"})
    assert application.result("entry.info", {}) == {"entry": first}
    assert application.state.session_env.readiness == ContextReadiness.DRAFT
    assert application.result("point.list", {})["points"] == [
        {"label": "bias", "description": ""}
    ]


@pytest.mark.parametrize(
    ("method", "params"),
    [
        ("entry.new", {"name": "first"}),
        ("entry.open", {"name": "missing"}),
        ("entry.new", {"name": "../invalid"}),
        ("entry.open", {"name": "../invalid"}),
        ("entry.new", {"name": ""}),
        ("entry.open", {"name": 12}),
        ("entry.new", {}),
    ],
)
def test_rejected_entry_request_preserves_selected_point(
    application: PointApplication, method: str, params: dict[str, object]
) -> None:
    application.result("entry.new", {"name": "first"})
    application.result("point.new", {"label": "selected"})
    before = application.controller.context_control.read_context()
    info = application.result("entry.info", {})
    rejected = application.rpc(method, params)
    assert rejected["ok"] is False, rejected
    assert application.result("entry.info", {}) == info
    assert application.controller.context_control.read_context().target == before.target
    assert application.state.session_env.readiness == ContextReadiness.ACTIVE
    assert application.result("entry.list", {}) == {"entries": ["first"]}


def test_new_install_failure_keeps_created_entry_for_open_recovery(
    application: PointApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from zcu_tools.gui.session.services.context import ContextService

    first = application.controller.setup_control.new_entry(EntryRequest("first"))
    original = ContextService.install_entry

    def fail_install(self: ContextService, entry: ResultEntry) -> EntryInfo:
        raise OSError("installation unavailable")

    monkeypatch.setattr(ContextService, "install_entry", fail_install)
    failed = application.rpc("entry.new", {"name": "retained"})
    assert failed["ok"] is False
    error = failed["error"]
    assert isinstance(error, dict)
    assert "Open" in str(error["message"])
    assert application.controller.setup_control.get_entry_info() == first
    retained = ResultEntry.open(
        "retained", result_root=tmp_path / "result", database_root=tmp_path / "Database"
    )
    monkeypatch.setattr(ContextService, "install_entry", original)
    opened = application.result("entry.open", {"name": "retained"})
    assert opened["entry_id"] == retained.entry_id
    assert application.state.session_env.readiness == ContextReadiness.DRAFT


def test_discovery_io_error_is_not_reported_as_cached_success(
    application: PointApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened = application.result("entry.new", {"name": "first"})
    assert application.result("entry.list", {}) == {"entries": ["first"]}

    def failed_discovery(*, result_root: Path) -> tuple[str, ...]:
        raise OSError("entry discovery unavailable")

    monkeypatch.setattr(ResultEntry, "list_names", staticmethod(failed_discovery))
    failed = application.rpc("entry.list", {})
    assert failed["ok"] is False
    error = failed["error"]
    assert isinstance(error, dict)
    assert error["code"] == "controller_error"
    assert error["message"] == "entry discovery unavailable"
    assert application.result("entry.info", {}) == {"entry": opened}
