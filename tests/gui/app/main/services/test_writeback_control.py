"""WritebackControlFacet public contract tests."""

from __future__ import annotations

from typing import Any, cast

from zcu_tools.gui.app.main.services.writeback_control import WritebackControlFacet

from tests.gui._control_fakes import CallLog, call


class RecordingState:
    def __init__(self, log: CallLog) -> None:
        self._log = log

    def has_tab(self, tab_id: str) -> bool:
        self._log.add("state", "has_tab", tab_id)
        return tab_id == "tab-1"


class RecordingGuard:
    def __init__(self, log: CallLog) -> None:
        self._log = log

    def acquire_writeback_permit(self, tab_id: str) -> str:
        self._log.add("guard", "acquire_writeback_permit", tab_id)
        return f"permit:{tab_id}"


class RecordingWriteback:
    def __init__(self, log: CallLog) -> None:
        self._log = log
        self.items = [object()]

    def get_tab_writeback_items(self, tab_id: str) -> list[object]:
        self._log.add("writeback", "get_tab_writeback_items", tab_id)
        return self.items

    def set_item_field(
        self, tab_id: str, session_id: str, **changes: Any
    ) -> dict[str, object]:
        self._log.add("writeback", "set_item_field", tab_id, session_id, **changes)
        return {"valid": True, "removed": [], "added": []}

    def apply_tab_writeback(self, permit: object) -> dict[str, Any]:
        self._log.add("writeback", "apply_tab_writeback", permit)
        return {
            "applied_ids": ["md-1"],
            "written": {"md": ["r_f"], "ml_modules": [], "ml_waveforms": []},
        }


def _facet() -> tuple[
    WritebackControlFacet,
    CallLog,
    RecordingState,
    RecordingWriteback,
    dict[str, int],
]:
    log = CallLog()
    state = RecordingState(log)
    writeback = RecordingWriteback(log)
    versions = {"context": 7}
    return (
        WritebackControlFacet(
            state=cast(Any, state),
            guard=cast(Any, RecordingGuard(log)),
            writeback=cast(Any, writeback),
            resource_versions=lambda: versions,
        ),
        log,
        state,
        writeback,
        versions,
    )


def test_has_tab_reads_state() -> None:
    facet, log, _state, _writeback, _versions = _facet()

    assert facet.has_tab("tab-1") is True

    assert log.calls == [call("state", "has_tab", "tab-1")]


def test_get_tab_writeback_items_reads_persistent_draft() -> None:
    facet, log, _state, writeback, _versions = _facet()

    assert facet.get_tab_writeback_items("tab-1") == writeback.items

    assert log.calls == [call("writeback", "get_tab_writeback_items", "tab-1")]


def test_set_writeback_item_checks_permit_then_updates_item() -> None:
    facet, log, _state, _writeback, _versions = _facet()

    assert facet.set_writeback_item(
        "tab-1", "md-1", selected=False, proposed_value=1.25
    ) == {
        "valid": True,
        "removed": [],
        "added": [],
    }

    assert log.calls == [
        call("guard", "acquire_writeback_permit", "tab-1"),
        call(
            "writeback",
            "set_item_field",
            "tab-1",
            "md-1",
            selected=False,
            proposed_value=1.25,
        ),
    ]


def test_apply_writeback_checks_permit_then_applies_draft() -> None:
    facet, log, _state, _writeback, _versions = _facet()

    assert facet.apply_writeback("tab-1") == {
        "applied_ids": ["md-1"],
        "written": {"md": ["r_f"], "ml_modules": [], "ml_waveforms": []},
    }

    assert log.calls == [
        call("guard", "acquire_writeback_permit", "tab-1"),
        call("writeback", "apply_tab_writeback", "permit:tab-1"),
    ]


def test_get_context_version_reads_resource_versions() -> None:
    facet, _log, _state, _writeback, versions = _facet()

    assert facet.get_context_version() == 7
    versions.clear()
    assert facet.get_context_version() == 0
