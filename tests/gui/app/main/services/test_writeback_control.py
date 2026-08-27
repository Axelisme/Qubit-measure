"""WritebackControlFacet public contract tests."""

from __future__ import annotations

from typing import Any, cast

import pytest
from zcu_tools.gui.app.main.services.writeback_control import WritebackControlFacet
from zcu_tools.gui.expected_error import FailedPreconditionError, InvalidInputError

from tests.gui._control_fakes import CallLog, call


class RecordingState:
    def __init__(self, log: CallLog) -> None:
        self._log = log
        self.busy = False

    def has_tab(self, tab_id: str) -> bool:
        self._log.add("state", "has_tab", tab_id)
        return tab_id == "tab-1"

    def is_tab_busy(self, tab_id: str) -> bool:
        self._log.add("state", "is_tab_busy", tab_id)
        return self.busy


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
        self.analysis_draft: object | None = object()
        self.post_draft: object | None = object()

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

    def get_tab_writeback_draft(self, tab_id: str) -> object | None:
        self._log.add("writeback", "get_tab_writeback_draft", tab_id)
        return self.analysis_draft

    def get_tab_post_writeback_draft(self, tab_id: str) -> object | None:
        self._log.add("writeback", "get_tab_post_writeback_draft", tab_id)
        return self.post_draft

    def get_item_draft(self, draft: object, session_id: str) -> object:
        self._log.add("writeback", "get_item_draft", draft, session_id)
        return {"draft": draft, "session_id": session_id}

    def edit_draft(
        self, draft: object, session_id: str, **changes: Any
    ) -> dict[str, object]:
        self._log.add("writeback", "edit_draft", draft, session_id, **changes)
        return {"valid": True}

    def apply_draft(self, draft: object) -> dict[str, Any]:
        self._log.add("writeback", "apply_draft", draft)
        return {"applied_ids": ["pane-item"], "written": {}}


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
        call("state", "is_tab_busy", "tab-1"),
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
        call("state", "is_tab_busy", "tab-1"),
        call("writeback", "apply_tab_writeback", "permit:tab-1"),
    ]


@pytest.mark.parametrize(
    ("pane", "draft_attr", "draft_getter"),
    [
        ("analysis", "analysis_draft", "get_tab_writeback_draft"),
        ("post_analysis", "post_draft", "get_tab_post_writeback_draft"),
    ],
)
def test_pane_writeback_reads_edits_and_applies_its_own_draft(
    pane: str, draft_attr: str, draft_getter: str
) -> None:
    facet, log, _state, writeback, _versions = _facet()
    draft = getattr(writeback, draft_attr)

    assert facet.get_writeback_item_draft_for_pane(
        "tab-1", cast(Any, pane), "md-1"
    ) == {"draft": draft, "session_id": "md-1"}
    assert facet.set_writeback_item_for_pane(
        "tab-1", cast(Any, pane), "md-1", selected=False
    ) == {"valid": True}
    assert facet.apply_writeback_for_pane("tab-1", cast(Any, pane)) == {
        "applied_ids": ["pane-item"],
        "written": {},
    }

    assert [entry for entry in log.calls if entry.method == draft_getter] == [
        call("writeback", draft_getter, "tab-1"),
        call("writeback", draft_getter, "tab-1"),
        call("writeback", draft_getter, "tab-1"),
    ]
    assert call("writeback", "get_item_draft", draft, "md-1") in log.calls
    assert call("writeback", "edit_draft", draft, "md-1", selected=False) in log.calls
    assert call("writeback", "apply_draft", draft) in log.calls


@pytest.mark.parametrize("pane", ["analysis", "post_analysis"])
def test_pane_writeback_requires_an_existing_draft(pane: str) -> None:
    facet, _log, _state, writeback, _versions = _facet()
    setattr(writeback, "analysis_draft" if pane == "analysis" else "post_draft", None)

    with pytest.raises(FailedPreconditionError, match="No .*writeback draft"):
        facet.apply_writeback_for_pane("tab-1", cast(Any, pane))


def test_pane_writeback_rejects_unknown_pane() -> None:
    facet, _log, _state, _writeback, _versions = _facet()

    with pytest.raises(InvalidInputError, match="unknown writeback pane"):
        facet.apply_writeback_for_pane("tab-1", cast(Any, "save"))


def test_pane_writeback_rejects_edit_while_tab_is_busy() -> None:
    facet, _log, state, _writeback, _versions = _facet()
    state.busy = True

    with pytest.raises(FailedPreconditionError, match="busy"):
        facet.set_writeback_item_for_pane("tab-1", "analysis", "md-1", selected=False)


def test_get_context_version_reads_resource_versions() -> None:
    facet, _log, _state, _writeback, versions = _facet()

    assert facet.get_context_version() == 7
    versions.clear()
    assert facet.get_context_version() == 0
