"""WritebackControlFacet public contract tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.app.main.services.writeback_control import WritebackControlFacet
from zcu_tools.gui.expected_error import FailedPreconditionError, InvalidInputError

from tests.gui._control_fakes import CallLog, call


class RecordingState:
    def __init__(self, log: CallLog) -> None:
        self._log = log
        self.busy = False
        self.tab = SimpleNamespace(
            analysis=SimpleNamespace(writeback_draft=None),
            post_analysis=SimpleNamespace(writeback_draft=None),
        )

    def has_tab(self, tab_id: str) -> bool:
        self._log.add("state", "has_tab", tab_id)
        return tab_id == "tab-1"

    def get_tab(self, tab_id: str) -> object:
        return self.tab

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
        self.analysis_draft: MagicMock = MagicMock(is_active=True)
        self.post_draft: MagicMock = MagicMock(is_active=True)

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
    state.tab.analysis.writeback_draft = writeback.analysis_draft
    state.tab.post_analysis.writeback_draft = writeback.post_draft
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


@pytest.mark.parametrize(
    ("pane", "draft_attr"),
    [("analysis", "analysis_draft"), ("post_analysis", "post_draft")],
)
def test_pane_writeback_reads_edits_and_applies_its_own_draft(
    pane: str, draft_attr: str
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

    assert call("writeback", "get_item_draft", draft, "md-1") in log.calls
    assert call("writeback", "edit_draft", draft, "md-1", selected=False) in log.calls
    assert call("writeback", "apply_draft", draft) in log.calls


@pytest.mark.parametrize("pane", ["analysis", "post_analysis"])
def test_pane_writeback_requires_an_existing_draft(pane: str) -> None:
    facet, _log, state, writeback, _versions = _facet()
    draft_attr = "analysis_draft" if pane == "analysis" else "post_draft"
    setattr(writeback, draft_attr, None)
    setattr(
        state.tab.analysis if pane == "analysis" else state.tab.post_analysis,
        "writeback_draft",
        None,
    )

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
