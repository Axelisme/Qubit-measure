"""SaveControlFacet public contract tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from zcu_tools.gui.app.main.artifact_tracker import (
    ArtifactKind,
    ArtifactSnapshot,
    SaveStatus,
)
from zcu_tools.gui.app.main.services.ports import (
    SaveArtifactsSubmission,
    SaveDataSubmission,
    SaveDestination,
)
from zcu_tools.gui.app.main.services.save_control import SaveControlFacet
from zcu_tools.gui.expected_error import FailedPreconditionError

from tests.gui._control_fakes import CallLog, call


class RecordingState:
    def __init__(self, log: CallLog) -> None:
        self._log = log
        self.comment = "existing draft"
        self.artifacts = tuple(
            ArtifactSnapshot(
                kind, SaveStatus.NOT_SAVED, f"{kind.value}.out", None, True
            )
            for kind in ArtifactKind
        )

    def get_artifact_snapshots(self, tab_id: str) -> tuple[ArtifactSnapshot, ...]:
        return self.artifacts

    def get_tab(self, tab_id: str) -> SimpleNamespace:
        self._log.add("state", "get_tab", tab_id)
        return SimpleNamespace(save=SimpleNamespace(comment=self.comment))

    def update_tab_comment(self, tab_id: str, comment: str) -> None:
        self._log.add("state", "update_tab_comment", tab_id, comment)
        self.comment = comment

    def has_tab(self, tab_id: str) -> bool:
        self._log.add("state", "has_tab", tab_id)
        return tab_id == "tab-1"

    def is_tab_busy(self, tab_id: str) -> bool:
        self._log.add("state", "is_tab_busy", tab_id)
        return False


class RecordingGuard:
    def __init__(self, log: CallLog) -> None:
        self._log = log

    def acquire_save_permit(self, tab_id: str) -> str:
        self._log.add("guard", "acquire_save_permit", tab_id)
        return f"permit:{tab_id}"


class RecordingTab:
    def __init__(self, log: CallLog) -> None:
        self._log = log
        self.data_path: str | None = "default.h5"
        self.analysis_image_path: str | None = "default.png"
        self.post_analysis_image_path: str | None = "default.png"

    def get_tab_data_path(self, tab_id: str) -> str | None:
        self._log.add("tab", "get_tab_data_path", tab_id)
        return self.data_path

    def update_tab_data_path_override(self, tab_id: str, path: str) -> None:
        self._log.add("tab", "update_tab_data_path_override", tab_id, path)
        self.data_path = path

    def get_tab_analysis_image_path(self, tab_id: str) -> str | None:
        self._log.add("tab", "get_tab_analysis_image_path", tab_id)
        return self.analysis_image_path

    def update_tab_analysis_image_path_override(self, tab_id: str, path: str) -> None:
        self._log.add("tab", "update_tab_analysis_image_path_override", tab_id, path)
        self.analysis_image_path = path

    def update_tab_post_analysis_image_path_override(
        self, tab_id: str, path: str
    ) -> None:
        self._log.add(
            "tab", "update_tab_post_analysis_image_path_override", tab_id, path
        )
        self.post_analysis_image_path = path

    def get_tab_post_analysis_image_path(self, tab_id: str) -> str | None:
        self._log.add("tab", "get_tab_post_analysis_image_path", tab_id)
        return self.post_analysis_image_path


class RecordingSave:
    def __init__(self, log: CallLog) -> None:
        self._log = log

    def start_save_data(
        self, permit: object, data_path: str, comment: str = ""
    ) -> SaveDataSubmission:
        self._log.add("save", "start_save_data", permit, data_path, comment=comment)
        return SaveDataSubmission(7, f"written:{data_path}")

    def start_save_artifacts(
        self, permit: object, destinations: tuple[SaveDestination, ...], comment: str
    ) -> SaveArtifactsSubmission:
        self._log.add("save", "start_save_artifacts", permit, destinations, comment)
        return SaveArtifactsSubmission(8, destinations)

    def save_image_sync(self, permit: object, image_path: str) -> None:
        self._log.add("save", "save_image_sync", permit, image_path)

    def save_post_image_sync(self, permit: object, image_path: str) -> None:
        self._log.add("save", "save_post_image_sync", permit, image_path)


class RecordingBus:
    def __init__(self, log: CallLog) -> None:
        self._log = log
        self.payloads: list[object] = []

    def emit(self, payload: object) -> None:
        self._log.add("bus", "emit", type(payload).__name__)
        self.payloads.append(payload)


@pytest.mark.parametrize("analysis_saveable", [False, True])
def test_save_all_selects_only_saveable_artifacts(analysis_saveable: bool) -> None:
    facet, log, state, _tab, _save, _bus, _notices = _facet()
    state.artifacts = tuple(
        replace(a, is_saveable=analysis_saveable)
        if a.kind is ArtifactKind.ANALYSIS
        else a
        for a in state.artifacts
    )
    submission = facet.save_artifacts("tab-1")
    assert {d.kind for d in submission.destinations} == {
        a.kind for a in state.artifacts if a.is_saveable
    }
    assert log.calls[-1].args[-1] == "existing draft"


def test_explicit_save_subset_commits_paths_and_comment_before_submission() -> None:
    facet, log, state, tab, _save, bus, _notices = _facet()
    submission = facet.save_artifacts(
        "tab-1",
        artifacts=(ArtifactKind.ANALYSIS,),
        paths={ArtifactKind.ANALYSIS: "chosen.png"},
        comment="shared draft",
    )
    assert submission.destinations == (
        SaveDestination(ArtifactKind.ANALYSIS, "chosen.png"),
    )
    assert tab.analysis_image_path == "chosen.png"
    assert state.comment == "shared draft"
    assert bus.payloads
    assert log.calls[-1].method == "start_save_artifacts"


@pytest.mark.parametrize(
    "artifacts,paths",
    [
        ((), {}),
        ((ArtifactKind.DATA, ArtifactKind.DATA), {}),
        ((ArtifactKind.DATA,), {ArtifactKind.DATA: "  "}),
        ((ArtifactKind.DATA,), {ArtifactKind.ANALYSIS: "other.png"}),
    ],
)
def test_save_artifact_selection_errors_do_not_mutate_drafts(artifacts, paths) -> None:
    facet, log, state, tab, _save, bus, _notices = _facet()
    with pytest.raises(FailedPreconditionError):
        facet.save_artifacts("tab-1", artifacts=artifacts, paths=paths, comment="new")
    assert state.comment == "existing draft"
    assert tab.data_path == "default.h5"
    assert not bus.payloads
    assert not any(entry.target == "save" for entry in log.calls)


@pytest.mark.parametrize("data_collision", [False, True])
def test_colliding_paths_do_not_change_shared_drafts(
    tmp_path: Path, data_collision: bool
) -> None:
    facet, log, state, tab, _save, bus, _notices = _facet()
    other_kind = ArtifactKind.DATA if data_collision else ArtifactKind.POST_ANALYSIS
    paths = {
        ArtifactKind.ANALYSIS: str(
            tmp_path / ("shared_1.hdf5" if data_collision else "shared.png")
        ),
        other_kind: str(
            tmp_path / ("shared.hdf5" if data_collision else "sub/../shared.png")
        ),
    }
    with pytest.raises(FailedPreconditionError, match="distinct"):
        facet.save_artifacts(
            "tab-1", artifacts=tuple(paths), paths=paths, comment="not committed"
        )
    assert state.comment == "existing draft"
    assert tab.data_path == "default.h5"
    assert tab.analysis_image_path == "default.png"
    assert tab.post_analysis_image_path == "default.png"
    assert not bus.payloads
    assert not any(entry.target == "save" for entry in log.calls)
    assert not list(tmp_path.iterdir())


def _facet() -> tuple[
    SaveControlFacet,
    CallLog,
    RecordingState,
    RecordingTab,
    RecordingSave,
    RecordingBus,
    list[str],
]:
    log = CallLog()
    state = RecordingState(log)
    tab = RecordingTab(log)
    save = RecordingSave(log)
    bus = RecordingBus(log)
    notifications: list[str] = []
    return (
        SaveControlFacet(
            state=cast(Any, state),
            bus=cast(Any, bus),
            guard=cast(Any, RecordingGuard(log)),
            tab=cast(Any, tab),
            save=cast(Any, save),
            notify_info=notifications.append,
        ),
        log,
        state,
        tab,
        save,
        bus,
        notifications,
    )


def test_has_tab_reads_state() -> None:
    facet, log, _state, _tab, _save, _bus, _notifications = _facet()

    assert facet.has_tab("tab-1") is True

    assert log.calls == [call("state", "has_tab", "tab-1")]


def test_save_data_applies_explicit_path_and_comment_to_shared_draft() -> None:
    facet, log, state, tab, _save, _bus, _notifications = _facet()

    assert facet.save_data(
        "tab-1", "explicit.h5", comment="note"
    ) == SaveDataSubmission(7, "written:explicit.h5")
    assert tab.data_path == "explicit.h5"
    assert state.comment == "note"
    assert (
        call("tab", "update_tab_data_path_override", "tab-1", "explicit.h5")
        in log.calls
    )
    assert call("state", "update_tab_comment", "tab-1", "note") in log.calls
    assert log.calls[-1] == call(
        "save", "start_save_data", "permit:tab-1", "explicit.h5", comment="note"
    )


def test_save_data_rejects_explicit_empty_path_before_reserving_save() -> None:
    facet, log, _state, _tab, _save, _bus, _notifications = _facet()

    with pytest.raises(FailedPreconditionError, match="empty data path"):
        facet.save_data("tab-1", data_path="")
    assert not any(entry.method == "start_save_data" for entry in log.calls)


def test_save_data_omissions_inherit_draft_but_explicit_empty_comment_clears_it() -> (
    None
):
    facet, log, state, tab, _save, _bus, _notifications = _facet()

    assert facet.save_data("tab-1") == SaveDataSubmission(7, "written:default.h5")
    assert log.calls[-1] == call(
        "save",
        "start_save_data",
        "permit:tab-1",
        "default.h5",
        comment="existing draft",
    )
    assert state.comment == "existing draft"
    assert tab.data_path == "default.h5"
    assert not any(entry.method == "update_tab_comment" for entry in log.calls)

    assert facet.save_data("tab-1", comment="") == SaveDataSubmission(
        7, "written:default.h5"
    )
    assert state.comment == ""
    assert log.calls[-1] == call(
        "save", "start_save_data", "permit:tab-1", "default.h5", comment=""
    )


def test_save_image_uses_default_path_and_notifies() -> None:
    facet, log, _state, _tab, _save, _bus, notifications = _facet()

    assert facet.save_image("tab-1") == "default.png"

    assert log.calls == [
        call("guard", "acquire_save_permit", "tab-1"),
        call("state", "is_tab_busy", "tab-1"),
        call("tab", "get_tab_analysis_image_path", "tab-1"),
        call("save", "save_image_sync", "permit:tab-1", "default.png"),
    ]
    assert notifications == ["Image saved to default.png"]


def test_save_post_image_uses_default_path_and_notifies() -> None:
    facet, log, _state, _tab, _save, _bus, notifications = _facet()

    assert facet.save_post_image("tab-1") == "default.png"

    assert log.calls == [
        call("guard", "acquire_save_permit", "tab-1"),
        call("state", "is_tab_busy", "tab-1"),
        call("tab", "get_tab_post_analysis_image_path", "tab-1"),
        call("save", "save_post_image_sync", "permit:tab-1", "default.png"),
    ]
    assert notifications == ["Post-analysis image saved to default.png"]


@pytest.mark.parametrize("post", [False, True])
def test_image_save_explicit_path_becomes_shared_draft_and_omission_reuses_it(
    post: bool,
) -> None:
    facet, log, _state, tab, _save, _bus, _notifications = _facet()
    save = facet.save_post_image if post else facet.save_image
    assert save("tab-1", "chosen.png") == "chosen.png"
    assert (
        tab.post_analysis_image_path if post else tab.analysis_image_path
    ) == "chosen.png"
    assert save("tab-1") == "chosen.png"
    calls = [entry for entry in log.calls if entry.target == "save"]
    assert (
        calls
        == [
            call(
                "save",
                "save_post_image_sync" if post else "save_image_sync",
                "permit:tab-1",
                "chosen.png",
            )
        ]
        * 2
    )


@pytest.mark.parametrize("post", [False, True])
@pytest.mark.parametrize("path", ["", "  "])
def test_image_save_rejects_empty_path_without_mutating_draft(
    post: bool, path: str
) -> None:
    facet, log, _state, tab, _save, _bus, _notifications = _facet()
    save = facet.save_post_image if post else facet.save_image
    with pytest.raises(FailedPreconditionError, match="empty .*image path"):
        save("tab-1", path)
    assert tab.analysis_image_path == "default.png"
    assert tab.post_analysis_image_path == "default.png"
    assert log.calls == []


@pytest.mark.parametrize("post", [False, True])
def test_image_save_failure_keeps_explicit_draft_without_success_notification(
    post: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    facet, _log, _state, tab, saver, _bus, notifications = _facet()

    def fail(permit: object, path: str) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(
        saver, "save_post_image_sync" if post else "save_image_sync", fail
    )
    save = facet.save_post_image if post else facet.save_image
    with pytest.raises(OSError, match="disk full"):
        save("tab-1", "chosen.png")
    assert (
        tab.post_analysis_image_path if post else tab.analysis_image_path
    ) == "chosen.png"
    assert notifications == []


@pytest.mark.parametrize("post", [False, True])
def test_busy_image_save_does_not_change_shared_draft(
    post: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    facet, log, state, tab, _save, _bus, _notifications = _facet()
    monkeypatch.setattr(state, "is_tab_busy", lambda tab_id: True)
    save = facet.save_post_image if post else facet.save_image
    with pytest.raises(FailedPreconditionError, match="busy"):
        save("tab-1", "chosen.png")
    assert tab.analysis_image_path == "default.png"
    assert tab.post_analysis_image_path == "default.png"
    assert not any(entry.target == "save" for entry in log.calls)


def test_missing_save_paths_fast_fails() -> None:
    facet, log, _state, tab, _save, _bus, notifications = _facet()
    tab.analysis_image_path = None

    with pytest.raises(
        FailedPreconditionError, match="no analysis image path configured"
    ):
        facet.save_image("tab-1")

    assert log.calls == [
        call("guard", "acquire_save_permit", "tab-1"),
        call("state", "is_tab_busy", "tab-1"),
        call("tab", "get_tab_analysis_image_path", "tab-1"),
    ]
    assert notifications == []
