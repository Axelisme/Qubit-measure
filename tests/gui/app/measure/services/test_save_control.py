"""SaveControlFacet public contract tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from zcu_tools.gui.app.measure.artifact_tracker import (
    ArtifactKey,
    ArtifactKind,
    ArtifactSnapshot,
    SaveStatus,
)
from zcu_tools.gui.app.measure.services.ports import (
    SaveArtifactsSubmission,
    SaveDataSubmission,
    SaveDestination,
)
from zcu_tools.gui.app.measure.services.save_control import SaveControlFacet
from zcu_tools.gui.expected_error import FailedPreconditionError

from tests.gui._control_fakes import CallLog, call


def _key(kind: ArtifactKind) -> ArtifactKey:
    return ArtifactKey(kind, "fit" if kind is not ArtifactKind.DATA else None)


class RecordingState:
    def __init__(self, log: CallLog) -> None:
        self._log = log
        self.comment = "existing draft"
        self.artifacts = tuple(
            ArtifactSnapshot(
                _key(kind), SaveStatus.NOT_SAVED, f"{kind.value}.out", None, True
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
    def __init__(self, log: CallLog, state: RecordingState) -> None:
        self._log = log
        self._state = state
        self.data_path: str | None = "default.h5"
        self.analysis_image_path: str | None = "analysis.out"
        self.post_analysis_image_path: str | None = "post_analysis.out"

    def get_tab_data_path(self, tab_id: str) -> str | None:
        self._log.add("tab", "get_tab_data_path", tab_id)
        return self.data_path

    def update_tab_data_path_override(self, tab_id: str, path: str) -> None:
        self._log.add("tab", "update_tab_data_path_override", tab_id, path)
        self.data_path = path

    def update_tab_image_path_override(
        self, tab_id: str, key: ArtifactKey, path: str
    ) -> None:
        self._log.add("tab", "update_tab_image_path_override", tab_id, key, path)
        if key.kind is ArtifactKind.ANALYSIS:
            self.analysis_image_path = path
        else:
            self.post_analysis_image_path = path
        self._state.artifacts = tuple(
            replace(item, default_path=path) if item.key == key else item
            for item in self._state.artifacts
        )


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

    def save_image_sync(
        self, permit: object, key: ArtifactKey, image_path: str
    ) -> None:
        self._log.add("save", "save_image_sync", permit, key, image_path)


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
        if a.key.kind is ArtifactKind.ANALYSIS
        else a
        for a in state.artifacts
    )
    submission = facet.save_artifacts("tab-1")
    assert {d.key for d in submission.destinations} == {
        a.key for a in state.artifacts if a.is_saveable
    }
    assert log.calls[-1].args[-1] == "existing draft"


def test_save_all_selects_unsaved_name_without_resaving_saved_sibling() -> None:
    facet, _log, state, _tab, _save, _bus, _notices = _facet()
    saved_fit = _key(ArtifactKind.ANALYSIS)
    diagnostic = ArtifactKey(ArtifactKind.ANALYSIS, "diagnostic")
    state.artifacts = tuple(
        replace(item, status=SaveStatus.SAVED) if item.key == saved_fit else item
        for item in state.artifacts
    ) + (
        ArtifactSnapshot(
            diagnostic, SaveStatus.NOT_SAVED, "diagnostic.png", None, True
        ),
    )

    submission = facet.save_artifacts("tab-1")

    assert tuple(destination.key for destination in submission.destinations) == (
        _key(ArtifactKind.DATA),
        _key(ArtifactKind.POST_ANALYSIS),
        diagnostic,
    )
    assert submission.destinations[-1].path == "diagnostic.png"


def test_save_all_retry_skips_successes_and_stops_when_everything_is_saved() -> None:
    facet, log, state, _tab, _save, _bus, _notices = _facet()
    state.artifacts = tuple(
        replace(a, status=SaveStatus.SAVED)
        if a.key.kind is not ArtifactKind.POST_ANALYSIS
        else a
        for a in state.artifacts
    )
    submission = facet.save_artifacts("tab-1")
    assert tuple(d.key for d in submission.destinations) == (
        _key(ArtifactKind.POST_ANALYSIS),
    )
    state.artifacts = tuple(
        replace(a, status=SaveStatus.SAVED) for a in state.artifacts
    )
    with pytest.raises(FailedPreconditionError, match="nonempty unique"):
        facet.save_artifacts("tab-1")
    assert sum(entry.method == "start_save_artifacts" for entry in log.calls) == 1


def test_explicit_subset_can_export_a_saved_image() -> None:
    facet, _log, state, _tab, _save, _bus, _notices = _facet()
    state.artifacts = tuple(
        replace(a, status=SaveStatus.SAVED) for a in state.artifacts
    )
    submission = facet.save_artifacts("tab-1", artifacts=(_key(ArtifactKind.ANALYSIS),))
    assert tuple(d.key for d in submission.destinations) == (
        _key(ArtifactKind.ANALYSIS),
    )


@pytest.mark.parametrize("status", [SaveStatus.NOT_SAVED, SaveStatus.UNSAVED_CHANGES])
def test_save_all_includes_unsaved_or_changed_data(status: SaveStatus) -> None:
    facet, _log, state, _tab, _save, _bus, _notices = _facet()
    state.artifacts = tuple(
        replace(
            a, status=status if a.key.kind is ArtifactKind.DATA else SaveStatus.SAVED
        )
        for a in state.artifacts
    )
    submission = facet.save_artifacts("tab-1")
    assert tuple(d.key for d in submission.destinations) == (_key(ArtifactKind.DATA),)


@pytest.mark.parametrize(
    "path,comment,changed",
    [
        ("changed.h5", None, True),
        (None, "new comment", True),
        (None, "", True),
        ("data.out", None, False),
        (None, "existing draft", False),
    ],
)
def test_save_all_previews_data_drafts_before_selecting_saved_data(
    path: str | None, comment: str | None, changed: bool
) -> None:
    facet, log, state, tab, _save, bus, _notices = _facet()
    state.artifacts = tuple(
        replace(a, status=SaveStatus.SAVED) for a in state.artifacts
    )
    paths = {_key(ArtifactKind.DATA): path} if path is not None else None
    if changed:
        submission = facet.save_artifacts("tab-1", paths=paths, comment=comment)
        assert tuple(d.key for d in submission.destinations) == (
            _key(ArtifactKind.DATA),
        )
        assert state.comment == (comment if comment is not None else "existing draft")
    else:
        with pytest.raises(FailedPreconditionError, match="nonempty unique"):
            facet.save_artifacts("tab-1", paths=paths, comment=comment)
        assert state.comment == "existing draft"
        assert tab.data_path == "default.h5"
        assert not bus.payloads
        assert not any(entry.target == "save" for entry in log.calls)


def test_saved_image_path_override_requires_explicit_selection() -> None:
    facet, log, state, tab, _save, bus, _notices = _facet()
    state.artifacts = tuple(
        replace(a, status=SaveStatus.SAVED)
        if a.key.kind is not ArtifactKind.DATA
        else a
        for a in state.artifacts
    )
    with pytest.raises(FailedPreconditionError, match="must name selected artifacts"):
        facet.save_artifacts(
            "tab-1", paths={_key(ArtifactKind.ANALYSIS): "new.png"}, comment="new"
        )
    assert tab.analysis_image_path == "analysis.out"
    assert state.comment == "existing draft"
    assert not bus.payloads
    assert not any(entry.target == "save" for entry in log.calls)


def test_explicit_save_subset_commits_paths_and_comment_before_submission() -> None:
    facet, log, state, tab, _save, bus, _notices = _facet()
    submission = facet.save_artifacts(
        "tab-1",
        artifacts=(_key(ArtifactKind.ANALYSIS),),
        paths={_key(ArtifactKind.ANALYSIS): "chosen.png"},
        comment="shared draft",
    )
    assert submission.destinations == (
        SaveDestination(_key(ArtifactKind.ANALYSIS), "chosen.png"),
    )
    assert tab.analysis_image_path == "chosen.png"
    assert state.comment == "shared draft"
    assert bus.payloads
    assert log.calls[-1].method == "start_save_artifacts"


@pytest.mark.parametrize(
    "artifacts,paths",
    [
        ((), {}),
        ((_key(ArtifactKind.DATA), _key(ArtifactKind.DATA)), {}),
        ((_key(ArtifactKind.DATA),), {_key(ArtifactKind.DATA): "  "}),
        ((_key(ArtifactKind.DATA),), {_key(ArtifactKind.ANALYSIS): "other.png"}),
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


@pytest.mark.parametrize(
    "data_collision,extensionless", [(False, False), (True, False), (False, True)]
)
def test_colliding_paths_do_not_change_shared_drafts(
    tmp_path: Path, data_collision: bool, extensionless: bool
) -> None:
    facet, log, state, tab, _save, bus, _notices = _facet()
    other_kind = _key(
        ArtifactKind.DATA if data_collision else ArtifactKind.POST_ANALYSIS
    )
    paths = {
        _key(ArtifactKind.ANALYSIS): str(
            tmp_path / ("shared_1.hdf5" if data_collision else "shared.png")
        ),
        other_kind: str(
            tmp_path / ("shared.hdf5" if data_collision else "sub/../shared.png")
        ),
    }
    if extensionless:
        paths[_key(ArtifactKind.ANALYSIS)] = str(tmp_path / "shared")
    with pytest.raises(FailedPreconditionError, match="distinct"):
        facet.save_artifacts(
            "tab-1", artifacts=tuple(paths), paths=paths, comment="not committed"
        )
    assert state.comment == "existing draft"
    assert tab.data_path == "default.h5"
    assert tab.analysis_image_path == "analysis.out"
    assert tab.post_analysis_image_path == "post_analysis.out"
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
    tab = RecordingTab(log, state)
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
    key = _key(ArtifactKind.ANALYSIS)

    assert facet.save_image("tab-1", key) == "analysis.out"

    assert log.calls == [
        call("guard", "acquire_save_permit", "tab-1"),
        call("state", "is_tab_busy", "tab-1"),
        call("save", "save_image_sync", "permit:tab-1", key, "analysis.out"),
    ]
    assert notifications == ["Image saved to analysis.out"]


def test_save_post_image_uses_default_path_and_notifies() -> None:
    facet, log, _state, _tab, _save, _bus, notifications = _facet()
    key = _key(ArtifactKind.POST_ANALYSIS)

    assert facet.save_image("tab-1", key) == "post_analysis.out"

    assert log.calls == [
        call("guard", "acquire_save_permit", "tab-1"),
        call("state", "is_tab_busy", "tab-1"),
        call("save", "save_image_sync", "permit:tab-1", key, "post_analysis.out"),
    ]
    assert notifications == ["Image saved to post_analysis.out"]


@pytest.mark.parametrize("post", [False, True])
def test_image_save_explicit_path_becomes_shared_draft_and_omission_reuses_it(
    post: bool,
) -> None:
    facet, log, _state, tab, _save, _bus, _notifications = _facet()
    key = _key(ArtifactKind.POST_ANALYSIS if post else ArtifactKind.ANALYSIS)
    assert facet.save_image("tab-1", key, "chosen.png") == "chosen.png"
    assert (
        tab.post_analysis_image_path if post else tab.analysis_image_path
    ) == "chosen.png"
    assert facet.save_image("tab-1", key) == "chosen.png"
    calls = [entry for entry in log.calls if entry.target == "save"]
    assert (
        calls
        == [call("save", "save_image_sync", "permit:tab-1", key, "chosen.png")] * 2
    )


@pytest.mark.parametrize("post", [False, True])
@pytest.mark.parametrize("path", ["", "  "])
def test_image_save_rejects_empty_path_without_mutating_draft(
    post: bool, path: str
) -> None:
    facet, log, _state, tab, _save, _bus, _notifications = _facet()
    key = _key(ArtifactKind.POST_ANALYSIS if post else ArtifactKind.ANALYSIS)
    with pytest.raises(FailedPreconditionError, match="empty image path"):
        facet.save_image("tab-1", key, path)
    assert tab.analysis_image_path == "analysis.out"
    assert tab.post_analysis_image_path == "post_analysis.out"
    assert log.calls == []


@pytest.mark.parametrize("post", [False, True])
def test_single_image_invalid_name_with_override_does_not_commit_draft(
    post: bool, tmp_path: Path
) -> None:
    facet, log, state, tab, _save, bus, notifications = _facet()
    kind = ArtifactKind.POST_ANALYSIS if post else ArtifactKind.ANALYSIS
    key = ArtifactKey(kind, "\ud800")
    state.artifacts = tuple(
        replace(artifact, key=key, default_path=None)
        if artifact.key.kind is kind
        else artifact
        for artifact in state.artifacts
    )
    original_artifacts = state.artifacts
    path = tmp_path / "chosen.png"

    with pytest.raises(FailedPreconditionError, match="UTF-8"):
        facet.save_image("tab-1", key, str(path))

    assert state.artifacts == original_artifacts
    assert tab.analysis_image_path == "analysis.out"
    assert tab.post_analysis_image_path == "post_analysis.out"
    assert bus.payloads == []
    assert notifications == []
    assert not path.exists()
    assert all(
        entry.method not in {"save_image_sync", "update_tab_image_path_override"}
        for entry in log.calls
    )


@pytest.mark.parametrize("post", [False, True])
def test_image_save_failure_keeps_explicit_draft_without_success_notification(
    post: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    facet, _log, _state, tab, saver, _bus, notifications = _facet()

    def fail(permit: object, key: ArtifactKey, path: str) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(saver, "save_image_sync", fail)
    key = _key(ArtifactKind.POST_ANALYSIS if post else ArtifactKind.ANALYSIS)
    with pytest.raises(OSError, match="disk full"):
        facet.save_image("tab-1", key, "chosen.png")
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
    key = _key(ArtifactKind.POST_ANALYSIS if post else ArtifactKind.ANALYSIS)
    with pytest.raises(FailedPreconditionError, match="busy"):
        facet.save_image("tab-1", key, "chosen.png")
    assert tab.analysis_image_path == "analysis.out"
    assert tab.post_analysis_image_path == "post_analysis.out"
    assert not any(entry.target == "save" for entry in log.calls)


def test_missing_save_paths_fast_fails() -> None:
    facet, log, state, _tab, _save, _bus, notifications = _facet()
    key = _key(ArtifactKind.ANALYSIS)
    state.artifacts = tuple(
        replace(item, default_path=None) if item.key == key else item
        for item in state.artifacts
    )

    with pytest.raises(FailedPreconditionError, match="no image path configured"):
        facet.save_image("tab-1", key)

    assert log.calls == [
        call("guard", "acquire_save_permit", "tab-1"),
        call("state", "is_tab_busy", "tab-1"),
    ]
    assert notifications == []
