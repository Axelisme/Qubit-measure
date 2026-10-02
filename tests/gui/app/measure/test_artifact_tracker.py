from dataclasses import replace

import pytest
from matplotlib.figure import Figure
from zcu_tools.gui.app.measure.artifact_tracker import (
    ArtifactKey,
    ArtifactKind,
    ArtifactObservation,
    ArtifactTracker,
    SaveStatus,
)


def test_named_images_have_independent_history_and_retry() -> None:
    tracker = ArtifactTracker()
    result = object()
    first = ArtifactObservation(
        ArtifactKey(ArtifactKind.ANALYSIS, "fit"), result, Figure(), "fit.png"
    )
    second = ArtifactObservation(
        ArtifactKey(ArtifactKind.ANALYSIS, "residual"), result, Figure(), "residual.png"
    )
    initial = tracker.project((first, second))
    assert [item.key for item in initial] == [first.key, second.key]
    assert all(item.needs_save for item in initial)
    done = tracker.started(first.key)
    failed = tracker.started(second.key)
    done.succeed("saved-fit.png")
    failed.fail()
    assert not done.pending and not failed.pending
    snapshots = tracker.project((first, second))
    assert [item.needs_save for item in snapshots] == [False, True]
    assert snapshots[0].last_saved_path == "saved-fit.png"

    retry = tracker.started(second.key)
    retry.succeed("saved-residual.png")
    assert all(not item.needs_save for item in tracker.project((first, second)))
    explicit_export = tracker.started(first.key)
    explicit_export.fail()
    assert tracker.project((first, second))[0].status is SaveStatus.SAVED


def test_image_edits_and_destination_do_not_invalidate_success() -> None:
    tracker = ArtifactTracker()
    figure = Figure()
    observation = ArtifactObservation(
        ArtifactKey(ArtifactKind.POST_ANALYSIS, "fit"), object(), figure, "old.png"
    )
    tracker.project((observation,))
    tracker.started(observation.key).succeed("old.png")
    figure.subplots().plot([0, 1], [2, 3])
    edited = replace(observation, path="new.png", comment="new comment")
    snapshot = tracker.project((edited,))[0]
    assert snapshot.status is SaveStatus.SAVED
    assert snapshot.default_path == "new.png"
    assert snapshot.last_saved_path == "old.png"


@pytest.mark.parametrize("change", ["replace", "remove", "load"])
def test_late_completion_cannot_mark_new_or_reset_image_saved(change: str) -> None:
    tracker = ArtifactTracker()
    original = ArtifactObservation(
        ArtifactKey(ArtifactKind.ANALYSIS, "fit"), object(), Figure(), "fit.png"
    )
    tracker.project((original,))
    old_attempt = tracker.started(original.key)
    current = original
    if change == "replace":
        current = replace(original, figure=Figure())
    elif change == "remove":
        assert tracker.project(()) == ()
    else:
        tracker.reset_for_load()
    assert tracker.project((current,))[0].needs_save
    new_attempt = tracker.started(current.key)
    old_attempt.succeed("old-export.png")
    snapshot = tracker.project((current,))[0]
    assert snapshot.needs_save
    assert snapshot.last_saved_path is None
    assert new_attempt.pending
    new_attempt.succeed("new-export.png")
    assert tracker.project((current,))[0].last_saved_path == "new-export.png"


def test_data_preserves_submission_signature_and_previous_success() -> None:
    tracker = ArtifactTracker()
    original = ArtifactObservation(
        ArtifactKey(ArtifactKind.DATA), object(), None, "data.hdf5", "first"
    )
    tracker.project((original,))
    attempt = tracker.started(original.key)
    edited = replace(original, path="other.hdf5", comment="second")
    tracker.project((edited,))
    attempt.succeed("data.hdf5")
    snapshot = tracker.project((edited,))[0]
    assert snapshot.status is SaveStatus.UNSAVED_CHANGES
    assert snapshot.last_saved_path == "data.hdf5"
    assert tracker.project((original,))[0].status is SaveStatus.SAVED
    newer_result = replace(original, result=object())
    snapshot = tracker.project((newer_result,))[0]
    assert snapshot.status is SaveStatus.UNSAVED_CHANGES
    tracker.started(original.key).fail()
    assert tracker.project((newer_result,))[0].last_saved_path == "data.hdf5"


def test_invalid_projection_and_attempts_do_not_change_current_history() -> None:
    tracker = ArtifactTracker()
    observation = ArtifactObservation(
        ArtifactKey(ArtifactKind.ANALYSIS, "fit"), object(), Figure(), "fit.png"
    )
    tracker.project((observation,))
    attempt = tracker.started(observation.key)
    with pytest.raises(RuntimeError, match="already pending"):
        tracker.started(observation.key)
    with pytest.raises(ValueError, match="unique keys"):
        tracker.project((replace(observation, figure=Figure()), observation))
    assert attempt.pending
    attempt.succeed("fit.png")
    assert tracker.project((observation,))[0].status is SaveStatus.SAVED
    with pytest.raises(RuntimeError, match="already settled"):
        attempt.fail()
    with pytest.raises(ValueError, match="not saveable"):
        tracker.started(ArtifactKey(ArtifactKind.DATA))


def test_missing_result_is_not_saveable() -> None:
    tracker = ArtifactTracker()
    observation = ArtifactObservation(ArtifactKey(ArtifactKind.DATA), None, None, None)
    snapshot = tracker.project((observation,))[0]
    assert snapshot.status is SaveStatus.NO_RESULT
    assert not snapshot.needs_save
    with pytest.raises(ValueError, match="not saveable"):
        tracker.started(observation.key)


@pytest.mark.parametrize("name", [None, ""])
def test_image_keys_require_a_name(name: str | None) -> None:
    with pytest.raises(ValueError, match="nonempty"):
        ArtifactKey(ArtifactKind.ANALYSIS, name)


def test_data_keys_do_not_accept_a_figure_name() -> None:
    with pytest.raises(ValueError, match="cannot have"):
        ArtifactKey(ArtifactKind.DATA, "fit")
