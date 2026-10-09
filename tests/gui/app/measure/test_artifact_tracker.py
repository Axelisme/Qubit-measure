"""Section/member/output behavior through the State tracker public contract."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from zcu_tools.experiment.run_saving import ArtifactRef, OutputResult, SaveReport
from zcu_tools.gui.app.measure.artifact_tracker import (
    ArtifactDeclaration,
    ArtifactTracker,
    SaveStatus,
)
from zcu_tools.resources.entry import ArtifactKey, Output, SaveLayout

RUN = ArtifactRef("run", "measurement_data")
PRIMARY = ArtifactRef("analysis", "results")
POST = ArtifactRef("post_analysis", "results")
DATA = ArtifactKey(RUN.section, RUN.name, "data")
FIGURE = ArtifactKey(PRIMARY.section, PRIMARY.name, "figure", "fit")
JSON = ArtifactKey(PRIMARY.section, PRIMARY.name, "analysis", "analysis")
POST_JSON = ArtifactKey(POST.section, POST.name, "analysis", "post_analysis")


@pytest.fixture
def layout(tmp_path: Path) -> SaveLayout:
    return SaveLayout(
        result_path=tmp_path / "Result",
        database_path=tmp_path / "Database",
        run_id="captured-run",
        point="p",
        saved_at=datetime(2026, 10, 8, tzinfo=UTC),
    )


@pytest.fixture
def tracker(layout: SaveLayout) -> ArtifactTracker:
    tracker = ArtifactTracker()
    tracker.replace_section("run", (ArtifactDeclaration(RUN, (DATA,)),), layout=layout)
    tracker.replace_section(
        "analysis", (ArtifactDeclaration(PRIMARY, (FIGURE, JSON)),), layout=layout
    )
    tracker.replace_section(
        "post_analysis", (ArtifactDeclaration(POST, (POST_JSON,)),), layout=layout
    )
    return tracker


def test_sections_members_and_planned_outputs_follow_producer_order(
    tracker: ArtifactTracker, layout: SaveLayout
) -> None:
    sections = tracker.snapshot()
    assert tuple(section.section for section in sections) == (
        "run",
        "analysis",
        "post_analysis",
    )
    primary = sections[1].artifacts[0]
    assert primary.ref == PRIMARY
    assert tuple(member.key for member in primary.members) == (FIGURE, JSON)
    assert primary.status is SaveStatus.NOT_SAVED
    assert tuple(
        (output.format, output.path) for output in primary.members[0].outputs
    ) == tuple((output.format, output.path) for output in layout.outputs(FIGURE))
    assert all(
        output.status == "reserved" and output.error is None
        for member in primary.members
        for output in member.outputs
    )


def test_getter_does_not_reserve_next_labber_filename(tracker: ArtifactTracker) -> None:
    before = tracker.snapshot()
    labber = before[0].artifacts[0].members[0].outputs[1].path
    labber.parent.mkdir(parents=True)
    labber.touch()
    assert tracker.snapshot() == before


def test_latest_report_replaces_success_and_actual_paths(
    tracker: ArtifactTracker, layout: SaveLayout
) -> None:
    planned = layout.outputs(DATA)
    actual_labber = planned[1].path.with_name("captured-run_9.hdf5")
    tracker.apply_report(
        SaveReport(
            "captured-run",
            (
                OutputResult(DATA, planned[0], "saved", None),
                OutputResult(DATA, Output("labber", actual_labber), "saved", None),
            ),
        )
    )
    saved = tracker.snapshot()[0].artifacts[0]
    assert saved.status is SaveStatus.SAVED
    assert saved.members[0].outputs[1].path == actual_labber
    assert tuple(declaration.ref for declaration in tracker.select()) == (
        RUN,
        PRIMARY,
        POST,
    )
    tracker.apply_report(
        SaveReport(
            "captured-run",
            tuple(
                OutputResult(DATA, output, "failed", "disk full") for output in planned
            ),
        )
    )
    current = tracker.snapshot()[0].artifacts[0]
    assert current.status is SaveStatus.NOT_SAVED
    assert all(
        output.status == "failed" and output.error == "disk full"
        for output in current.members[0].outputs
    )
    assert saved.status is SaveStatus.SAVED


def test_partial_failure_and_dirty_preserve_paths_and_other_sections(
    tracker: ArtifactTracker, layout: SaveLayout
) -> None:
    before = tracker.snapshot()
    outputs = layout.outputs(DATA)
    tracker.apply_report(
        SaveReport(
            "captured-run",
            (
                OutputResult(DATA, outputs[0], "saved", None),
                OutputResult(DATA, outputs[1], "failed", "Labber failure"),
            ),
        )
    )
    partial = tracker.snapshot()[0].artifacts[0]
    assert partial.status is SaveStatus.PARTIALLY_SAVED
    tracker.mark_dirty(RUN)
    dirty = tracker.snapshot()[0].artifacts[0]
    assert dirty.status is SaveStatus.UNSAVED_CHANGES
    assert dirty.members == partial.members
    assert tracker.snapshot()[1:] == before[1:]
    tracker.mark_dirty(PRIMARY)
    assert tracker.snapshot()[1:] == before[1:]


def test_section_replacement_removes_omitted_members_and_resets_saved(
    tracker: ArtifactTracker, layout: SaveLayout
) -> None:
    tracker.apply_report(
        SaveReport(
            "captured-run",
            tuple(
                OutputResult(key, output, "saved", None)
                for key in (FIGURE, JSON)
                for output in layout.outputs(key)
            ),
        )
    )
    tracker.replace_section(
        "analysis", (ArtifactDeclaration(PRIMARY, (JSON,)),), layout=layout
    )
    primary = tracker.snapshot()[1].artifacts[0]
    assert primary.status is SaveStatus.NOT_SAVED
    assert tuple(member.key for member in primary.members) == (JSON,)
    tracker.replace_section("post_analysis", (), layout=layout)
    assert tuple(section.section for section in tracker.snapshot()) == (
        "run",
        "analysis",
    )
    tracker.clear()
    assert tracker.snapshot() == ()


@pytest.mark.parametrize("refs", [(), (RUN, RUN), (ArtifactRef("missing", "results"),)])
def test_invalid_selection_preserves_snapshot(
    tracker: ArtifactTracker, refs: tuple[ArtifactRef, ...]
) -> None:
    before = tracker.snapshot()
    with pytest.raises(ValueError, match="Save requires"):
        tracker.select(refs)
    assert tracker.snapshot() == before


def test_explicit_selection_retains_requested_order(tracker: ArtifactTracker) -> None:
    assert tuple(declaration.ref for declaration in tracker.select((POST, RUN))) == (
        POST,
        RUN,
    )


@pytest.mark.parametrize(
    "declarations",
    [
        (ArtifactDeclaration(RUN, (DATA,)),),
        (ArtifactDeclaration(PRIMARY, (JSON,)), ArtifactDeclaration(PRIMARY, (JSON,))),
    ],
)
def test_invalid_replacement_preserves_section(
    tracker: ArtifactTracker,
    layout: SaveLayout,
    declarations: tuple[ArtifactDeclaration, ...],
) -> None:
    before = tracker.snapshot()
    with pytest.raises(ValueError, match="section"):
        tracker.replace_section("analysis", declarations, layout=layout)
    assert tracker.snapshot() == before


def test_path_planning_failure_preserves_section(
    tracker: ArtifactTracker, layout: SaveLayout, tmp_path: Path
) -> None:
    before = tracker.snapshot()
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "Database").mkdir()
    (tmp_path / "Database" / "runs").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        tracker.replace_section(
            "analysis", (ArtifactDeclaration(PRIMARY, (JSON,)),), layout=layout
        )
    assert tracker.snapshot() == before


@pytest.mark.parametrize("members", [(), (DATA, DATA), (JSON,)])
def test_declaration_rejects_empty_duplicate_or_mismatched_members(
    members: tuple[ArtifactKey, ...],
) -> None:
    with pytest.raises(ValueError, match="artifact|member"):
        ArtifactDeclaration(RUN, members)
