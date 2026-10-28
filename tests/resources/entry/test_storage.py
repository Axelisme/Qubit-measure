"""Storage capture through bound point and ledger public contracts."""

from datetime import UTC, datetime
from pathlib import Path
from shutil import copytree
from uuid import uuid4

import pytest
from zcu_tools.resources.entry import (
    AcquiredPayload,
    ArtifactKey,
    LedgerEvent,
    PointView,
    ResultEntry,
    bind_run_outputs,
    capture_point_storage,
)

from tests._filesystem import symlink_or_skip


@pytest.mark.parametrize(
    "location", ["result-entry", "database-entry", "records", "run", "point"]
)
def test_recorded_output_binding_rejects_symlink_escape_without_writes(
    tmp_path: Path,
    location: str,
) -> None:
    result_root, database_root = tmp_path / "result", tmp_path / "Database"
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "retained.txt"
    sentinel.write_text("unchanged")
    paths = {
        "result-entry": result_root / "recorded",
        "database-entry": database_root / "recorded",
        "records": result_root / "recorded/records",
        "run": database_root / "recorded/runs",
        "point": result_root / "recorded/points/p",
    }
    source = paths[location]
    source.parent.mkdir(parents=True)
    symlink_or_skip(source, outside, target_is_directory=True)
    before = tuple(sorted(tmp_path.rglob("*")))
    if location in ("result-entry", "database-entry"):
        with pytest.raises(ValueError, match="escape"):
            bind_run_outputs(
                result_root=result_root,
                database_root=database_root,
                entry_name="recorded",
                point="p",
            )
    else:
        binding = bind_run_outputs(
            result_root=result_root,
            database_root=database_root,
            entry_name="recorded",
            point="p",
        )
        if location == "records":
            with pytest.raises(ValueError, match="escape"):
                binding.prepare_ledger()
        else:
            layout = binding.layout(stem="capture", saved_at=datetime.now(UTC))
            with pytest.raises(ValueError, match="escape"):
                layout.outputs(ArtifactKey("analysis", "fit", "figure", "plot"))
    assert sentinel.read_text() == "unchanged"
    assert tuple(sorted(tmp_path.rglob("*"))) == before


@pytest.mark.parametrize("field", ["entry_name", "point"])
def test_recorded_output_binding_rejects_unsafe_names_without_writes(
    tmp_path: Path, field: str
) -> None:
    names = {"entry_name": "recorded", "point": "p", field: "../outside"}
    with pytest.raises(ValueError, match=field):
        bind_run_outputs(
            result_root=tmp_path / "result",
            database_root=tmp_path / "Database",
            entry_name=names["entry_name"],
            point=names["point"],
        )
    assert tuple(tmp_path.iterdir()) == ()


def test_capture_retains_cached_context_and_exact_entry_attribution(
    tmp_path: Path,
) -> None:
    entry = ResultEntry.create(
        "entry",
        result_root=tmp_path / "results",
        database_root=tmp_path / "Database",
    )
    point = entry.new_point("sweet", description="cached")
    newer = entry.use_point("sweet")
    newer.description = "published later"
    before = tuple(sorted(tmp_path.rglob("*")))
    captured = capture_point_storage(point)

    assert captured.entry_name == "entry"
    assert captured.point == "sweet"
    assert captured.description == "cached"
    assert captured.roles == {}
    assert captured.result_path == entry.result_path
    assert captured.database_path == entry.database_path
    assert tuple(sorted(tmp_path.rglob("*"))) == before

    event = LedgerEvent(
        id=str(uuid4()),
        kind="acquired",
        at=datetime.now(UTC).isoformat(),
        origin="notebook",
        payload=AcquiredPayload(
            run_id="captured-run",
            experiment="test/saving",
            point=captured.point,
            roles={},
        ),
    )
    captured.ledger.append(event)
    assert entry.ledger.get(event.id).event == event


def _role_point(entry: ResultEntry) -> PointView:
    entry.setup.add_component("R1", kind="fake/sensor", rate=6500.0)
    entry.setup.add_component("R2", kind="fake/sensor", rate=7100.0)
    entry.setup.add_component("Q1", kind="fake/drive/a", rate=4300.0, sense="R1")
    entry.setup.add_component("Q2", kind="fake/drive/a", rate=5100.0, sense="R2")
    return entry.new_point("working")


def _published_state(root: Path) -> dict[Path, bytes | None]:
    return {
        path.relative_to(root): path.read_bytes() if path.is_file() else None
        for path in root.rglob("*")
    }


def test_capture_keeps_resolved_roles_after_input_and_point_changes(
    entry: ResultEntry, tmp_path: Path
) -> None:
    point = _role_point(entry)
    choices = {"driver": "Q2"}
    resolved = point.resolve(["driver", "sense"], **choices)
    assert entry.ledger.events() == ()
    before = _published_state(tmp_path)

    captured = capture_point_storage(point, roles=resolved)

    assert captured.roles == {"driver": "Q2", "sense": "R2"}
    assert _published_state(tmp_path) == before
    assert entry.ledger.events() == ()

    choices["driver"] = "Q1"
    point.Q2.sense = "R1"
    point.resolve(["driver", "sense"], focus="Q1")
    assert captured.roles == {"driver": "Q2", "sense": "R2"}
    assert capture_point_storage(point, roles=resolved).roles == captured.roles
    assert capture_point_storage(
        point, roles=point.resolve(["driver", "sense"], focus="Q2")
    ).roles == {"driver": "Q2", "sense": "R1"}


@pytest.mark.parametrize("empty", [False, True])
def test_capture_accepts_roles_from_another_handle_of_the_same_point(
    entry: ResultEntry, tmp_path: Path, empty: bool
) -> None:
    point = _role_point(entry)
    resolved = point.resolve([] if empty else ["driver", "sense"], focus="Q1")
    other = ResultEntry.open(
        "entry",
        result_root=entry.result_path.parent,
        database_root=entry.database_path.parent,
    ).use_point("working")
    other.description = "other handle"
    other.Q1.sense = "R2"
    assert entry.ledger.events() == ()
    before = _published_state(tmp_path)

    captured = capture_point_storage(other, roles=resolved)

    assert captured.description == "other handle"
    assert captured.roles == ({} if empty else {"driver": "Q1", "sense": "R1"})
    assert _published_state(tmp_path) == before
    assert entry.ledger.events() == ()


def test_role_capture_accepts_current_point_at_same_location_after_replacement(
    entry: ResultEntry, tmp_path: Path
) -> None:
    original = _role_point(entry)
    resolved = original.resolve(["driver", "sense"], focus="Q1")
    result_root = entry.result_path.parent
    database_root = entry.database_path.parent
    entry.result_path.rename(tmp_path / "retired")
    replacement = ResultEntry.create(
        "entry", result_root=result_root, database_root=database_root
    )
    current = _role_point(replacement)

    captured = capture_point_storage(current, roles=resolved)

    assert captured.entry_name == "entry"
    assert captured.roles == {"driver": "Q1", "sense": "R1"}


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("mismatch", ["point", "entry", "result-root", "database-root"])
def test_capture_rejects_roles_from_another_owner_without_saving(
    entry: ResultEntry, tmp_path: Path, empty: bool, mismatch: str
) -> None:
    point = _role_point(entry)
    resolved = point.resolve([] if empty else ["driver", "sense"], focus="Q1")
    if mismatch == "point":
        other = entry.new_point("other")
    elif mismatch == "entry":
        other_entry = ResultEntry.create(
            "other",
            result_root=entry.result_path.parent,
            database_root=entry.database_path.parent,
        )
        other = _role_point(other_entry)
    else:
        result_root = entry.result_path.parent
        database_root = entry.database_path.parent
        if mismatch == "result-root":
            result_root = tmp_path / "other-results"
            copytree(entry.result_path, result_root / "entry")
        else:
            database_root = tmp_path / "other-Database"
            (database_root / "entry").mkdir(parents=True)
        other_entry = ResultEntry.open(
            "entry", result_root=result_root, database_root=database_root
        )
        other = other_entry.use_point("working")
    assert entry.ledger.events() == ()
    before = _published_state(tmp_path)

    with pytest.raises(ValueError, match="same bound point"):
        capture_point_storage(other, roles=resolved)

    assert _published_state(tmp_path) == before
    assert entry.ledger.events() == ()
