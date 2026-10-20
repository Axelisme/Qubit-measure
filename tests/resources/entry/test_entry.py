"""ResultEntry creation, identity and Result-only rename through public APIs."""

from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

import pytest
from pydantic import ValidationError
from ruamel.yaml import YAML
from zcu_tools.resources import entry as entry_api
from zcu_tools.resources.entry import (
    ResultEntry,
    rename_entry,
)

from tests._filesystem import symlink_or_skip


def test_create_accepts_a_fixed_identity_without_changing_default_creation(
    entry_roots: tuple[Path, Path],
) -> None:
    assert hasattr(entry_api, "EntryIdentity"), (
        "owner must accept stable creation values"
    )
    result, database = entry_roots
    identity = entry_api.EntryIdentity(
        entry_id=UUID("00000000-0000-0000-0000-000000000042"),
        created_at=datetime(2026, 10, 20, 1, 2, 3, tzinfo=timezone.utc),
    )
    fixed = ResultEntry.create(
        "fixed", result_root=result, database_root=database, identity=identity
    )
    reopened = ResultEntry.open("fixed", result_root=result, database_root=database)
    assert fixed.entry_id == reopened.entry_id == str(identity.entry_id)
    assert (
        entry_api.snapshot(reopened.setup).general.created_at == "2026-10-20T01:02:03Z"
    )
    fresh = ResultEntry.create("fresh", result_root=result, database_root=database)
    assert fresh.entry_id != fixed.entry_id
    assert (
        datetime.fromisoformat(
            entry_api.snapshot(fresh.setup).general.created_at
        ).tzinfo
        is not None
    )


def test_fixed_identity_rejects_naive_creation_time_before_creating_directories(
    entry_roots: tuple[Path, Path],
) -> None:
    assert hasattr(entry_api, "EntryIdentity"), "fixed identity must validate UTC time"
    with pytest.raises(ValueError, match="UTC"):
        ResultEntry.create(
            "fixed",
            result_root=entry_roots[0],
            database_root=entry_roots[1],
            identity=entry_api.EntryIdentity(
                entry_id=UUID("00000000-0000-0000-0000-000000000042"),
                created_at=datetime(2026, 10, 20),
            ),
        )
    assert all(not root.exists() for root in entry_roots)


def read_entry_files(path: Path) -> dict[Path, bytes]:
    return {
        item.relative_to(path): item.read_bytes()
        for item in path.rglob("*")
        if item.is_file()
    }


def test_open_rejects_entry_links_that_escape_the_configured_root(
    entry_roots: tuple[Path, Path],
) -> None:
    results, database = entry_roots
    linked_path = results / "entry"
    linked_path.parent.mkdir()
    symlink_or_skip(linked_path, results.parent / "external", target_is_directory=True)

    with pytest.raises(ValueError, match="entry escapes its root"):
        ResultEntry.open("entry", result_root=results, database_root=database)


def test_setup_reads_use_memory_even_when_the_file_is_unavailable(
    entry_roots: tuple[Path, Path], entry: ResultEntry
) -> None:
    results, _database = entry_roots
    entry.setup.description = "cached annotation"
    original_id = entry.entry_id
    setup_path = results / "entry" / "setup.yaml"
    setup_path.unlink()

    assert entry.setup.description == "cached annotation"
    assert entry.entry_id == original_id
    with pytest.raises(FileNotFoundError) as failure:
        entry.setup.refresh()
    assert failure.value.filename == str(setup_path)
    assert entry.setup.description == "cached annotation"
    assert entry.entry_id == original_id


def test_setup_edit_discards_body_failure_and_allows_the_next_transaction(
    entry_roots: tuple[Path, Path], entry: ResultEntry
) -> None:
    results, _database = entry_roots
    entry.setup.description = "committed annotation"
    setup_path = results / "entry" / "setup.yaml"
    before = setup_path.read_bytes()

    def abort_transaction() -> None:
        with entry.setup.edit() as draft:
            draft.description = "discarded annotation"
            assert entry.setup.description == "committed annotation"
            raise RuntimeError("abort sentinel")

    with pytest.raises(RuntimeError, match="abort sentinel"):
        abort_transaction()
    assert entry.setup.description == "committed annotation"
    assert setup_path.read_bytes() == before

    entry.setup.description = "next annotation"
    assert entry.setup.description == "next annotation"


def test_entry_identity_is_read_only_and_refresh_cannot_publish_a_replacement(
    entry_roots: tuple[Path, Path], entry: ResultEntry
) -> None:
    results, _database = entry_roots
    original_id = entry.entry_id
    entry.setup.description = "original annotation"
    replacement_id = "00000000-0000-0000-0000-000000000001"
    with pytest.raises(AttributeError, match="entry_id"):
        setattr(entry, "entry_id", replacement_id)  # noqa: B010 -- Exercise the readonly descriptor at runtime; static assignment is invalid.

    setup_path = results / "entry" / "setup.yaml"
    yaml = YAML(typ="safe")
    with setup_path.open(encoding="utf-8") as stream:
        document = yaml.load(stream)
    document["general"]["entry_id"] = replacement_id
    document["general"]["description"] = "external annotation"
    with setup_path.open("w", encoding="utf-8") as stream:
        yaml.dump(document, stream)
    before = setup_path.read_bytes()

    with pytest.raises(ValueError, match="entry_id is immutable"):
        entry.setup.refresh()

    assert entry.entry_id == original_id
    assert entry.setup.description == "original annotation"
    assert setup_path.read_bytes() == before


def test_setup_description_uses_one_transaction_for_direct_and_edit_writes(
    entry_roots: tuple[Path, Path], entry: ResultEntry
) -> None:
    results, database = entry_roots
    assert entry.setup.description is None

    entry.setup.description = "chip annotation"
    with entry.setup.edit() as draft:
        draft.description = "updated annotation"

    assert entry.setup.description == "updated annotation"
    reopened = ResultEntry.open("entry", result_root=results, database_root=database)
    assert reopened.setup.description == "updated annotation"
    with (results / "entry" / "setup.yaml").open(encoding="utf-8") as stream:
        document = YAML(typ="safe").load(stream)
    assert document["general"]["description"] == "updated annotation"


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("entry_id", "not-a-uuid"),
        ("created_at", "yesterday"),
        ("created_at", "2026-10-04T06:00:00"),
        ("created_at", "2026-10-04T06:00:00+08:00"),
    ],
)
def test_open_validates_uuid_and_utc_created_at_before_publishing_entry(
    entry_roots: tuple[Path, Path],
    entry: ResultEntry,
    field: str,
    invalid: str,
) -> None:
    results, database = entry_roots
    setup_path = results / "entry" / "setup.yaml"
    yaml = YAML(typ="safe")
    with setup_path.open(encoding="utf-8") as stream:
        document = yaml.load(stream)
    document["general"][field] = invalid
    with setup_path.open("w", encoding="utf-8") as stream:
        yaml.dump(document, stream)
    before = setup_path.read_bytes()

    with pytest.raises(ValidationError) as failure:
        ResultEntry.open("entry", result_root=results, database_root=database)

    assert ("general", field) in [error["loc"] for error in failure.value.errors()]
    assert setup_path.read_bytes() == before


@pytest.mark.parametrize(
    "shape", ["empty-directory", "file", "broken-symlink", "external-symlink"]
)
def test_rename_refuses_existing_result_destination(
    entry_roots: tuple[Path, Path], entry: ResultEntry, shape: str
) -> None:
    results, database = entry_roots
    destination = results / "renamed"
    if shape == "empty-directory":
        destination.mkdir()
    elif shape == "file":
        destination.write_bytes(b"unrelated data")
    elif shape == "external-symlink":
        target = results.parent / "external"
        target.mkdir()
        (target / "external.bin").write_bytes(b"unrelated data")
        symlink_or_skip(destination, target, target_is_directory=True)
    else:
        symlink_or_skip(destination, "missing-target")
    before_result = read_entry_files(results / "entry")

    with pytest.raises(FileExistsError) as failure:
        rename_entry("entry", "renamed", result_root=results)

    assert failure.value.filename == str(destination)
    assert read_entry_files(results / "entry") == before_result
    assert (
        ResultEntry.open("entry", result_root=results, database_root=database).entry_id
        == entry.entry_id
    )
    if shape == "file":
        assert destination.read_bytes() == b"unrelated data"
    elif shape == "broken-symlink":
        assert destination.readlink() == Path("missing-target")
    elif shape == "external-symlink":
        assert (
            results.parent / "external" / "external.bin"
        ).read_bytes() == b"unrelated data"
        assert destination.is_symlink()
    else:
        assert destination.is_dir()


@pytest.mark.parametrize("database_exists", [False, True])
def test_rename_moves_only_result_and_retargets_future_save_paths(
    entry_roots: tuple[Path, Path], entry: ResultEntry, database_exists: bool
) -> None:
    results, database = entry_roots
    result_path = results / "entry"
    database_path = database / "entry"
    (result_path / "records" / "synthetic.json").write_bytes(b'{"unchanged": true}')
    if database_exists:
        database_path.mkdir(parents=True, exist_ok=True)
        (database_path / "data.bin").write_bytes(b"synthetic data")
        (database / "renamed").mkdir()
        (database / "renamed/other.bin").write_bytes(b"unrelated output")
    result_before = read_entry_files(result_path)
    database_before = read_entry_files(database)

    try:
        rename_entry("entry", "renamed", result_root=results)
    except TypeError as error:
        pytest.fail(f"Result-only rename must not require Database settings: {error}")

    assert not result_path.exists()
    assert read_entry_files(results / "renamed") == result_before
    assert read_entry_files(database) == database_before
    assert database_path.exists() == database_exists
    reopened = ResultEntry.open("renamed", result_root=results, database_root=database)
    assert reopened.entry_id == entry.entry_id
    assert reopened.database_path == database / "renamed"


@pytest.mark.parametrize("operation", ["create", "open"])
@pytest.mark.parametrize(
    "name",
    [
        "",
        ".",
        "..",
        "../escape",
        "nested/name",
        "nested\\\\name",
        "C:escape",
        "nul\u0000name",
        "absolute",
    ],
)
def test_entry_names_are_safe_single_path_components(
    entry_roots: tuple[Path, Path], name: str, operation: str
) -> None:
    results, database = entry_roots
    if name == "absolute":
        name = str(results.parent / "absolute-outside-root")
    action = ResultEntry.create if operation == "create" else ResultEntry.open

    with pytest.raises(ValueError, match="single path component"):
        action(name, result_root=results, database_root=database)


@pytest.mark.parametrize(
    "shape", ["legacy-directory", "file", "broken-symlink", "external-symlink"]
)
def test_create_rejects_existing_destination_without_touching_either_entry(
    entry_roots: tuple[Path, Path], shape: str
) -> None:
    results, database = entry_roots
    existing = results / "entry"
    existing.parent.mkdir()
    if shape == "legacy-directory":
        existing.mkdir()
        marker = existing / "legacy-data.hdf5"
        marker.write_bytes(b"original measurement")
    elif shape == "file":
        existing.write_bytes(b"original measurement")
        marker = existing
    elif shape == "external-symlink":
        target = existing.parent.parent / "external"
        target.mkdir()
        marker = target / "legacy-data.hdf5"
        marker.write_bytes(b"original measurement")
        symlink_or_skip(existing, target, target_is_directory=True)
    else:
        symlink_or_skip(existing, "missing-target")
        marker = None
    other = database / "entry"

    with pytest.raises(FileExistsError) as failure:
        ResultEntry.create("entry", result_root=results, database_root=database)

    assert failure.value.filename == str(existing)
    assert not other.exists()
    if marker is not None:
        assert marker.read_bytes() == b"original measurement"
    else:
        assert existing.is_symlink()
        assert existing.readlink() == Path("missing-target")


@pytest.mark.parametrize("missing", ["setup.yaml", "points", "records"])
def test_open_rejects_incomplete_entry_with_actual_missing_path(
    entry_roots: tuple[Path, Path], missing: str
) -> None:
    results, database = entry_roots
    ResultEntry.create("entry", result_root=results, database_root=database)
    missing_path = results / "entry" / missing
    if missing_path.is_dir():
        missing_path.rmdir()
    else:
        missing_path.unlink()

    with pytest.raises(FileNotFoundError) as failure:
        ResultEntry.open("entry", result_root=results, database_root=database)

    assert failure.value.filename == str(missing_path)


def test_open_reads_existing_identity_without_rewriting_setup(tmp_path: Path) -> None:
    results = tmp_path / "results"
    database = tmp_path / "Database"
    entry = ResultEntry.create(
        "plain (label)", result_root=results, database_root=database
    )
    setup_path = results / "plain (label)" / "setup.yaml"
    before = setup_path.read_bytes()

    reopened = ResultEntry.open(
        "plain (label)", result_root=results, database_root=database
    )

    assert reopened.entry_id == entry.entry_id
    assert setup_path.read_bytes() == before


def prepare_database_destination(database: Path, database_state: str) -> Path | None:
    """Arrange an unrelated output location and return its existing data file."""
    destination = database / "entry"
    marker = None
    target = database.parent / "external"
    if database_state == "root-file":
        database.write_bytes(b"unrelated root")
        marker = database
    elif database_state != "absent":
        database.mkdir()
        if database_state == "directory":
            destination.mkdir()
            marker = destination / "existing.bin"
            marker.write_bytes(b"unrelated output")
        elif database_state == "file":
            destination.write_bytes(b"unrelated output")
            marker = destination
        elif database_state == "broken-symlink":
            symlink_or_skip(destination, "missing-target")
        else:
            target.mkdir()
            marker = target / "existing.bin"
            marker.write_bytes(b"unrelated output")
            symlink_or_skip(destination, target, target_is_directory=True)
    return marker


@pytest.mark.parametrize("operation", ["create", "open"])
@pytest.mark.parametrize(
    "database_state",
    ["absent", "root-file", "directory", "file", "broken-symlink", "external-symlink"],
)
def test_entry_ignores_database_filesystem(
    entry_roots: tuple[Path, Path], operation: str, database_state: str
) -> None:
    results, database = entry_roots
    if operation == "open":
        original = ResultEntry.create(
            "entry", result_root=results, database_root=database.parent / "unused-db"
        )
        expected_id = original.entry_id
    else:
        expected_id = None
    destination = database / "entry"
    marker = prepare_database_destination(database, database_state)
    before = marker.read_bytes() if marker is not None else None
    action = ResultEntry.create if operation == "create" else ResultEntry.open

    try:
        opened = action("entry", result_root=results, database_root=database)
    except (OSError, ValueError) as error:
        pytest.fail(f"Entry {operation} must not inspect or create Database: {error}")

    assert opened.result_path == results / "entry"
    assert opened.database_path == destination
    if expected_id is not None:
        assert opened.entry_id == expected_id
    if marker is not None:
        assert marker.read_bytes() == before
    if database_state == "absent":
        assert not database.exists()
    elif database_state == "broken-symlink":
        assert destination.readlink() == Path("missing-target")
    elif database_state == "external-symlink":
        assert destination.readlink() == database.parent / "external"


def test_create_builds_new_format_entry_with_uuid_and_utc_identity(
    tmp_path: Path,
) -> None:
    results = tmp_path / "results"
    database = tmp_path / "Database"
    entry = ResultEntry.create("Q12_2D[3]", result_root=results, database_root=database)

    assert UUID(entry.entry_id).version == 4
    result_path = results / "Q12_2D[3]"
    assert (result_path / "records").is_dir()
    assert (result_path / "points").is_dir()
    assert entry.database_path == database / "Q12_2D[3]"
    assert not database.exists()
    with (result_path / "setup.yaml").open(encoding="utf-8") as stream:
        document = YAML(typ="safe").load(stream)
    assert document["format"] == "zcu.parameter-container"
    assert document["format_version"] == "2.0"
    assert document["metadata"] == {}
    assert document["general"]["entry_id"] == entry.entry_id
    assert datetime.fromisoformat(
        document["general"]["created_at"]
    ).utcoffset() == timezone.utc.utcoffset(None)
