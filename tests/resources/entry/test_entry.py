"""ResultEntry creation, identity and filesystem transactions through public APIs."""

from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

import pytest
from ruamel.yaml import YAML
from zcu_tools.resources.entry import PartialCommitError, ResultEntry, rename_entry


@pytest.fixture
def entry_roots(tmp_path: Path) -> tuple[Path, Path]:
    return tmp_path / "results", tmp_path / "Database"


@pytest.fixture
def entry(entry_roots: tuple[Path, Path]) -> ResultEntry:
    results, database = entry_roots
    return ResultEntry.create("entry", result_root=results, database_root=database)


def read_entry_files(path: Path) -> dict[Path, bytes]:
    return {
        item.relative_to(path): item.read_bytes()
        for item in path.rglob("*")
        if item.is_file()
    }


@pytest.mark.parametrize("recovery_reason", ["io-failure", "destination-reappeared"])
def test_failed_rename_recovery_reports_current_paths_and_both_causes(
    entry_roots: tuple[Path, Path],
    entry: ResultEntry,
    monkeypatch: pytest.MonkeyPatch,
    recovery_reason: str,
) -> None:
    results, database = entry_roots
    before = read_entry_files(results / "entry")
    original_rename = Path.rename
    cause = OSError("second rename failure")
    recovery_cause = OSError("rename recovery failure")

    def fail_second_and_recovery(source: Path, target: str | Path) -> Path:
        if source == database / "entry":
            if recovery_reason == "destination-reappeared":
                (results / "entry").mkdir()
                (results / "entry" / "external.bin").write_bytes(b"unrelated data")
            raise cause
        if source == results / "renamed" and recovery_reason == "io-failure":
            raise recovery_cause
        return original_rename(source, target)

    monkeypatch.setattr(Path, "rename", fail_second_and_recovery)
    with pytest.raises(PartialCommitError, match="Partial commit") as failure:
        rename_entry("entry", "renamed", result_root=results, database_root=database)

    error = failure.value
    assert error.completed == (results / "renamed",)
    assert error.pending == (database / "renamed",)
    assert error.recovery_failed == (results / "entry",)
    assert error.cause is cause
    assert error.__cause__ is cause
    if recovery_reason == "io-failure":
        assert error.recovery_cause is recovery_cause
    else:
        assert error.recovery_cause.filename == str(results / "entry")
        assert (results / "entry" / "external.bin").read_bytes() == b"unrelated data"
    assert read_entry_files(results / "renamed") == before
    assert (database / "entry").is_dir()
    assert not (database / "renamed").exists()
    assert UUID(entry.entry_id).version == 4


def test_rename_recovers_first_root_when_second_rename_fails(
    entry_roots: tuple[Path, Path], entry: ResultEntry, monkeypatch: pytest.MonkeyPatch
) -> None:
    results, database = entry_roots
    before = read_entry_files(results / "entry")
    original_rename = Path.rename
    cause = OSError("second rename failure")

    def fail_second(source: Path, target: str | Path) -> Path:
        if source == database / "entry":
            raise cause
        return original_rename(source, target)

    monkeypatch.setattr(Path, "rename", fail_second)
    with pytest.raises(OSError, match="second rename failure") as failure:
        rename_entry("entry", "renamed", result_root=results, database_root=database)

    assert failure.value is cause
    assert read_entry_files(results / "entry") == before
    assert not (results / "renamed").exists()
    assert not (database / "renamed").exists()
    assert (
        ResultEntry.open("entry", result_root=results, database_root=database).entry_id
        == entry.entry_id
    )


@pytest.mark.parametrize("root_index", [0, 1])
@pytest.mark.parametrize("shape", ["empty-directory", "file", "broken-symlink"])
def test_rename_refuses_existing_destination_before_moving_either_root(
    entry_roots: tuple[Path, Path], entry: ResultEntry, root_index: int, shape: str
) -> None:
    results, database = entry_roots
    destination = entry_roots[root_index] / "renamed"
    if shape == "empty-directory":
        destination.mkdir()
    elif shape == "file":
        destination.write_bytes(b"unrelated data")
    else:
        destination.symlink_to("missing-target")
    before_result = read_entry_files(results / "entry")
    before_database = read_entry_files(database / "entry")

    with pytest.raises(FileExistsError) as failure:
        rename_entry("entry", "renamed", result_root=results, database_root=database)

    assert failure.value.filename == str(destination)
    assert read_entry_files(results / "entry") == before_result
    assert read_entry_files(database / "entry") == before_database
    assert (
        ResultEntry.open("entry", result_root=results, database_root=database).entry_id
        == entry.entry_id
    )
    if shape == "file":
        assert destination.read_bytes() == b"unrelated data"
    elif shape == "broken-symlink":
        assert destination.readlink() == Path("missing-target")
    else:
        assert destination.is_dir()


def test_rename_moves_both_roots_preserving_identity_and_all_file_bytes(
    entry_roots: tuple[Path, Path], entry: ResultEntry
) -> None:
    results, database = entry_roots
    result_path = results / "entry"
    database_path = database / "entry"
    (result_path / "records" / "synthetic.json").write_bytes(b'{"unchanged": true}')
    (database_path / "data.bin").write_bytes(b"synthetic data")
    result_before = read_entry_files(result_path)
    database_before = read_entry_files(database_path)

    rename_entry(
        "entry", "renamed (no meaning)", result_root=results, database_root=database
    )

    assert not result_path.exists()
    assert not database_path.exists()
    assert read_entry_files(results / "renamed (no meaning)") == result_before
    assert read_entry_files(database / "renamed (no meaning)") == database_before
    assert (
        ResultEntry.open(
            "renamed (no meaning)", result_root=results, database_root=database
        ).entry_id
        == entry.entry_id
    )


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


def test_create_cleans_only_new_entry_when_second_root_cannot_be_created(
    entry_roots: tuple[Path, Path],
) -> None:
    results, database = entry_roots
    database.write_bytes(b"unrelated preexisting file")

    with pytest.raises(NotADirectoryError):
        ResultEntry.create("entry", result_root=results, database_root=database)

    assert not (results / "entry").exists()
    assert database.read_bytes() == b"unrelated preexisting file"


@pytest.mark.parametrize("root_index", [0, 1])
@pytest.mark.parametrize("shape", ["legacy-directory", "file", "broken-symlink"])
def test_create_rejects_existing_destination_without_touching_either_entry(
    entry_roots: tuple[Path, Path], root_index: int, shape: str
) -> None:
    results, database = entry_roots
    existing = entry_roots[root_index] / "entry"
    existing.parent.mkdir()
    if shape == "legacy-directory":
        existing.mkdir()
        marker = existing / "legacy-data.hdf5"
        marker.write_bytes(b"original measurement")
    elif shape == "file":
        existing.write_bytes(b"original measurement")
        marker = existing
    else:
        existing.symlink_to("missing-target")
        marker = None
    other = entry_roots[1 - root_index] / "entry"

    with pytest.raises(FileExistsError) as failure:
        ResultEntry.create("entry", result_root=results, database_root=database)

    assert failure.value.filename == str(existing)
    assert not other.exists()
    if marker is not None:
        assert marker.read_bytes() == b"original measurement"
    else:
        assert existing.is_symlink()
        assert existing.readlink() == Path("missing-target")


@pytest.mark.parametrize("missing", ["database", "setup.yaml", "points", "records"])
def test_open_rejects_incomplete_entry_with_actual_missing_path(
    entry_roots: tuple[Path, Path], missing: str
) -> None:
    results, database = entry_roots
    ResultEntry.create("entry", result_root=results, database_root=database)
    missing_path = (
        database / "entry" if missing == "database" else results / "entry" / missing
    )
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
    assert (database / "Q12_2D[3]").is_dir()
    with (result_path / "setup.yaml").open(encoding="utf-8") as stream:
        document = YAML(typ="safe").load(stream)
    assert document["format"] == "zcu.parameter-container"
    assert document["format_version"] == "1.0"
    assert document["general"]["entry_id"] == entry.entry_id
    assert datetime.fromisoformat(
        document["general"]["created_at"]
    ).utcoffset() == timezone.utc.utcoffset(None)
