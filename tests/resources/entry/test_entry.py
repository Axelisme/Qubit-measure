"""ResultEntry creation, identity and filesystem transactions through public APIs."""

from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

import pytest
from ruamel.yaml import YAML
from zcu_tools.resources.entry import ResultEntry


@pytest.fixture
def entry_roots(tmp_path: Path) -> tuple[Path, Path]:
    return tmp_path / "results", tmp_path / "Database"


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
