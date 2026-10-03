"""ResultEntry creation, identity and filesystem transactions through public APIs."""

from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from ruamel.yaml import YAML
from zcu_tools.resources.entry import ResultEntry


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
