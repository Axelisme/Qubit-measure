"""Read record history without validating producer fields that are not used."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from zcu_tools.resources.record_table import JsonlRecordTable


def test_read_and_latest_preserve_unused_fields_without_metadata_validation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text(
        '{"run_id":"a","t1_us":1}\n'
        '{"run_id":"b","t1_us":2}\n'
        '{"run_id":"a","t1_us":80,"unused":NaN,'
        '"at":"local time","source":{},"extra":{"future":true}}\n',
        encoding="utf-8",
    )
    table = JsonlRecordTable(path)

    rows = table.read()

    assert rows["run_id"].tolist() == ["b", "a"]
    assert rows["t1_us"].tolist() == [2, 80]
    assert np.isnan(rows.loc[1, "unused"])
    latest = table.latest("a")
    assert latest is not None
    assert latest["at"] == "local time"
    assert latest["extra"] == {"future": True}
    assert latest["source"] == {}


@pytest.mark.parametrize(
    "fields",
    [
        '"note.kind":"root","note":{"kind":"nested"}',
        '"note":{"kind":"nested"},"note.kind":"root"',
        '"note":{"kind.label":"root","kind":{"label":"nested"}}',
        '"note.kind":["root"],"note":{"kind":["nested"]}',
        '"note.kind":"same","note":{"kind":"same"}',
    ],
)
def test_flatten_rejects_colliding_columns_without_losing_raw_fields(
    tmp_path: Path, fields: str
) -> None:
    path = tmp_path / "records.jsonl"
    source = '{"run_id":"first","extra":NaN}\n' + (
        '{"run_id":"collision",' + fields + "}\n"
    )
    path.write_text(source, encoding="utf-8")
    table = JsonlRecordTable(path)
    before = table.read()
    expected = json.loads('{"run_id":"collision",' + fields + "}")
    assert before.loc[1, "note"] == expected["note"]
    if "note.kind" in expected:
        assert before.loc[1, "note.kind"] == expected["note.kind"]

    with pytest.raises(ValueError, match=r"note\.kind") as error:
        table.read(flatten=True)

    message = str(error.value)
    assert str(path) in message
    assert "collision" in message
    assert "row 1" in message
    assert path.read_text(encoding="utf-8") == source
    pd.testing.assert_frame_equal(table.read(), before)


def test_flatten_preserves_noncolliding_dotted_keys_and_nested_metadata(
    tmp_path: Path,
) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text(
        '{"run_id":"mixed","note.kind":"root","note":{"other":"nested"}}\n'
        '{"run_id":"nested","note":{"kind":"other row"}}\n',
        encoding="utf-8",
    )

    rows = JsonlRecordTable(path).read(flatten=True)

    assert rows["note.kind"].tolist() == ["root", "other row"]
    assert rows.loc[0, "note.other"] == "nested"
