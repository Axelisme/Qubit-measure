"""Read record history without validating producer fields that are not used."""

from pathlib import Path

import numpy as np
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
