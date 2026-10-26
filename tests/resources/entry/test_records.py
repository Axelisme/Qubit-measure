"""Entry-bound record history, latest-per-run reads and explicit CSV exports."""

import json
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
from pydantic import JsonValue
from zcu_tools.resources.entry import ResultEntry
from zcu_tools.resources.record_table import JsonlRecordTable

from tests._filesystem import symlink_or_skip


def test_component_record_binding_cannot_alias_the_event_ledger(
    entry: ResultEntry,
) -> None:
    symlink_or_skip(
        entry.result_path / "records" / "components",
        entry.result_path / "records",
        target_is_directory=True,
    )
    with pytest.raises(ValueError, match="records escape"):
        entry.records("ledger")


def test_records_append_preserves_history_and_reads_last_complete_row(
    entry: ResultEntry,
) -> None:
    records = entry.records("Q1")
    assert records.read().empty
    path = entry.result_path / "records" / "components" / "Q1.jsonl"
    assert not path.exists()

    records.append(
        run_id="run-a", dev_value=0.001, dev_unit="A", point="bias", t1_us=20.0
    )
    first_bytes = path.read_bytes()
    records.append(
        run_id="run-b",
        dev_value=0.2,
        dev_unit="V",
        freq_mhz=5000.0,
        fit={"label": "合成", "values": [1, 2]},
    )
    records.append(
        run_id="run-a", dev_value=0.002, dev_unit="A", point="bias", t2r_us=35.0
    )

    assert path.read_bytes().startswith(first_bytes)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 3
    assert rows[0]["t1_us"] == 20.0
    assert rows[-1]["t2r_us"] == 35.0
    assert all(
        "flux" not in row and "flux_int" not in row and "flux_period" not in row
        for row in rows
    )
    for row in rows:
        assert {
            "run_id",
            "point",
            "at",
            "source",
            "dev_value",
            "dev_unit",
        } <= row.keys()
        assert row["source"] == "manual"
        assert datetime.fromisoformat(row["at"]).utcoffset() == timedelta(0)
    assert rows[1]["point"] is None

    latest = entry.records("Q1").read().set_index("run_id")
    assert set(latest.index) == {"run-a", "run-b"}
    assert latest.loc["run-a", "t2r_us"] == 35.0
    assert latest.loc["run-a", "dev_value"] == 0.002
    assert "t1_us" not in latest.columns  # last line replaces, not merges, the row
    assert latest.loc["run-b", "fit"] == {"label": "合成", "values": [1, 2]}
    assert entry.records("Q2").read().empty


@pytest.mark.parametrize("component", ["Q1", "Q2"])
def test_append_records_carries_bound_component(
    entry: ResultEntry, component: str
) -> None:
    records = entry.records(component)
    row = records.append(run_id="run-a", component="producer-name", t1_us=20.0)

    assert row["component"] == component
    assert entry.records(component).latest("run-a") == row
    assert entry.records(component).read()["component"].tolist() == [component]


def test_old_records_without_component_remain_readable(entry: ResultEntry) -> None:
    records = entry.records("Q1")
    legacy = {
        "run_id": "old-run",
        "point": None,
        "at": "2020-01-01T00:00:00Z",
        "source": "manual",
        "t1_us": 10.0,
    }
    records.path.parent.mkdir(parents=True, exist_ok=True)
    records.path.write_text(json.dumps(legacy) + "\n", encoding="utf-8")
    original = records.path.read_bytes()

    assert records.latest("old-run") == legacy
    assert records.read().to_dict(orient="records") == [legacy]
    assert JsonlRecordTable(records.path).latest("old-run") == legacy
    assert records.path.read_bytes() == original


def test_explicit_component_does_not_depend_on_jsonl_filename(tmp_path: Path) -> None:
    path = tmp_path / "unrelated-temporary-name.jsonl"
    records = JsonlRecordTable(path, component="Q2")
    row = records.append(run_id="run-a", t1_us=30.0)

    assert row["component"] == "Q2"
    assert JsonlRecordTable(path).latest("run-a") == row


@pytest.mark.parametrize("component", [None, ""])
def test_append_without_bound_component_fails_before_writing(
    tmp_path: Path, component: str | None
) -> None:
    records = JsonlRecordTable(tmp_path / "Q1.jsonl", component=component)
    assert records.read().empty
    with pytest.raises(ValueError, match="nonempty bound component"):
        records.append(run_id="run-a", component="producer-name")
    assert not records.path.exists()


def test_records_support_workflow_source_and_native_device_unit(
    entry: ResultEntry,
) -> None:
    records = entry.records("Q1")
    records.append(
        run_id="workflow-run",
        dev_value=12.0,
        dev_unit="native",
        source="workflow:workflow-run",
        t1_us=40.0,
    )
    row = records.read().iloc[0]
    assert row["source"] == "workflow:workflow-run"
    assert row["dev_unit"] == "native"
    assert row["dev_value"] == 12.0


def test_records_support_accepted_event_source(entry: ResultEntry) -> None:
    records = entry.records("Q1")
    records.append(
        run_id="run-a", dev_value=1.0, dev_unit="V", source="accepted-event-id"
    )
    assert records.read().iloc[0]["source"] == "accepted-event-id"


def test_records_reject_missing_run_id_without_writing(entry: ResultEntry) -> None:
    records = entry.records("Q1")
    # Widen the callable only to exercise runtime rejection of invalid calls.
    append: Callable[..., dict[str, JsonValue]] = records.append
    with pytest.raises(TypeError, match="run_id"):
        append(dev_value=1.0, dev_unit="A")
    assert not (entry.result_path / "records" / "components" / "Q1.jsonl").exists()


def test_records_reject_empty_run_id_without_writing(entry: ResultEntry) -> None:
    records = entry.records("Q1")
    with pytest.raises(ValueError, match="run_id"):
        records.append(run_id="", dev_value=1.0, dev_unit="A")
    assert records.read().empty


def test_records_export_csv_requires_explicit_overwrite(
    entry: ResultEntry, tmp_path: Path
) -> None:
    records = entry.records("Q1")
    records.append(
        run_id="run-a", dev_value=1.0, dev_unit="V", point="bias", t1_us=15.0
    )
    records.append(
        run_id="run-a", dev_value=2.0, dev_unit="V", point="bias", t1_us=25.0
    )
    destination = tmp_path / "export.csv"
    destination.write_text("keep me\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="export.csv"):
        records.export_csv(destination)
    assert destination.read_text(encoding="utf-8") == "keep me\n"
    records.export_csv(destination, overwrite=True)
    exported = pd.read_csv(destination)
    pd.testing.assert_frame_equal(exported, records.read())
    new_destination = tmp_path / "fresh.csv"
    records.export_csv(new_destination)
    pd.testing.assert_frame_equal(pd.read_csv(new_destination), records.read())


@pytest.mark.parametrize("kind", ["canonical", "symlink", "hardlink"])
@pytest.mark.parametrize("overwrite", [False, True])
def test_records_export_rejects_own_history_without_changing_bytes(
    entry: ResultEntry, tmp_path: Path, kind: str, overwrite: bool
) -> None:
    records = entry.records("Q1")
    records.append(run_id="run-a", dev_value=1.0, dev_unit="A", t1_us=15.0)
    source = entry.result_path / "records" / "components" / "Q1.jsonl"
    original = source.read_bytes()
    destination = source
    if kind == "symlink":
        destination = tmp_path / "record-link.csv"
        symlink_or_skip(destination, source)
    elif kind == "hardlink":
        destination = tmp_path / "record-hardlink.csv"
        destination.hardlink_to(source)
    with pytest.raises(ValueError, match="record history"):
        records.export_csv(destination, overwrite=overwrite)
    assert source.read_bytes() == original
    assert records.read().iloc[0]["t1_us"] == 15.0


@pytest.mark.parametrize("component", ["", "..", "../Q1", "Q1/Q2", "Q1\\Q2", "C:\\Q1"])
def test_records_reject_unsafe_component_paths(
    entry: ResultEntry, component: str
) -> None:
    with pytest.raises(ValueError, match="component"):
        entry.records(component)
