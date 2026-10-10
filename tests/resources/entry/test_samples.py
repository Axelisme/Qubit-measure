"""Entry-bound sample history, latest-per-run reads and explicit CSV exports."""

import json
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
from zcu_tools.resources.entry import ResultEntry


def test_samples_append_preserves_history_and_reads_last_complete_row(
    entry: ResultEntry,
) -> None:
    samples = entry.samples("Q1")
    assert samples.read().empty
    path = entry.result_path / "samples" / "Q1.jsonl"
    assert not path.exists()

    samples.append(
        run_id="run-a", dev_value=0.001, dev_unit="A", point="bias", t1_us=20.0
    )
    first_bytes = path.read_bytes()
    samples.append(
        run_id="run-b",
        dev_value=0.2,
        dev_unit="V",
        freq_mhz=5000.0,
        fit={"label": "合成", "values": [1, 2]},
    )
    samples.append(
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

    latest = entry.samples("Q1").read().set_index("run_id")
    assert set(latest.index) == {"run-a", "run-b"}
    assert latest.loc["run-a", "t2r_us"] == 35.0
    assert latest.loc["run-a", "dev_value"] == 0.002
    assert "t1_us" not in latest.columns  # last line replaces, not merges, the row
    assert latest.loc["run-b", "fit"] == {"label": "合成", "values": [1, 2]}
    assert entry.samples("Q2").read().empty


def test_samples_support_workflow_source_and_native_device_unit(
    entry: ResultEntry,
) -> None:
    samples = entry.samples("Q1")
    samples.append(
        run_id="workflow-run",
        dev_value=12.0,
        dev_unit="native",
        source="workflow:workflow-run",
        t1_us=40.0,
    )
    row = samples.read().iloc[0]
    assert row["source"] == "workflow:workflow-run"
    assert row["dev_unit"] == "native"
    assert row["dev_value"] == 12.0


def test_samples_support_accepted_event_source(entry: ResultEntry) -> None:
    samples = entry.samples("Q1")
    samples.append(
        run_id="run-a", dev_value=1.0, dev_unit="V", source="accepted-event-id"
    )
    assert samples.read().iloc[0]["source"] == "accepted-event-id"


@pytest.mark.parametrize("missing", ["run_id", "dev_value", "dev_unit"])
def test_samples_reject_missing_required_coordinates_without_writing(
    entry: ResultEntry, missing: str
) -> None:
    samples = entry.samples("Q1")
    # Widen the callable only to exercise runtime rejection of invalid calls.
    append: Callable[..., None] = samples.append
    if missing == "run_id":
        with pytest.raises(TypeError, match="run_id"):
            append(dev_value=1.0, dev_unit="A")
    elif missing == "dev_value":
        with pytest.raises(TypeError, match="dev_value"):
            append(run_id="run-a", dev_unit="A")
    else:
        with pytest.raises(TypeError, match="dev_unit"):
            append(run_id="run-a", dev_value=1.0)
    assert not (entry.result_path / "samples" / "Q1.jsonl").exists()


def test_samples_reject_empty_run_id_without_writing(entry: ResultEntry) -> None:
    samples = entry.samples("Q1")
    with pytest.raises(ValueError, match="run_id"):
        samples.append(run_id="", dev_value=1.0, dev_unit="A")
    assert samples.read().empty


def test_samples_export_csv_requires_explicit_overwrite(
    entry: ResultEntry, tmp_path: Path
) -> None:
    samples = entry.samples("Q1")
    samples.append(run_id="run-a", dev_value=1.0, dev_unit="V", t1_us=15.0)
    samples.append(run_id="run-a", dev_value=2.0, dev_unit="V", t1_us=25.0)
    destination = tmp_path / "export.csv"
    destination.write_text("keep me\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="export.csv"):
        samples.export_csv(destination)
    assert destination.read_text(encoding="utf-8") == "keep me\n"
    samples.export_csv(destination, overwrite=True)
    exported = pd.read_csv(destination)
    pd.testing.assert_frame_equal(exported, samples.read())
    new_destination = tmp_path / "fresh.csv"
    samples.export_csv(new_destination)
    pd.testing.assert_frame_equal(pd.read_csv(new_destination), samples.read())


@pytest.mark.parametrize("component", ["", "..", "../Q1", "Q1/Q2", "Q1\\Q2", "C:\\Q1"])
def test_samples_reject_unsafe_component_paths(
    entry: ResultEntry, component: str
) -> None:
    with pytest.raises(ValueError, match="component"):
        entry.samples(component)
