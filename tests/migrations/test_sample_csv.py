"""Public offline conversion promises, using synthetic CSVs only."""

from pathlib import Path

import pytest
from zcu_tools.resources.entry import ResultEntry
from zcu_tools.resources.sample_table import LegacyDeviceValueUnit

from migrations import sample_csv
from migrations.__main__ import main
from migrations.storage_migration import MigrationInputError


@pytest.fixture
def entry(tmp_path: Path) -> ResultEntry:
    return ResultEntry.create(
        "synthetic",
        result_root=tmp_path / "results",
        database_root=tmp_path / "Database",
    )


def test_v2_conversion_renames_measurements_preserves_source_and_appends(
    tmp_path: Path, entry: ResultEntry
) -> None:
    source = tmp_path / "old.csv"
    source.write_text(
        "dev_value,dev_unit,Freq (MHz),T1 (us),T1err (us),T2r (us),"
        "T2r err (us),T2e (us),T2e err (us),Tcomment,date\n"
        "0.002,A,4500,50,2,30,1,60,3,合成,2020-01-01\n",
        encoding="utf-8",
    )
    before = source.read_bytes()
    entry.samples("Q1").append(run_id="existing", dev_value=0.1, dev_unit="V", t1_us=10)
    output = entry.result_path / "samples" / "Q1.jsonl"
    history = output.read_bytes()

    report = sample_csv.convert_sample_csv(
        [source], entry=entry, component="Q1", point="bias"
    )

    assert report.converted_rows == 1
    assert report.pending_rows == ()
    assert report.output_path == output
    assert source.read_bytes() == before
    assert output.read_bytes().startswith(history)
    row = entry.samples("Q1").read().set_index("run_id").loc["legacy-old-1"]
    assert row["dev_value"] == 0.002
    assert row["dev_unit"] == "A"
    assert row["point"] == "bias"
    assert row["source"] == "manual"
    assert row["freq_mhz"] == 4500
    assert row["t1_us"] == 50
    assert row["t1_err_us"] == 2
    assert row["t2r_us"] == 30
    assert row["t2r_err_us"] == 1
    assert row["t2e_us"] == 60
    assert row["t2e_err_us"] == 3


def test_synthetic_key_rerun_skips_equal_history_and_rejects_differences(
    tmp_path: Path, entry: ResultEntry
) -> None:
    source = tmp_path / "rerun.csv"
    source.write_text("dev_value,dev_unit,T1 (us),comment\n0.1,V,50,raw\n")
    first = sample_csv.convert_sample_csv([source], entry=entry, component="Q1")
    before = first.output_path.read_bytes(), first.output_path.stat().st_mtime_ns
    report = sample_csv.convert_sample_csv([source], entry=entry, component="Q1")
    assert report.converted_rows == 0, "Same synthetic key must not append history"
    assert len(report.skipped) == 1
    assert (
        first.output_path.read_bytes(),
        first.output_path.stat().st_mtime_ns,
    ) == before
    source.write_text("dev_value,dev_unit,T1 (us),comment\n0.1,V,51,changed\n")
    with pytest.raises(MigrationInputError) as caught:
        sample_csv.convert_sample_csv([source], entry=entry, component="Q1")
    assert any(
        row.source == source
        and row.destination == first.output_path
        and row.expected != row.actual
        for row in caught.value.differences
    )
    assert (
        first.output_path.read_bytes(),
        first.output_path.stat().st_mtime_ns,
    ) == before


def test_csv_cli_rerun_skips_and_conflict_exits_two(
    tmp_path: Path, entry: ResultEntry, capsys: pytest.CaptureFixture[str]
) -> None:
    import json
    source = tmp_path / "cli.csv"
    source.write_text("dev_value,dev_unit,T1 (us)\n0.1,V,50\n")
    arguments = [str(source), "--result-root", str(entry.result_path.parent),
                 "--database-root", str(entry.database_path.parent),
                 "--entry", "synthetic", "--component", "Q1"]
    assert main(arguments) == 0
    capsys.readouterr()
    output = entry.samples("Q1").path
    before = output.read_bytes(), output.stat().st_mtime_ns
    assert main(arguments) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["converted_rows"] == 0, "CLI must not append the same synthetic row"
    assert report["skipped"][0]["summary"] == "legacy-cli-1"
    assert (output.read_bytes(), output.stat().st_mtime_ns) == before
    source.write_text("dev_value,dev_unit,T1 (us)\n0.1,V,51\n")
    assert main(arguments) == 2
    assert "legacy-cli-1" in capsys.readouterr().err
    assert (output.read_bytes(), output.stat().st_mtime_ns) == before


def test_v2_native_unit_is_preserved_without_scaling(
    tmp_path: Path, entry: ResultEntry
) -> None:
    source = tmp_path / "native.csv"
    source.write_text("dev_value,dev_unit,T1 (us)\n1.25,native,50\n", encoding="utf-8")
    before = source.read_bytes()

    report = sample_csv.convert_sample_csv([source], entry=entry, component="Q1")

    assert report.converted_rows == 1
    assert report.pending_rows == ()
    row = entry.samples("Q1").read().iloc[0]
    assert row["run_id"] == "legacy-native-1"
    assert row["dev_value"] == 1.25
    assert row["dev_unit"] == "native"
    assert row["t1_us"] == 50
    assert row["source"] == "manual"
    assert row["point"] is None
    assert source.read_bytes() == before


@pytest.mark.parametrize(
    ("source_unit", "expected_value", "expected_unit"),
    [("A", 2.0, "A"), ("mA", 0.002, "A"), ("V", 2.0, "V"), ("mV", 0.002, "V")],
)
def test_legacy_conversion_requires_explicit_units(
    tmp_path: Path,
    entry: ResultEntry,
    source_unit: LegacyDeviceValueUnit,
    expected_value: float,
    expected_unit: str,
) -> None:
    source = tmp_path / "old.csv"
    # The header is not evidence for the unit; the caller's declaration is.
    source.write_text("calibrated mA,T1 (us)\n2,50\n", encoding="utf-8")
    before = source.read_bytes()

    missing = sample_csv.convert_sample_csv(
        [source], entry=entry, component="Q1", dev_value_column="calibrated mA"
    )
    assert missing.converted_rows == 0
    assert len(missing.pending_rows) == 1
    assert "source_unit" in missing.pending_rows[0].reason
    assert not missing.output_path.exists()

    report = sample_csv.convert_sample_csv(
        [source],
        entry=entry,
        component="Q1",
        dev_value_column="calibrated mA",
        source_unit=source_unit,
    )
    assert report.converted_rows == 1
    assert report.pending_rows == ()
    row = entry.samples("Q1").read().iloc[0]
    assert row["run_id"] == "legacy-old-1"
    assert row["point"] is None
    assert row["dev_value"] == expected_value
    assert row["dev_unit"] == expected_unit
    assert source.read_bytes() == before


def test_pending_coordinates_are_reported_and_unknown_measurements_are_preserved(
    tmp_path: Path, entry: ResultEntry
) -> None:
    known = tmp_path / "known.csv"
    known.write_text(
        "dev_value,dev_unit,T1 (us)\n0.1,V,50\nnot-a-number,V,60\n0.2,mV,70\n",
        encoding="utf-8",
    )
    unknown = tmp_path / "unknown.csv"
    unknown.write_text("dev_value,dev_unit,T1 error (us)\n0.1,V,1\n", encoding="utf-8")
    report = sample_csv.convert_sample_csv(
        [known, unknown], entry=entry, component="Q1"
    )
    assert report.converted_rows == 2
    assert [(row.source_path, row.row_number) for row in report.pending_rows] == [
        (known, 2),
        (known, 3),
    ]
    assert "dev_value" in report.pending_rows[0].reason
    assert "dev_unit" in report.pending_rows[1].reason
    assert list(entry.samples("Q1").read()["run_id"]) == [
        "legacy-known-1",
        "legacy-unknown-1",
    ]
    assert entry.samples("Q1").read().iloc[1]["T1 error (us)"] == "1"


@pytest.mark.parametrize("legacy", [False, True])
def test_unknown_csv_fields_keep_raw_values_in_plan_and_public_readback(
    tmp_path: Path, entry: ResultEntry, legacy: bool
) -> None:
    source = tmp_path / "synthetic.csv"
    coordinate_header = "calibrated mA" if legacy else "dev_value,dev_unit"
    first = "2" if legacy else "0.002,A"
    second = "3" if legacy else "0.003,A"
    source.write_text(
        f"{coordinate_header},T1 (us),Tcomment,comment,custom,empty\n"
        f"{first},50,known,Manual Added, 001 ,\n"
        f"{second},60,,Manual Added,NaN,\n",
        encoding="utf-8",
    )
    original = source.read_bytes()
    plan = sample_csv.plan_sample_csv(
        [source],
        dev_value_column="calibrated mA" if legacy else None,
        source_unit="mA" if legacy else None,
    )
    assert len(plan.rows) == 2
    assert plan.pending == ()
    expected = [
        (source, 1, "comment", "Manual Added", "legacy-synthetic-1"),
        (source, 1, "custom", " 001 ", "legacy-synthetic-1"),
        (source, 1, "empty", "", "legacy-synthetic-1"),
        (source, 2, "comment", "Manual Added", "legacy-synthetic-2"),
        (source, 2, "custom", "NaN", "legacy-synthetic-2"),
        (source, 2, "empty", "", "legacy-synthetic-2"),
    ]
    assert [
        (item.source_path, item.row_number, item.column, item.value, item.run_id)
        for item in plan.preserved_fields
    ] == expected
    report = sample_csv.convert_sample_csv(
        [source],
        entry=entry,
        component="Q1",
        dev_value_column="calibrated mA" if legacy else None,
        source_unit="mA" if legacy else None,
    )
    assert report.converted_rows == 2
    assert report.pending_rows == ()
    assert report.preserved_fields == plan.preserved_fields
    rows = entry.samples("Q1").read()
    assert list(rows["comment"]) == ["Manual Added", "Manual Added"]
    assert list(rows["custom"]) == [" 001 ", "NaN"]
    assert list(rows["empty"]) == ["", ""]
    assert list(rows["dev_value"]) == [0.002, 0.003]
    assert list(rows["dev_unit"]) == ["A", "A"]
    assert list(rows["t1_us"]) == [50, 60]
    assert rows.iloc[0]["tcomment"] == "known"
    last = entry.samples("Q1").latest("legacy-synthetic-2")
    assert last is not None
    assert last["tcomment"] is None
    assert source.read_bytes() == original


def test_unknown_csv_collisions_use_lossless_nested_fields_and_literal_report_keys(
    tmp_path: Path, entry: ResultEntry
) -> None:
    source = tmp_path / "collision.csv"
    source.write_text(
        "dev_value,dev_unit,T1 (us),Tcomment,t1_us,tcomment,run_id,point,source,at,"
        "legacy_fields,extra.with.dot\n"
        "0.1,V,50,known, 001 ,raw text,original id,old point,old source,old time,"
        "original container,literal\n",
        encoding="utf-8",
    )
    original = source.read_bytes()
    plan = sample_csv.plan_sample_csv([source])
    assert len(plan.rows) == 1
    assert plan.pending == ()
    report = sample_csv.convert_sample_csv(
        [source], entry=entry, component="Q1", point="new point"
    )
    assert report.converted_rows == 1
    assert report.pending_rows == ()
    row = entry.samples("Q1").latest("legacy-collision-1")
    assert row is not None
    assert row["run_id"] == "legacy-collision-1"
    assert row["point"] == "new point"
    assert row["source"] == "manual"
    assert row["at"] != "old time"
    assert row["t1_us"] == 50
    assert row["tcomment"] == "known"
    assert row["legacy_fields"] == {
        "t1_us": " 001 ",
        "tcomment": "raw text",
        "run_id": "original id",
        "point": "old point",
        "source": "old source",
        "at": "old time",
        "legacy_fields": "original container",
    }
    assert row["extra.with.dot"] == "literal"
    assert [item.destination_keys for item in report.preserved_fields] == [
        ("legacy_fields", "t1_us"),
        ("legacy_fields", "tcomment"),
        ("legacy_fields", "run_id"),
        ("legacy_fields", "point"),
        ("legacy_fields", "source"),
        ("legacy_fields", "at"),
        ("legacy_fields", "legacy_fields"),
        ("extra.with.dot",),
    ]
    assert report.preserved_fields == plan.preserved_fields
    assert source.read_bytes() == original


@pytest.mark.parametrize("second_name", ["old.csv", "old.txt"])
def test_duplicate_stems_fail_before_writing_any_input(
    tmp_path: Path, entry: ResultEntry, second_name: str
) -> None:
    source = tmp_path / "old.csv"
    source.write_text("dev_value,dev_unit,T1 (us)\n0.1,V,50\n", encoding="utf-8")
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    duplicate = other_dir / second_name
    duplicate.write_bytes(source.read_bytes())
    before = source.read_bytes()

    with pytest.raises(ValueError, match="stem.*old"):
        sample_csv.convert_sample_csv([source, duplicate], entry=entry, component="Q1")
    assert not (entry.result_path / "samples" / "Q1.jsonl").exists()
    assert source.read_bytes() == before
    assert duplicate.read_bytes() == before


def test_source_alias_of_output_is_rejected_without_changing_bytes(
    tmp_path: Path, entry: ResultEntry
) -> None:
    output = entry.result_path / "samples" / "Q1.jsonl"
    output.parent.mkdir()
    output.write_text("dev_value,dev_unit,T1 (us)\n0.1,V,50\n", encoding="utf-8")
    source = tmp_path / "old.csv"
    source.hardlink_to(output)
    before = source.read_bytes()

    with pytest.raises(ValueError, match="source"):
        sample_csv.convert_sample_csv([source], entry=entry, component="Q1")
    assert output.read_bytes() == before
    assert source.read_bytes() == before
