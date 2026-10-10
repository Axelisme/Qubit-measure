"""Public offline conversion promises, using synthetic CSVs only."""

from pathlib import Path

import pytest
from zcu_tools.resources.entry import ResultEntry
from zcu_tools.resources.sample_table import LegacyDeviceValueUnit

from migrations import sample_csv


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


def test_pending_coordinates_and_unknown_measurements_are_reported_not_written(
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
    assert report.converted_rows == 1
    assert [(row.source_path, row.row_number) for row in report.pending_rows] == [
        (known, 2),
        (known, 3),
        (unknown, 1),
    ]
    assert "dev_value" in report.pending_rows[0].reason
    assert "dev_unit" in report.pending_rows[1].reason
    assert "T1 error (us)" in report.pending_rows[2].reason
    assert list(entry.samples("Q1").read()["run_id"]) == ["legacy-known-1"]


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
