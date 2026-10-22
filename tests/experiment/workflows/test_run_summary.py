"""Persisted run summaries through the saved-run reader's public seam."""

from pathlib import Path

import pytest
from pydantic import BaseModel, TypeAdapter

from zcu_tools.experiment.workflows import EncodedRecord, saved_run
from zcu_tools.experiment.workflows.encoding import encode_record


class Summary(BaseModel):
    points: int


def read_summary(run: Path) -> EncodedRecord | None:
    """Call the reader and report the missing public operation as an assertion."""
    try:
        return saved_run.read_summary(run)
    except AttributeError as error:
        pytest.fail(f"Saved-run reading must support a run summary: {error}")


def test_read_summary_returns_the_detached_encoded_return(tmp_path: Path) -> None:
    encoded = encode_record(Summary(points=2))
    (tmp_path / "summary.json").write_bytes(TypeAdapter(EncodedRecord).dump_json(encoded))
    assert read_summary(tmp_path) == encoded


def test_read_summary_returns_none_only_when_the_file_is_missing(tmp_path: Path) -> None:
    assert read_summary(tmp_path) is None


def test_read_summary_rejects_a_missing_run_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_summary(tmp_path / "missing")


def test_read_summary_reports_the_malformed_source(tmp_path: Path) -> None:
    source = tmp_path / "summary.json"
    source.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="summary.json"):
        read_summary(tmp_path)
