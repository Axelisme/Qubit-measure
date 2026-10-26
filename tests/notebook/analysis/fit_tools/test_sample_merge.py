"""Record concatenation and CSV output without physical-field interpretation."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from zcu_tools.notebook.analysis.fit_tools import (
    merge_sample_sources,
    write_merged_samples,
)


def test_merge_concatenates_latest_source_rows_and_preserves_values(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.jsonl"
    first.write_text(
        '{"run_id":"a","flux":0.1,"dev_value":2}\n'
        '{"run_id":"b","Flux":9,"unused":NaN}\n'
        '{"run_id":"a","flux":0.7,"dev_value":3,"dev_unit":"raw","flux_int":6}\n',
        encoding="utf-8",
    )
    second = tmp_path / "second.jsonl"
    second.write_text('{"run_id":"a","extra":"metadata only"}\n', encoding="utf-8")
    originals = [first.read_bytes(), second.read_bytes()]

    result = merge_sample_sources(sources=(first, second))

    assert result.merged["run_id"].tolist() == ["b", "a", "a"]
    np.testing.assert_allclose(
        result.merged["flux"], [np.nan, 0.7, np.nan], equal_nan=True
    )
    assert result.merged.loc[1, "dev_value"] == 3
    assert result.merged.loc[1, "dev_unit"] == "raw"
    assert result.merged.loc[1, "flux_int"] == 6
    assert result.merged.loc[0, "Flux"] == 9
    assert result.merged.loc[2, "extra"] == "metadata only"
    assert [first.read_bytes(), second.read_bytes()] == originals


def test_merge_flattens_nested_records_into_csv(tmp_path: Path) -> None:
    source = tmp_path / "nested.jsonl"
    source.write_text(
        '{"run_id":"a","devices":{"bias":{"value":3,"value_unit":"raw"}},'
        '"extra":{"nested":{"flag":true}},"tags":["x","y"],"unused":NaN}\n'
        '{"run_id":"b","at":"local time","extra":{"nested":{"flag":false}}}\n',
        encoding="utf-8",
    )

    result = merge_sample_sources(sources=[source])

    assert result.merged["devices.bias.value"].iloc[0] == 3
    assert result.merged["devices.bias.value_unit"].iloc[0] == "raw"
    assert result.merged["extra.nested.flag"].tolist() == [True, False]
    assert result.merged["tags"].iloc[0] == ["x", "y"]
    assert result.merged["at"].iloc[1] == "local time"
    output = write_merged_samples(result, tmp_path / "output" / "samples.csv")
    exported = pd.read_csv(output)
    assert exported["devices.bias.value"].iloc[0] == 3
    assert exported["devices.bias.value_unit"].iloc[0] == "raw"
    assert exported["extra.nested.flag"].tolist() == [True, False]


@pytest.mark.parametrize("empty", [False, True])
def test_write_protects_source_histories(tmp_path: Path, empty: bool) -> None:
    source = tmp_path / "records.jsonl"
    source.write_text("" if empty else '{"run_id":"a","flux":0.5}\n', encoding="utf-8")
    original = source.read_bytes()
    result = merge_sample_sources(sources=[source])

    with pytest.raises(ValueError, match="refusing to overwrite source JSONL"):
        write_merged_samples(result, source)

    assert source.read_bytes() == original
    output = tmp_path / "output" / "samples.csv"
    assert write_merged_samples(result, output) == output
    assert output.exists()


def test_merge_requires_at_least_one_source() -> None:
    with pytest.raises(ValueError, match="At least one source JSONL"):
        merge_sample_sources(sources=[])
