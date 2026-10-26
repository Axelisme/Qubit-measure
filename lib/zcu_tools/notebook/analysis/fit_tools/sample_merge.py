"""Flatten and concatenate record histories without interpreting physical fields."""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from zcu_tools.resources.record_table import JsonlRecordTable


@dataclass(frozen=True)
class SampleMergeResult:
    """Merged table and its protected source paths.

    merged contains the latest rows from each source in source order, with
    nested object keys joined by dots. source_paths contains absolute JSONL
    paths, including empty sources, which write_merged_samples must not overwrite.
    """

    merged: pd.DataFrame
    source_paths: tuple[Path, ...]


def merge_sample_sources(*, sources: Iterable[str | Path]) -> SampleMergeResult:
    """Flatten each record JSONL source and concatenate rows in source order.

    Each source keeps the last whole row per run_id in last-occurrence order.
    Nested objects use dotted column names; lists remain cell values. Fields
    are preserved without interpreting aliases, units, flux or measurements.
    Missing fields become NaN when tables are concatenated. Source files are
    unchanged. An empty sources iterable raises ValueError; absent files count
    as empty sources. JSON syntax/run_id errors name the source path and line;
    other I/O errors propagate. Returns the table and protected source paths.
    """
    paths = tuple(Path(path).resolve() for path in sources)
    if not paths:
        raise ValueError("At least one source JSONL is required")
    parts = [JsonlRecordTable(path).read(flatten=True) for path in paths]
    return SampleMergeResult(pd.concat(parts, ignore_index=True), paths)


def write_merged_samples(
    result: SampleMergeResult, path: str | Path, *, index: bool = False
) -> Path:
    """Write result.merged as CSV to path, creating missing parent directories.

    Existing output files are replaced, except source JSONL paths (including
    symlink aliases), which raise ValueError before writing. index
    controls whether the DataFrame index is included. Returns the output Path.
    I/O errors propagate. This is not an atomic or multi-file transaction.
    """
    output = Path(path)
    if output.resolve() in result.source_paths:
        raise ValueError(f"refusing to overwrite source JSONL {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.merged.to_csv(output, index=index)
    return output
