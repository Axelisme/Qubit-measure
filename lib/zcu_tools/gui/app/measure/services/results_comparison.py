"""Saved comparison values, independent of Qt and live analysis."""

import json
from dataclasses import dataclass

from pydantic import JsonValue

from .results_index import ResultRun, ResultsIndex

_IDENTITY = ("entry", "point", "experiment", "time")


@dataclass(frozen=True)
class ComparisonRow:
    """run_key identifies the pin; cells follow ComparisonTable.columns."""

    run_key: str
    cells: tuple[str, ...]


@dataclass(frozen=True)
class ComparisonTable:
    """Ordered column paths and input-ordered rows; missing cells contain —."""

    columns: tuple[str, ...]
    rows: tuple[ComparisonRow, ...]


def build_comparison(index: ResultsIndex, run_keys: tuple[str, ...]) -> ComparisonTable:
    """Project saved summary leaves after entry/point/experiment/time columns.

    Rows follow run_keys; unknown keys raise KeyError. Summary column paths form
    an insertion-ordered union across pins and saved JSON keys. Values retain
    saved units and scale; parameters are not projected. Objects use dotted
    paths. Nonempty all-object lists union leaf paths with indexed multiline
    cells (missing leaves show —); other lists use JSON text. Null shows null,
    empty objects add no columns. Multiple analysis members merge cells with
    filename prefixes rather than overwrite them. Unreadable members contribute
    no cells, without hiding their run or other members. ResultsIndex owns file
    reads and containment validation.
    """
    runs = tuple(index.get_run(key) for key in run_keys)
    summaries = tuple(_run_summary(index, run) for run in runs)
    paths = tuple(dict.fromkeys(key for summary in summaries for key in summary))
    rows = tuple(
        ComparisonRow(
            run.key,
            (
                run.entry,
                run.point or "—",
                run.experiment or "—",
                run.at or "—",
                *(summary.get(path, "—") for path in paths),
            ),
        )
        for run, summary in zip(runs, summaries, strict=True)
    )
    return ComparisonTable((*_IDENTITY, *paths), rows)


def _flatten(value: JsonValue, path: str = "") -> dict[str, str]:
    if isinstance(value, dict):
        cells: dict[str, str] = {}
        for key, child in value.items():
            cells.update(_flatten(child, f"{path}.{key}" if path else key))
        return cells
    if isinstance(value, list) and value:
        elements = [_flatten(item) for item in value if isinstance(item, dict)]
        if len(elements) == len(value):
            keys = dict.fromkeys(key for element in elements for key in element)
            return {
                f"{path}.{key}": "\n".join(
                    f"{index}: {element.get(key, '—')}"
                    for index, element in enumerate(elements)
                )
                for key in keys
            }
    return {
        path: value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    }


def _run_summary(index: ResultsIndex, run: ResultRun) -> dict[str, str]:
    cells: dict[str, str] = {}
    for path in run.analysis_paths:
        try:
            document = index.read_analysis(run.key, path.name)
        except (OSError, ValueError):
            # A broken member must not hide other saved summaries or the run.
            continue
        summary = document.get("summary")
        if isinstance(summary, dict):
            for key, value in _flatten(summary).items():
                if len(run.analysis_paths) > 1:
                    value = f"{path.name}: {value}"
                cells[key] = f"{cells[key]}\n{value}" if key in cells else value
    return cells
