"""Saved comparison values, independent of Qt and live analysis."""

import json
import re
from dataclasses import dataclass

from pydantic import JsonValue

from zcu_tools.datafile import JsonObject

from .results_index import ResultRun, ResultsIndex

# Explicit historical names/path tokens and saved keys; never load lab adapters.
_KIND_FIELDS = (
    (("t1",), ("t1", "t1_err", "t1b")),
    (("ramsey", "t2ramsey", "t2r"), ("t2r", "t2r_err", "detune")),
    (("echo", "t2echo", "t2e"), ("t2e", "t2e_err")),
    (("amp_rabi", "rabi_amplitude"), ("pi_gain", "pi_gain_err", "pi2_gain")),
    (("len_rabi", "rabi_length"), ("pi_len", "pi_len_err", "pi2_len", "rabi_f")),
    (
        ("twotone_spectrum", "qubit_spectrum", "twotone/freq"),
        ("freq", "freq_err", "fwhm"),
    ),
    (("onetone", "resonator"), ("freq", "fwhm", "kappa", "depth")),
    (("ge", "singleshot_ge"), ("fidelity", "theta", "threshold", "ge_s")),
)
_IDENTITY = ("entry", "point", "experiment", "time")
_FACTS = ("r2", "invalid", "failure", "issues")
_MUX_FIELDS = {"t2r": "t2", "t2r_err": "t2_err", "t2e": "t2", "t2e_err": "t2_err"}


def _fields(run: ResultRun) -> tuple[str, ...]:
    experiment = (run.experiment or "").casefold()
    tokens = re.split(r"[/ .-]+", experiment)
    for names, fields in _KIND_FIELDS:
        if any(name in tokens or name == experiment for name in names):
            if "mux" in tokens:
                return tuple(_MUX_FIELDS.get(key, key) for key in fields)
            return fields
    return ()


def _display(value: JsonValue) -> str:
    if value is None:
        return "—"
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _quality_cell(quality: JsonValue, key: str, prefix: str = "") -> str:
    if not isinstance(quality, dict):
        return "—"
    facts = []
    if key in quality:
        facts.append(
            f"{prefix}: {_display(quality[key])}" if prefix else _display(quality[key])
        )
    for name, value in quality.items():
        if isinstance(value, dict):
            child = _quality_cell(value, key, f"{prefix} / {name}" if prefix else name)
            if child != "—":
                facts.append(child)
    return "\n".join(facts) or "—"


def _leaf_cell(leaf: JsonObject, key: str, prefix: str = "") -> str:
    if key in ("r2", "invalid"):
        quality = _quality_cell(leaf.get("fit_quality"), key, prefix)
        if quality != "—":
            return quality
    value = _display(leaf.get(key)) if key != "invalid" else "—"
    return f"{prefix}: {value}" if prefix else value


def _summary_cell(summary: JsonObject, key: str) -> str:
    groups = [summary.get("tones"), summary.get("resonators")]
    cells = []
    for group in groups:
        if isinstance(group, list):
            for index, leaf in enumerate(group):
                if not isinstance(leaf, dict):
                    continue
                label = next(
                    (
                        leaf.get(name)
                        for name in ("label", "name", "index", "position")
                        if leaf.get(name) is not None
                    ),
                    index,
                )
                cells.append(_leaf_cell(leaf, key, _display(label)))
    return "\n".join(cells) if cells else _leaf_cell(summary, key)


@dataclass(frozen=True)
class ComparisonRow:
    """run_key identifies the pin; cells follow ComparisonTable.columns."""

    run_key: str
    cells: tuple[str, ...]


@dataclass(frozen=True)
class ComparisonTable:
    """Ordered column keys and input-ordered rows; unavailable cells contain —."""

    columns: tuple[str, ...]
    rows: tuple[ComparisonRow, ...]


def build_comparison(index: ResultsIndex, run_keys: tuple[str, ...]) -> ComparisonTable:
    """Project indexed saved summaries without analysis or unit conversion.

    Rows follow run_keys; unknown keys raise KeyError. Fixed numeric columns
    form a kind-ordered union. Unknown experiments expose identity and saved
    quality/issues only. Missing values display —. Per-file read errors appear
    in issues without hiding the run; ResultsIndex owns all reads/validation.
    """
    runs = tuple(index.get_run(key) for key in run_keys)
    requested = {key for run in runs for key in _fields(run)}
    ordered = tuple(
        dict.fromkeys(
            key
            for _, fields in _KIND_FIELDS
            for field in fields
            for key in (field, _MUX_FIELDS.get(field, field))
        )
    )
    columns = (*_IDENTITY, *(key for key in ordered if key in requested), *_FACTS)
    rows = tuple(_comparison_row(index, run, columns) for run in runs)
    return ComparisonTable(columns, rows)


def _comparison_row(
    index: ResultsIndex, run: ResultRun, columns: tuple[str, ...]
) -> ComparisonRow:
    values: dict[str, str] = dict(
        zip(
            _IDENTITY,
            (run.entry, run.point or "—", run.experiment or "—", run.at or "—"),
            strict=True,
        )
    )
    issues = list(run.issues)
    for path in run.analysis_paths:
        try:
            document = index.read_analysis(run.key, path.name)
        except (OSError, ValueError) as error:
            issues.append(str(error))
            continue
        summary = document.get("summary")
        if isinstance(summary, dict):
            for key in (*_fields(run), "r2", "invalid", "failure"):
                cell = _summary_cell(summary, key)
                if len(run.analysis_paths) > 1:
                    cell = f"{path.name}: {cell}"
                values[key] = f"{values[key]}\n{cell}" if key in values else cell
    values["issues"] = "\n".join(issues) or "—"
    return ComparisonRow(run.key, tuple(values.get(key, "—") for key in columns))
