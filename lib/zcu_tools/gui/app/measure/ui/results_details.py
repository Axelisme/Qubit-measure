"""Read-only relative paths for saved cfg and analysis documents."""

from __future__ import annotations

import json

from zcu_tools.gui.app.measure.services.results_index import ResultRun, ResultsIndex


def _items(value: object, prefix: str = "") -> list[tuple[str, str]]:
    if isinstance(value, dict):
        return [
            item
            for key, child in value.items()
            for item in _items(child, f"{prefix}.{key}" if prefix else str(key))
        ]
    if isinstance(value, list):
        return [
            item
            for i, child in enumerate(value)
            for item in _items(child, f"{prefix}[{i}]")
        ]
    return [(prefix, str(value))]


def _cfg_lines(value: object, prefix: str = "", depth: int = 0) -> list[str]:
    if isinstance(value, dict) and depth < 3:
        return [
            line
            for key, child in value.items()
            for line in _cfg_lines(
                child, f"{prefix}.{key}" if prefix else str(key), depth + 1
            )
        ]
    text = (
        json.dumps(value, ensure_ascii=False)
        if isinstance(value, (dict, list))
        else str(value)
    )
    return [f"{prefix} = {text[:200]}{' …' if len(text) > 200 else ''}"]


def result_details(index: ResultsIndex, run: ResultRun) -> str:
    """Return metadata and bounded cfg values without absolute path body rows.

    Cfg identity is separate from relative values. Values descend at most three
    mapping levels and truncate each rendered value to 200 characters.
    Read failures and indexed issues remain visible for diagnosis.
    """
    sections = [
        f"{run.entry} / {run.point or 'Global / unknown'}",
        run.experiment or "Unknown experiment",
        run.run_path.name,
        "Registered" if run.registered else "Unregistered",
        f"Time: {run.at or 'Unknown'}",
        f"Roles: {dict(run.roles)}",
    ]
    if run.data_path is not None:
        try:
            cfg = index.read_cfg(run.key)
        except (OSError, ValueError, KeyError) as error:
            sections.append(f"Cfg: {error}")
        else:
            sections.append("\nCfg identity")
            for field in ("cfg_type", "schema_version"):
                if field in cfg:
                    sections.append(f"{field} = {cfg[field]}")
            sections.extend(["\nCfg values", *_cfg_lines(cfg.get("values", cfg))])
    sections.extend(run.issues)
    return "\n".join(sections)


def analysis_details(
    index: ResultsIndex, run: ResultRun
) -> tuple[tuple[str, str], ...]:
    """Return key/value rows with relative summary keys and separate parameters.

    Each document starts with its summary heading. Read failures retain the
    filename and error as a row; no summary or quality facts are synthesized.
    """
    rows: list[tuple[str, str]] = []
    for path in run.analysis_paths:
        try:
            document = index.read_analysis(run.key, path.name)
        except (OSError, ValueError, KeyError) as error:
            rows.append((path.name, str(error)))
            continue
        rows.extend(
            [(f"Summary · {path.name}", ""), *_items(document.get("summary", document))]
        )
        if "parameters" in document:
            rows.extend(
                [(f"Parameters · {path.name}", ""), *_items(document["parameters"])]
            )
    return tuple(rows)
