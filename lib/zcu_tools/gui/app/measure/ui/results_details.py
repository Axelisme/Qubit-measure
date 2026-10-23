"""Read-only relative paths for saved cfg and analysis documents."""

from __future__ import annotations

from zcu_tools.gui.app.measure.services.results_index import ResultRun, ResultsIndex


def _lines(value: object, prefix: str = "") -> list[str]:
    if isinstance(value, dict):
        return [
            line
            for key, child in value.items()
            for line in _lines(child, f"{prefix}.{key}" if prefix else str(key))
        ]
    if isinstance(value, list):
        return [
            line
            for i, child in enumerate(value)
            for line in _lines(child, f"{prefix}[{i}]")
        ]
    return [f"{prefix} = {value}"]


def result_details(index: ResultsIndex, run: ResultRun) -> str:
    """Project one run's documents without cfg values/analysis summary root prefixes.

    Saved cfg_type/schema_version and analysis parameters have separate sections.
    Per-document read failures are
    displayed beside the affected filename rather than hiding the run.
    """
    sections = [
        f"{run.entry} / {run.point or 'Global / unknown'}",
        run.experiment or "Unknown experiment",
        run.run_path.name,
        "Registered" if run.registered else "Unregistered",
        f"Time: {run.at or 'Unknown'}",
        f"Roles: {dict(run.roles)}",
        str(run.run_path),
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
            sections.extend(["\nCfg values", *_lines(cfg.get("values", cfg))])
    for path in run.analysis_paths:
        try:
            document = index.read_analysis(run.key, path.name)
        except (OSError, ValueError, KeyError) as error:
            sections.append(f"{path.name}: {error}")
            continue
        sections.extend(
            [f"\nSummary · {path.name}", *_lines(document.get("summary", document))]
        )
        if "parameters" in document:
            sections.extend(
                [f"\nParameters · {path.name}", *_lines(document["parameters"])]
            )
    sections.extend(run.issues)
    return "\n".join(sections)
