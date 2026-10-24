"""Synthetic saved runs for comparison service and component scenarios."""

import json
from pathlib import Path

from zcu_tools.datafile import JsonObject
from zcu_tools.gui.app.measure.services.results_index import ResultRun


def saved_run(
    root: Path, name: str, experiment: str | None, summary: JsonObject
) -> ResultRun:
    """Write one analysis member and return its detached historical identity."""
    path = root / "Database" / "sample" / "runs" / name
    analysis = path / "analysis" / "analysis.json"
    analysis.parent.mkdir(parents=True)
    analysis.write_text(json.dumps({"summary": summary}), encoding="utf-8")
    return ResultRun(
        key=f"sample/{name}",
        entry="sample",
        run_id=name,
        experiment=experiment,
        point=name,
        roles={},
        at=None,
        registered=False,
        run_path=path,
        data_path=None,
        figure_paths=(),
        analysis_paths=(analysis,),
        search_text="",
        issues=(),
    )
