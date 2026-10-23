"""Saved-result wire projections; filesystem queries stay with ResultsIndex."""

from typing import Literal, TypedDict

from zcu_tools.gui.app.measure.services.results_index import ResultRun


class ResultRunReply(TypedDict):
    """One saved run, with identity/search/diagnostic semantics from ResultRun.

    key is an opaque project-local selector. entry names the Database entry.
    run_id, experiment, point and at retain unknown history as null; point can
    also be global. roles contains captured bindings only. registered means a
    matching acquired event. run_path, data_path, figure_paths and analysis_paths
    are exact absolute paths; data_path is null without data.h5. figures lists
    exact filenames accepted by results.figure. search_text is casefolded
    identity/cfg/analysis text including tone names. issues lists read failures.
    No full documents or arrays are included.
    """

    key: str
    entry: str
    run_id: str | None
    experiment: str | None
    point: str | None
    roles: dict[str, str]
    at: str | None
    registered: bool
    run_path: str
    data_path: str | None
    figure_paths: list[str]
    figures: list[str]
    analysis_paths: list[str]
    search_text: str
    issues: list[str]


class ResultsListReply(TypedDict):
    """runs contains every saved run in index order, including unregistered runs."""

    runs: list[ResultRunReply]


class ResultsFigureReply(TypedDict):
    """mime_type is image/png; png_b64 encodes the saved file's unchanged bytes."""

    mime_type: Literal["image/png"]
    png_b64: str


def project_result_run(run: ResultRun) -> ResultRunReply:
    """Project an indexed run to JSON carriers, without further reads or writes."""
    return {
        "key": run.key,
        "entry": run.entry,
        "run_id": run.run_id,
        "experiment": run.experiment,
        "point": run.point,
        "roles": dict(run.roles),
        "at": run.at,
        "registered": run.registered,
        "run_path": str(run.run_path),
        "data_path": str(run.data_path) if run.data_path is not None else None,
        "figure_paths": [str(path) for path in run.figure_paths],
        "figures": [path.name for path in run.figure_paths],
        "analysis_paths": [str(path) for path in run.analysis_paths],
        "search_text": run.search_text,
        "issues": list(run.issues),
    }
