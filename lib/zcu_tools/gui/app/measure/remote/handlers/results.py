"""Readonly saved-result queries through the same facade as the browser."""

from __future__ import annotations

from base64 import b64encode
from collections.abc import Mapping
from typing import TYPE_CHECKING

from zcu_tools.gui.remote.errors import ErrorCode, RemoteError

from ..results_types import (
    ResultsFigureReply,
    ResultsListReply,
    project_result_run,
)

if TYPE_CHECKING:
    from ..service import RemoteControlAdapter


def h_results_list(
    adapter: RemoteControlAdapter, _params: Mapping[str, object]
) -> ResultsListReply:
    """Read all project runs with no parameters or entry/point prerequisite.

    Each call scans a fresh disk snapshot; no filters or session changes apply.
    Index scan failures propagate to shared dispatch's structured wire errors.
    """
    index = adapter.ctrl.query_results()
    return {"runs": [project_result_run(run) for run in index.runs]}


def h_results_figure(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> ResultsFigureReply:
    """Read unchanged PNG bytes using list's run_key and exact figure filename.

    Unknown keys/selectors return invalid_params with unknown_result_run or
    unknown_result_figure. Each call rescans; I/O and escaping-path errors
    propagate to shared dispatch. No rendering, analysis, writes or retries.
    """
    key, figure = params["run_key"], params["figure"]
    if not isinstance(key, str) or not isinstance(figure, str):
        raise TypeError("Invalid validated results.figure request")
    index = adapter.ctrl.query_results()
    try:
        index.get_run(key)
    except KeyError as exc:
        raise RemoteError(
            ErrorCode.INVALID_PARAMS, str(exc), reason="unknown_result_run"
        ) from exc
    try:
        png = index.read_figure(key, figure)
    except KeyError as exc:
        raise RemoteError(
            ErrorCode.INVALID_PARAMS, str(exc), reason="unknown_result_figure"
        ) from exc
    return {"mime_type": "image/png", "png_b64": b64encode(png).decode("ascii")}
