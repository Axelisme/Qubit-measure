"""Run Save remote handlers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from pydantic import ValidationError

from zcu_tools.experiment.run_saving import ArtifactRef
from zcu_tools.gui.cfg.resource import CfgInputError, CfgPreconditionError
from zcu_tools.gui.remote.errors import ErrorCode, RemoteError
from zcu_tools.gui.session.context_types import CommandOrigin

from ..cfg_observation import cfg_error_to_remote
from ..load_types import project_load_outcome
from ..save_types import SaveArtifactRefWire, project_save_report
from ._common import follow_tab
from .tab import temporary_aliases

if TYPE_CHECKING:
    from ..service import RemoteControlAdapter


def h_tab_run_start(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    tab_id = str(params["tab_id"])
    aliases = temporary_aliases(params)
    if aliases:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            "Run aliases are not supported; Reset the tab with aliases first",
            reason="run_aliases_not_supported",
        )
    control = adapter.run_analyze_control
    if not control.has_tab(tab_id):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
    try:
        follow_tab(adapter, tab_id, "run")
        operation_id = control.start_run(
            tab_id, aliases=aliases, origin=CommandOrigin("mcp")
        )
    except (CfgInputError, CfgPreconditionError) as exc:
        raise cfg_error_to_remote(exc) from exc
    return {"operation_id": operation_id}


def h_tab_load_data(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    import dataclasses

    tab_id = str(params["tab_id"])
    control = adapter.run_analyze_control
    if not control.has_tab(tab_id):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
    outcome = control.load_tab_result(tab_id, str(params["data_path"]))

    snap = control.get_tab_snapshot(tab_id)
    interaction = snap.interaction
    assert interaction is not None
    result = project_load_outcome(outcome)
    result["has_run_result"] = bool(interaction.has_run_result)
    ap = None if snap.analysis is None else snap.analysis.params
    if ap is None:
        result["analyze_params"] = None
    elif dataclasses.is_dataclass(ap) and not isinstance(ap, type):
        result["analyze_params"] = dataclasses.asdict(ap)
    else:
        result["analyze_params"] = {}
    return result


def h_tab_run_cancel(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    # cancelled is best-effort: True when a live run was signalled, False on a
    # no-op. The worker's true terminal is observed via the run handle (ADR-0066
    # §8) — cancel only requests, it does not wait for the stop.
    cancelled = adapter.run_analyze_control.cancel_run()
    return {"ok": True, "cancelled": cancelled}


def h_run_running_tab(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    return {"tab_id": adapter.run_analyze_control.get_running_tab_id()}


def _operation_id(value: object, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"{field} must be an integer")
    return value


def h_tab_save_data(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    tab_id = str(params["tab_id"])
    written = adapter.save_control.save_data(
        tab_id,
        comment=str(params["comment"]) if params["comment"] is not None else None,
        origin=CommandOrigin("mcp"),
        run_operation_id=_operation_id(params["run_operation_id"], "run_operation_id"),
    )
    snap = adapter.tab_control.get_tab_snapshot(tab_id)
    projection = project_save_report(written, snap.artifacts)
    return projection.model_dump(mode="json")


def h_tab_save_artifacts(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    raw_artifacts = params["artifacts"]
    if raw_artifacts == "all":
        artifacts = None
    elif isinstance(raw_artifacts, list):
        try:
            refs = [SaveArtifactRefWire.model_validate(item) for item in raw_artifacts]
            artifacts = tuple(ArtifactRef(ref.section, ref.name) for ref in refs)
        except (ValidationError, ValueError) as error:
            raise RemoteError(ErrorCode.INVALID_PARAMS, str(error)) from error
        if not artifacts or len(set(artifacts)) != len(artifacts):
            raise RemoteError(
                ErrorCode.INVALID_PARAMS, "artifacts must be nonempty and unique"
            )
    else:
        raise RemoteError(
            ErrorCode.INVALID_PARAMS, "artifacts must be all or artifact refs"
        )
    tab_id = str(params["tab_id"])
    if not adapter.save_control.has_tab(tab_id):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
    origin = CommandOrigin("mcp")
    run_operation_id = _operation_id(params["run_operation_id"], "run_operation_id")
    operation_id = _operation_id(params["operation_id"], "operation_id")
    if run_operation_id is not None:
        adapter.run_analyze_control.require_run_operation(tab_id, run_operation_id)
    if operation_id is not None:
        if (
            artifacts is None
            or len(artifacts) != 1
            or artifacts[0].section not in ("analysis", "post_analysis")
        ):
            raise RemoteError(
                ErrorCode.INVALID_PARAMS, "operation_id requires one analysis artifact"
            )
        adapter.run_analyze_control.require_analysis_operation(
            tab_id, artifacts[0].section, operation_id
        )
    follow_tab(adapter, tab_id, "data")
    written = adapter.save_control.save_artifacts(
        tab_id,
        artifacts=artifacts,
        comment=str(params["comment"]) if params["comment"] is not None else None,
        origin=origin,
    )
    snap = adapter.tab_control.get_tab_snapshot(tab_id)
    projection = project_save_report(written, snap.artifacts)
    return projection.model_dump(mode="json")
