"""Analysis remote handlers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from math import isfinite
from typing import TYPE_CHECKING, Literal, cast

from zcu_tools.gui.app.measure.adapter import AnalysisMode
from zcu_tools.gui.app.measure.adapter.analyze_params import (
    describe_analyze_params,
    reconstruct_params,
)
from zcu_tools.gui.remote.errors import ErrorCode, RemoteError

from ..analysis_types import AnalysisPresentation
from ..save_types import project_raw_data_state
from ._common import follow_tab
from .tab import tab_operation_state

if TYPE_CHECKING:
    from zcu_tools.gui.app.measure.services.ports import TabSnapshot

    from ..service import RemoteControlAdapter


def _summary_to_wire(summary: object) -> dict[str, object]:
    if not isinstance(summary, Mapping):
        raise RemoteError(ErrorCode.INTERNAL, "analysis summary must be an object")
    invalid: list[dict[str, str]] = []

    def project(value: object, path: str) -> object:
        if isinstance(value, float) and not isfinite(value):
            invalid.append({"path": path, "reason": "non_finite"})
            return None
        if isinstance(value, Mapping):
            projected: dict[str, object] = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    raise RemoteError(
                        ErrorCode.INTERNAL, "analysis summary keys must be strings"
                    )
                projected[key] = project(item, f"{path}.{key}")
            return projected
        if isinstance(value, (list, tuple)):
            return [
                project(item, f"{path}[{index}]") for index, item in enumerate(value)
            ]
        return value

    return {"summary": project(summary, "summary"), "invalid": invalid}


def _params_to_wire(params: object) -> dict[str, object] | None:
    if params is None:
        return None
    if not is_dataclass(params) or isinstance(params, type):
        return {}
    return asdict(params)


def _updated_params(
    current: object, updates: dict[str, object], *, label: str
) -> object:
    """Validate named wire edits against dataclass inputs or the no-input schema."""
    if current is not None and (not is_dataclass(current) or isinstance(current, type)):
        raise RemoteError(ErrorCode.INTERNAL, f"{label} is not a dataclass instance")
    values = {} if current is None else asdict(current)
    try:
        return reconstruct_params(type(current), {**values, **updates})
    except (RuntimeError, ValueError) as exc:
        definitions = describe_analyze_params(type(current))
        raise RemoteError(
            ErrorCode.INVALID_PARAMS,
            f"{exc}. Legal parameters: {definitions}",
            data={"definitions": definitions},
        ) from exc


def _presentation(
    snapshot: TabSnapshot, stage: Literal["primary", "post"]
) -> AnalysisPresentation:
    pane = snapshot.analysis if stage == "primary" else snapshot.post_analysis
    return AnalysisPresentation(
        run_id=None if snapshot.run is None else snapshot.run.acquisition_run_id,
        stage=stage,
        figure_names=list(() if pane is None else pane.figures or ()),
        figures_replaced=False if pane is None else pane.figures_replaced,
    )


def h_tab_get_analyze_result(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    tab_id = str(params["tab_id"])
    control = adapter.run_analyze_control
    if not control.has_tab(tab_id):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
    # tolerated(cast): optional_integer admits int or None; Mapping erases its type.
    operation_id = cast(int | None, params.get("operation_id"))
    if operation_id is not None:
        control.require_analysis_operation(tab_id, "analysis", operation_id)
    result = control.get_tab_analyze_result(tab_id)
    snap = control.get_tab_snapshot(tab_id)
    if result is None and (
        snap.interaction is None or not snap.interaction.has_analyze_result
    ):
        return {"summary": None, "invalid": []}
    if result is None:
        reply = _summary_to_wire({})
    else:
        to_summary = getattr(result, "to_summary_dict", None)
        if not callable(to_summary):
            raise RemoteError(
                ErrorCode.INTERNAL,
                "analyze result does not implement to_summary_dict()",
            )
        reply = _summary_to_wire(to_summary())
    if operation_id is not None:
        snap = control.get_tab_snapshot(tab_id)
        pane = snap.analysis
        reply.update(
            presentation=_presentation(snap, "primary"),
            params=_params_to_wire(None if pane is None else pane.result_params),
            operation_id=operation_id,
            operation_state=tab_operation_state(adapter, tab_id),
        )
    return reply


def h_tab_analyze(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    tab_id = str(params["tab_id"])
    control = adapter.run_analyze_control
    if not control.has_tab(tab_id):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
    # tolerated(cast): optional_integer admits int or None; Mapping erases its type.
    run_operation_id = cast(int | None, params.get("run_operation_id"))
    if run_operation_id is not None:
        control.require_run_operation(tab_id, run_operation_id)
    snap = control.get_tab_snapshot(tab_id)
    # Order the checks by the true cause: analyze params only exist once a run
    # produced a result (they are built from it). A run-in-flight / failed /
    # cancelled tab has no result, so report that — not the downstream "no
    # analyze params", which reads as a config gap rather than "nothing to
    # analyze yet".
    interaction = snap.interaction
    if interaction is not None and not interaction.has_run_result:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            "No run result available to analyze.",
            reason="no_run_result",
        )
    ap = None if snap.analysis is None else snap.analysis.params
    if ap is None and adapter.tab_control.analyze_param_definitions(
        tab_id, stage="primary"
    ):
        raise RemoteError(ErrorCode.PRECONDITION_FAILED, "no analyze params available")
    # tolerated(cast): default_object admits a JSON dict or {}; Mapping erases its type.
    raw_updates = cast(dict[str, object], params["updates"])
    updated = _updated_params(ap, raw_updates, label="analyze_params")
    if snap.capabilities is None:
        raise RemoteError(ErrorCode.INTERNAL, "snapshot has no capabilities")
    invalidated = []
    if snap.analysis is not None and snap.analysis.has_writeback_draft:
        invalidated.append("analysis.writeback")
    if snap.post_analysis is not None:
        if snap.post_analysis.result is not None or (
            snap.interaction is not None and snap.interaction.has_post_analyze_result
        ):
            invalidated.append("post.result")
        if snap.post_analysis.has_writeback_draft:
            invalidated.append("post.writeback")
    follow_tab(adapter, tab_id, "analysis")
    raw_data = project_raw_data_state(
        snap.artifacts,
        run_id=None if snap.run is None else snap.run.acquisition_run_id,
    )
    operation_id = control.analyze(tab_id, updated, run_operation_id=run_operation_id)
    return {
        "raw_data": raw_data.model_dump(mode="json"),
        "operation_id": operation_id,
        "interactive": snap.capabilities.analysis is AnalysisMode.INTERACTIVE,
        "params": _params_to_wire(updated),
        "invalidated_on_success": invalidated,
    }


def h_tab_get_post_analyze_result(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    tab_id = str(params["tab_id"])
    control = adapter.run_analyze_control
    if not control.has_tab(tab_id):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
    # tolerated(cast): optional_integer admits int or None; Mapping erases its type.
    operation_id = cast(int | None, params.get("operation_id"))
    if operation_id is not None:
        control.require_analysis_operation(tab_id, "post_analysis", operation_id)
    result = control.get_post_analyze_result(tab_id)
    snap = control.get_tab_snapshot(tab_id)
    if result is None and (
        snap.interaction is None or not snap.interaction.has_post_analyze_result
    ):
        return {"summary": None, "invalid": []}
    if result is None:
        reply = _summary_to_wire({})
    else:
        to_summary = getattr(result, "to_summary_dict", None)
        if not callable(to_summary):
            raise RemoteError(
                ErrorCode.INTERNAL,
                "post-analysis result does not implement to_summary_dict()",
            )
        reply = _summary_to_wire(to_summary())
    if operation_id is not None:
        snap = control.get_tab_snapshot(tab_id)
        pane = snap.post_analysis
        reply.update(
            presentation=_presentation(snap, "post"),
            params=_params_to_wire(None if pane is None else pane.result_params),
            operation_id=operation_id,
            operation_state=tab_operation_state(adapter, tab_id),
        )
    return reply


def h_tab_post_analyze(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    tab_id = str(params["tab_id"])
    control = adapter.run_analyze_control
    if not control.has_tab(tab_id):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
    # tolerated(cast): optional_integer admits int or None; Mapping erases its type.
    source_operation_id = cast(int | None, params.get("operation_id"))
    # tolerated(cast): optional_integer admits int or None; Mapping erases its type.
    run_operation_id = cast(int | None, params.get("run_operation_id"))
    if source_operation_id is not None:
        control.require_analysis_operation(tab_id, "analysis", source_operation_id)
    if run_operation_id is not None:
        control.require_run_operation(tab_id, run_operation_id)
    snap = control.get_tab_snapshot(tab_id)
    # Order the checks by the true cause: post params only exist once a primary
    # analyze produced a result (they are built from it). Report the missing
    # primary result first — it reads as "nothing to post-analyze yet" rather
    # than the downstream "no post params", which looks like a config gap.
    interaction = snap.interaction
    if interaction is not None and not interaction.has_analyze_result:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            "No primary analyze result available to post-analyze.",
            reason="no_analyze_result",
        )
    pp = None if snap.post_analysis is None else snap.post_analysis.params
    if pp is None and adapter.tab_control.analyze_param_definitions(
        tab_id, stage="post"
    ):
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED, "no post-analysis params available"
        )
    # tolerated(cast): default_object admits a JSON dict or {}; Mapping erases its type.
    raw_updates = cast(dict[str, object], params["updates"])
    updated = _updated_params(pp, raw_updates, label="post_analyze_params")
    invalidated = (
        ["post.writeback"]
        if snap.post_analysis is not None and snap.post_analysis.has_writeback_draft
        else []
    )
    read_timeout = control.analysis_result_read_timeout_seconds(tab_id)
    follow_tab(adapter, tab_id, "post_analysis")
    operation_id = control.start_post_analyze(
        tab_id,
        updated,
        operation_id=source_operation_id,
        run_operation_id=run_operation_id,
    )
    return {
        "operation_id": operation_id,
        "result_read_timeout_seconds": read_timeout,
        "recovery": {
            "rpc": "tab.get_post_analyze_result",
            "params": {"tab_id": tab_id, "operation_id": operation_id},
            "stage": "post",
        },
        "interactive": False,
        "params": _params_to_wire(updated),
        "invalidated_on_success": invalidated,
    }
