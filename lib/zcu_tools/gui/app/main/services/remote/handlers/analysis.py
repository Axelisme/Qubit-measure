"""Analysis remote handlers."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

from zcu_tools.gui.remote.errors import ErrorCode, RemoteError

if TYPE_CHECKING:
    from ..service import RemoteControlAdapter


logger = logging.getLogger(__name__)


def _h_analyze_cancel(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    tab_id = str(params["tab_id"])
    if not adapter.ctrl.has_tab(tab_id):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
    # Graceful by contract (no interactive analyze in flight is not an error): the
    # cancelled flag tells the agent whether anything was actually settled.
    cancelled = adapter.ctrl.cancel_analyze(tab_id)
    return {"ok": True, "cancelled": cancelled}


def _h_tab_get_analyze_result(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    tab_id = str(params["tab_id"])
    if not adapter.ctrl.has_tab(tab_id):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
    result = adapter.ctrl.get_tab_analyze_result(tab_id)
    if result is None:
        return {"summary": None}
    to_summary = getattr(result, "to_summary_dict", None)
    if not callable(to_summary):
        raise RemoteError(
            ErrorCode.INTERNAL,
            "analyze result does not implement to_summary_dict()",
        )
    return {"summary": to_summary()}


def _h_tab_get_analyze_params(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    import dataclasses

    tab_id = str(params["tab_id"])
    if not adapter.ctrl.has_tab(tab_id):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
    snap = adapter.ctrl.get_tab_snapshot(tab_id)
    if snap.analyze_params is None:
        return {"analyze_params": None}
    ap = snap.analyze_params
    if not dataclasses.is_dataclass(ap) or isinstance(ap, type):
        return {"analyze_params": {}}
    return {"analyze_params": dataclasses.asdict(ap)}


def _h_tab_analyze(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    import dataclasses

    tab_id = str(params["tab_id"])
    if not adapter.ctrl.has_tab(tab_id):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
    snap = adapter.ctrl.get_tab_snapshot(tab_id)
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
    if snap.analyze_params is None:
        raise RemoteError(ErrorCode.PRECONDITION_FAILED, "no analyze params available")
    raw_updates = cast(dict, params["updates"])  # ParamSpec(_obj)-validated
    ap = snap.analyze_params
    if not dataclasses.is_dataclass(ap) or isinstance(ap, type):
        raise RemoteError(
            ErrorCode.INTERNAL, "analyze_params is not a dataclass instance"
        )
    try:
        updated = dataclasses.replace(ap, **raw_updates)
    except (TypeError, ValueError) as exc:
        raise RemoteError(ErrorCode.INVALID_PARAMS, str(exc)) from exc
    try:
        operation_id = adapter.ctrl.analyze(tab_id, updated)
    except RuntimeError as exc:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            str(exc),
            reason=getattr(exc, "reason_code", ""),
        ) from exc
    return {"operation_id": operation_id}


def _h_tab_get_post_analyze_result(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    tab_id = str(params["tab_id"])
    if not adapter.ctrl.has_tab(tab_id):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
    result = adapter.ctrl.get_post_analyze_result(tab_id)
    if result is None:
        return {"summary": None}
    to_summary = getattr(result, "to_summary_dict", None)
    if not callable(to_summary):
        raise RemoteError(
            ErrorCode.INTERNAL,
            "post-analysis result does not implement to_summary_dict()",
        )
    return {"summary": to_summary()}


def _h_tab_get_post_analyze_params(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    import dataclasses

    tab_id = str(params["tab_id"])
    if not adapter.ctrl.has_tab(tab_id):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
    snap = adapter.ctrl.get_tab_snapshot(tab_id)
    if snap.post_analyze_params is None:
        return {"post_analyze_params": None}
    pp = snap.post_analyze_params
    if not dataclasses.is_dataclass(pp) or isinstance(pp, type):
        return {"post_analyze_params": {}}
    return {"post_analyze_params": dataclasses.asdict(pp)}


def _h_tab_post_analyze(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    import dataclasses

    tab_id = str(params["tab_id"])
    if not adapter.ctrl.has_tab(tab_id):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
    snap = adapter.ctrl.get_tab_snapshot(tab_id)
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
    if snap.post_analyze_params is None:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED, "no post-analysis params available"
        )
    raw_updates = cast(dict, params["updates"])  # ParamSpec(_obj)-validated
    pp = snap.post_analyze_params
    if not dataclasses.is_dataclass(pp) or isinstance(pp, type):
        raise RemoteError(
            ErrorCode.INTERNAL, "post_analyze_params is not a dataclass instance"
        )
    try:
        updated = dataclasses.replace(pp, **raw_updates)
    except (TypeError, ValueError) as exc:
        raise RemoteError(ErrorCode.INVALID_PARAMS, str(exc)) from exc
    try:
        operation_id = adapter.ctrl.start_post_analyze(tab_id, updated)
    except RuntimeError as exc:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            str(exc),
            reason=getattr(exc, "reason_code", ""),
        ) from exc
    return {"operation_id": operation_id}
