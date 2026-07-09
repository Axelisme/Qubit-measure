"""Run Save remote handlers."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING

from zcu_tools.gui.remote.errors import ErrorCode, RemoteError

from ...load import LoadDataError

if TYPE_CHECKING:
    from ..service import RemoteControlAdapter


logger = logging.getLogger(__name__)


def _h_tab_run_start(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    tab_id = str(params["tab_id"])
    control = adapter.run_analyze_control
    if not control.has_tab(tab_id):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
    try:
        operation_id = control.start_run(tab_id)
    except RuntimeError as exc:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            str(exc),
            reason=getattr(exc, "reason_code", ""),
        ) from exc
    return {"operation_id": operation_id}


def _h_tab_load_data(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    import dataclasses

    tab_id = str(params["tab_id"])
    control = adapter.run_analyze_control
    if not control.has_tab(tab_id):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
    try:
        outcome = control.load_tab_result(tab_id, str(params["data_path"]))
    except LoadDataError as exc:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            str(exc),
            reason=exc.reason_code,
        ) from exc
    except (RuntimeError, OSError, ValueError) as exc:
        reason = getattr(exc, "reason_code", "")
        if isinstance(exc, NotImplementedError):
            reason = "unsupported_load"
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            str(exc),
            reason=reason,
        ) from exc

    snap = control.get_tab_snapshot(tab_id)
    interaction = snap.interaction
    assert interaction is not None
    result: dict[str, object] = dataclasses.asdict(outcome)
    result["has_run_result"] = bool(interaction.has_run_result)
    ap = snap.analyze_params
    if ap is None:
        result["analyze_params"] = None
    elif dataclasses.is_dataclass(ap) and not isinstance(ap, type):
        result["analyze_params"] = dataclasses.asdict(ap)
    else:
        result["analyze_params"] = {}
    return result


def _h_tab_run_cancel(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    # cancelled is best-effort: True when a live run was signalled, False on a
    # no-op. The worker's true terminal is observed via the run handle (ADR-0026
    # §8) — cancel only requests, it does not wait for the stop.
    cancelled = adapter.run_analyze_control.cancel_run()
    return {"ok": True, "cancelled": cancelled}


def _h_run_running_tab(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    return {"tab_id": adapter.run_analyze_control.get_running_tab_id()}


def _h_tab_save_data(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    tab_id = str(params["tab_id"])
    data_path = params["data_path"]
    comment = str(params["comment"])
    try:
        written = adapter.ctrl.save_data(
            tab_id, str(data_path) if data_path is not None else None, comment=comment
        )
    except RuntimeError as exc:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            str(exc),
            reason=getattr(exc, "reason_code", ""),
        ) from exc
    # The save runs async, but the resolved path (.hdf5 + uniqueness suffix) is
    # known synchronously — return it so the caller need not recover it from a
    # later diagnostic / snapshot.
    return {"data_path": written}


def _h_tab_save_image(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    tab_id = str(params["tab_id"])
    image_path = params["image_path"]
    try:
        written = adapter.ctrl.save_image(
            tab_id, str(image_path) if image_path is not None else None
        )
    except RuntimeError as exc:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            str(exc),
            reason=getattr(exc, "reason_code", ""),
        ) from exc
    return {"image_path": written}


def _h_tab_save_post_image(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    tab_id = str(params["tab_id"])
    image_path = params["image_path"]
    try:
        written = adapter.ctrl.save_post_image(
            tab_id, str(image_path) if image_path is not None else None
        )
    except RuntimeError as exc:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            str(exc),
            reason=getattr(exc, "reason_code", ""),
        ) from exc
    return {"image_path": written}


def _h_tab_save_result(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    tab_id = str(params["tab_id"])
    data_path = params["data_path"]
    image_path = params["image_path"]
    comment = str(params["comment"])
    try:
        written_data, written_image = adapter.ctrl.save_result(
            tab_id,
            str(data_path) if data_path is not None else None,
            str(image_path) if image_path is not None else None,
            comment=comment,
        )
    except RuntimeError as exc:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            str(exc),
            reason=getattr(exc, "reason_code", ""),
        ) from exc
    # The data save runs async, but both resolved paths (the data path's .hdf5 +
    # uniqueness suffix included) are known synchronously — return them so the
    # caller need not recover them from a later diagnostic.
    return {"data_path": written_data, "image_path": written_image}


def _h_tab_save_set_paths(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    tab_id = str(params["tab_id"])
    if not adapter.ctrl.has_tab(tab_id):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
    data_path = str(params["data_path"])
    image_path = str(params["image_path"])
    adapter.ctrl.update_tab_save_paths(tab_id, data_path, image_path)
    return {"data_path": data_path, "image_path": image_path}
