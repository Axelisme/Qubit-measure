"""Run Save remote handlers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from zcu_tools.gui.app.main.artifact_tracker import ArtifactKind
from zcu_tools.gui.remote.errors import ErrorCode, RemoteError

if TYPE_CHECKING:
    from ..service import RemoteControlAdapter


def h_tab_run_start(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    tab_id = str(params["tab_id"])
    control = adapter.run_analyze_control
    if not control.has_tab(tab_id):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
    operation_id = control.start_run(tab_id)
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
    result: dict[str, object] = dataclasses.asdict(outcome)
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
    # no-op. The worker's true terminal is observed via the run handle (ADR-0026
    # §8) — cancel only requests, it does not wait for the stop.
    cancelled = adapter.run_analyze_control.cancel_run()
    return {"ok": True, "cancelled": cancelled}


def h_run_running_tab(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    return {"tab_id": adapter.run_analyze_control.get_running_tab_id()}


def h_tab_save_data(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    tab_id = str(params["tab_id"])
    data_path = params["data_path"]
    comment = params["comment"]
    written = adapter.save_control.save_data(
        tab_id,
        str(data_path) if data_path is not None else None,
        comment=str(comment) if comment is not None else None,
    )
    return {"data_path": written.data_path, "operation_id": written.operation_id}


_ARTIFACT_KINDS = {
    "data": ArtifactKind.DATA,
    "analysis": ArtifactKind.ANALYSIS,
    "post": ArtifactKind.POST_ANALYSIS,
}


def _artifact_kind(key: object) -> ArtifactKind:
    if not isinstance(key, str) or key not in _ARTIFACT_KINDS:
        raise RemoteError(
            ErrorCode.INVALID_PARAMS, "artifact keys must be data, analysis or post"
        )
    return _ARTIFACT_KINDS[key]


def h_tab_save_artifacts(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    raw_artifacts = params["artifacts"]
    if raw_artifacts == "all":
        artifacts = None
    elif isinstance(raw_artifacts, list):
        artifacts = tuple(_artifact_kind(key) for key in raw_artifacts)
        if not artifacts or len(set(artifacts)) != len(artifacts):
            raise RemoteError(
                ErrorCode.INVALID_PARAMS,
                "artifacts must be a nonempty list of unique keys",
            )
    else:
        raise RemoteError(
            ErrorCode.INVALID_PARAMS, "artifacts must be 'all' or a list of keys"
        )
    raw_paths = params["paths"]
    if not isinstance(raw_paths, dict):
        raise RemoteError(ErrorCode.INVALID_PARAMS, "paths must be an object")
    paths: dict[ArtifactKind, str] = {}
    for key, path in raw_paths.items():
        kind = _artifact_kind(key)
        if not isinstance(path, str) or not path:
            raise RemoteError(
                ErrorCode.INVALID_PARAMS, "artifact paths must be nonempty strings"
            )
        paths[kind] = path
    comment = params["comment"]
    written = adapter.save_control.save_artifacts(
        str(params["tab_id"]),
        artifacts=artifacts,
        paths=paths,
        comment=str(comment) if comment is not None else None,
    )
    keys = {kind: key for key, kind in _ARTIFACT_KINDS.items()}
    return {
        "operation_id": written.operation_id,
        "destinations": {keys[item.kind]: item.path for item in written.destinations},
    }


def h_tab_save_image(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    tab_id = str(params["tab_id"])
    subtab_id = str(params["subtab_id"])
    if subtab_id not in ("analysis", "post_analysis"):
        raise RemoteError(
            ErrorCode.INVALID_PARAMS,
            f"invalid subtab_id {subtab_id!r}; save_image only accepts analysis|post_analysis",
        )
    image_path = params["image_path"]
    path_str = str(image_path) if image_path is not None else None
    if subtab_id == "analysis":
        written = adapter.save_control.save_image(tab_id, path_str)
    else:
        written = adapter.save_control.save_post_image(tab_id, path_str)
    return {"image_path": written}
