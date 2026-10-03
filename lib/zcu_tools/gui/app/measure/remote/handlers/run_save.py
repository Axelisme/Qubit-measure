"""Run Save remote handlers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

from zcu_tools.gui.app.measure.artifact_tracker import ArtifactKey, ArtifactKind
from zcu_tools.gui.cfg.edit_codec import decode_ref
from zcu_tools.gui.cfg.resource import CfgInputError, CfgPreconditionError
from zcu_tools.gui.remote.errors import ErrorCode, RemoteError

from ..artifact_keys import artifact_key_wire, parse_artifact_key
from ..cfg_observation import cfg_error_to_remote
from ._common import follow_tab

if TYPE_CHECKING:
    from ..service import RemoteControlAdapter


def h_tab_run_start(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    tab_id = str(params["tab_id"])
    control = adapter.run_analyze_control
    if not control.has_tab(tab_id):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
    try:
        expected = decode_ref(params["expected"])
        follow_tab(adapter, tab_id, "run")
        operation_id = control.start_run(tab_id, expected)
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
    # no-op. The worker's true terminal is observed via the run handle (ADR-0066
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
        run_operation_id=cast(int | None, params["run_operation_id"]),
    )
    return {"data_path": written.data_path, "operation_id": written.operation_id}


def h_tab_save_artifacts(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    raw_artifacts = params["artifacts"]
    if raw_artifacts == "all":
        artifacts = None
    elif isinstance(raw_artifacts, list):
        artifacts = tuple(parse_artifact_key(key) for key in raw_artifacts)
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
    paths: dict[ArtifactKey, str] = {}
    for key, path in raw_paths.items():
        artifact_key = parse_artifact_key(key)
        if not isinstance(path, str) or not path:
            raise RemoteError(
                ErrorCode.INVALID_PARAMS, "artifact paths must be nonempty strings"
            )
        paths[artifact_key] = path
    comment = params["comment"]
    tab_id = str(params["tab_id"])
    if not adapter.save_control.has_tab(tab_id):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
    follow_tab(adapter, tab_id, "data")
    written = adapter.save_control.save_artifacts(
        tab_id,
        artifacts=artifacts,
        paths=paths,
        comment=str(comment) if comment is not None else None,
    )
    return {
        "operation_id": written.operation_id,
        "destinations": {
            artifact_key_wire(item.key): item.path for item in written.destinations
        },
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
    name = params["figure_name"]
    if not isinstance(name, str) or not name:
        raise RemoteError(ErrorCode.INVALID_PARAMS, "figure_name must be nonempty")
    kind = (
        ArtifactKind.ANALYSIS if subtab_id == "analysis" else ArtifactKind.POST_ANALYSIS
    )
    image_path = params["image_path"]
    path_str = str(image_path) if image_path is not None else None
    operation_id = cast(int | None, params.get("operation_id"))
    written = adapter.save_control.save_image(
        tab_id, ArtifactKey(kind, name), path_str, operation_id=operation_id
    )
    return {"image_path": written}
