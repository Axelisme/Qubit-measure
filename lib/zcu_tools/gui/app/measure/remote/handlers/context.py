"""Context remote handlers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np

from zcu_tools.gui.expected_error import InvalidInputError
from zcu_tools.gui.remote.errors import ErrorCode, RemoteError
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.context_types import LibraryChange, SavedPoint
from zcu_tools.gui.session.library_entry import LibraryEntryInvalid
from zcu_tools.gui.session.value_lookup import ValueInfo

from ._wire_values import context_wire_value

if TYPE_CHECKING:
    from ..service import RemoteControlAdapter


def h_context_aliases_set(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    """Apply complete ordered authored rows through the context owner.

    Each row has exactly string alias/value fields. The owner validates the
    whole table, saves the selected point's aliases.yaml atomically, then
    installs/publishes it. No selected point raises no_context before I/O.
    Input/storage failures preserve the previous table. Return installed
    authored rows without resolving them; success does not imply resolvability.
    No dialog, selection, other point-container or ledger effects.
    """
    raw = params["rows"]
    if not isinstance(raw, list):
        raise RemoteError(ErrorCode.INVALID_PARAMS, "rows must be a list")
    rows: list[AliasRow] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict) or set(item) != {"alias", "value"}:
            raise RemoteError(
                ErrorCode.INVALID_PARAMS,
                f"rows[{index}] must contain exactly alias and value",
            )
        alias, value = item["alias"], item["value"]
        if not isinstance(alias, str) or not isinstance(value, str):
            raise RemoteError(
                ErrorCode.INVALID_PARAMS,
                f"rows[{index}].alias and value must be strings",
            )
        rows.append(AliasRow(alias, value))
    adapter.context_control.apply_alias_rows(tuple(rows))
    return {
        "rows": [
            {"alias": row.alias, "value": row.value}
            for row in adapter.context_control.get_alias_rows()
        ]
    }


def h_context_set_parameter(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    """Set one concrete path at the currently selected point, preserving metadata.

    Fix the target and before value with one pre-write capture. Missing/null
    request values are rejected by the method entry; nested YAML null is valid.
    Before is null for both unset and explicit null. Return the submitted value,
    not a post-write readback. No ledger, dialog, selection or retry effects.
    Owner errors propagate. Run uses its fixed capture, so direct writes during
    Run are allowed without changing that capture.
    """
    path = str(params["path"])
    if path.startswith("$"):
        raise RemoteError(
            ErrorCode.INVALID_PARAMS, "path must be a concrete parameter path"
        )
    try:
        value = context_wire_value(params["value"])
    except (TypeError, ValueError, RecursionError) as exc:
        raise RemoteError(
            ErrorCode.INVALID_PARAMS, f"invalid parameter value: {exc}"
        ) from exc
    capture = adapter.context_control.read_context()
    before = context_wire_value(capture.lookup_point(path).value)
    adapter.context_control.set_parameter(capture.target, path, value, metadata=None)
    return {"path": path, "before": before, "value": value}


def h_context_ml_edit(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    """Write one atomic Form edit batch to a library entry; reply {valid, errors}.

    Edits use tab.edit_cfg paths and values. Values carrying the retired
    __kind tags are rejected with the expression form before any write.
    """
    from ..form_observation import form_writes

    raw = params["edits"]
    if not isinstance(raw, list) or not raw:
        raise RemoteError(ErrorCode.INVALID_PARAMS, "edits must be a nonempty list")
    for index, item in enumerate(raw):
        value = item.get("value") if isinstance(item, dict) else None
        if isinstance(value, dict) and "__kind" in value:
            raise RemoteError(
                ErrorCode.INVALID_PARAMS,
                f"edits[{index}].value uses a retired __kind tag; write an "
                'expression as {"expr": "<text>"} and other values directly',
            )
    batch = form_writes(raw)
    kind = params["kind"]
    # The method entry restricts kind to these names; this narrows its type.
    if kind not in ("module", "waveform"):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown library kind {kind!r}")
    save_as = params["save_as"]
    errors = adapter.ctrl.edit_library_entry(
        kind,
        str(params["name"]),
        batch,
        save_as=str(save_as) if save_as is not None else None,
    )
    if errors is None:
        return {"valid": True, "errors": []}
    return {
        "valid": False,
        "errors": [
            {"path": list(path), "message": message}
            for path, message in errors.messages.items()
        ],
    }


def h_context_labels(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    """Return cached point labels in the opened entry, empty without an entry."""
    del params
    return {"labels": [point.label for point in adapter.context_control.list_points()]}


def h_context_active(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    """Return the published saved-point label, or None before point selection.

    Reads only the cached parameter target. Component focus and tab role changes
    cannot change this session-level source identity. No refresh or I/O occurs.
    """
    del params
    parameters = adapter.ctrl.get_session_env().parameters
    target = parameters.target if parameters is not None else None
    return {"label": target.point if isinstance(target, SavedPoint) else None}


def h_context_snapshot(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    """Read one fixed native capture with complete resolved alias rows.

    Optional summaries=True describes array/list leaves using md_get rules.
    Optional path selects a full subtree of target/point/library/aliases by
    dotted mapping keys or canonical nonnegative list indices; it takes priority
    over summaries. Invalid paths raise InvalidInputError naming the first bad
    segment. Path replies include target, aliases, path and value.
    No refresh, file I/O or asset materialization occurs. EMPTY/source faults
    retain owner errors. Unsupported/nonfinite values fail the entire read with
    PRECONDITION_FAILED/unserializable_context, even in partial modes.
    This read does not establish a per-connection mutation baseline.
    """
    parameters = adapter.context_control.read_context()
    try:
        aliases: list[dict[str, object]] = []
        for row in parameters.resolved_aliases():
            projected: dict[str, object] = {
                "alias": row.alias,
                "value": row.value,
                "resolved": row.resolved,
            }
            if row.error is not None:
                projected["error"] = row.error
            aliases.append(projected)
        snapshot: dict[str, object] = {
            "target": parameters.target.model_dump(mode="python"),
            # Registered component subclasses own the complete persisted fields.
            "point": parameters.document().model_dump(
                mode="python", serialize_as_any=True
            ),
            "library": parameters.library.document().model_dump(
                mode="python", serialize_as_any=True
            ),
            "aliases": aliases,
        }
        # Validate the complete capture even when a projection hides some values.
        wire: dict[str, object] = json.loads(
            json.dumps(context_wire_value(snapshot), allow_nan=False)
        )
        path = params.get("path")
        if isinstance(path, str):
            return {
                "target": wire["target"],
                "aliases": wire["aliases"],
                "path": path,
                "value": _snapshot_path_value(wire, path),
            }
        if params.get("summaries", False):
            summary = {
                key: aliases if key == "aliases" else _snapshot_summary(value)
                for key, value in snapshot.items()
            }
            return json.loads(json.dumps(context_wire_value(summary), allow_nan=False))
        return wire
    except (TypeError, ValueError, RecursionError) as exc:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            f"cannot fully snapshot the active context: {exc}",
            reason="unserializable_context",
        ) from exc


def _md_summary(value: object) -> object:
    # Retain the former md_get leaf descriptions; native complex scalars stay values.
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool, int, float, complex)):
        return value
    if isinstance(value, np.ndarray):
        return f"{' × '.join(str(size) for size in value.shape)} array"
    if isinstance(value, (list, tuple)):
        if value and all(isinstance(row, (list, tuple)) for row in value):
            widths = {len(row) for row in value}
            if len(widths) == 1:
                return f"{len(value)} × {widths.pop()} matrix"
        return f"{len(value)} items"
    if isinstance(value, dict):
        return f"{len(value)} keys"
    return type(value).__name__


def _snapshot_summary(value: object) -> object:
    if isinstance(value, dict):
        if "__complex__" in value:
            return value
        return {key: _snapshot_summary(item) for key, item in value.items()}
    return _md_summary(value)


def _snapshot_path_value(snapshot: Mapping[str, object], path: str) -> object:
    value: object = snapshot
    for segment in path.split("."):
        if segment and isinstance(value, Mapping) and segment in value:
            value = value[segment]
        elif (
            isinstance(value, list)
            and segment.isascii()
            and segment.isdecimal()
            and (segment == "0" or not segment.startswith("0"))
            and len(segment) <= len(str(len(value)))
            and int(segment) < len(value)
        ):
            value = value[int(segment)]
        else:
            raise InvalidInputError(
                f"Invalid snapshot path {path!r}: segment {segment!r} is not readable"
            )
    return value


def _value_info_to_wire(info: ValueInfo) -> dict[str, object]:
    return {
        "key": info.key,
        "type": info.type_name,
        "owner": info.owner,
        "description": info.description,
    }


def h_value_list(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    return {
        "values": [
            _value_info_to_wire(info)
            for info in adapter.context_control.list_value_sources()
        ]
    }


def h_value_read(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    key = str(params["key"])
    raw_type = params.get("type")
    if raw_type is not None and not isinstance(raw_type, str):
        raise RemoteError(ErrorCode.INVALID_PARAMS, "'type' must be a string")
    type_name = raw_type
    try:
        info, value = adapter.context_control.read_value_source(key, type_name)
    except ValueError as exc:
        raise RemoteError(ErrorCode.INVALID_PARAMS, str(exc)) from exc
    return {**_value_info_to_wire(info), "value": value}


def h_context_ml_list_roles(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    """Project template metadata through the unchanged context.ml_list_roles RPC."""
    del params
    return {"roles": adapter.ctrl.list_templates()}


def h_context_ml_create_from_role(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    """Create a ModuleLibrary entry from the template ID in wire role_id.

    Optional edits use context.ml_edit's path/value Form input shape and apply
    before validating the captured template. Only a valid Spec is written.
    Invalid paths, values or Form results return invalid_params with
    data.errors containing path/message pairs; nothing is written.
    """
    from ..form_observation import form_writes

    raw_edits = params.get("edits")
    edits = form_writes(raw_edits) if raw_edits is not None else None
    template_id = str(params["role_id"])
    name = str(params["name"])
    # The item kind is a property of the template, not an independent agent input —
    # derive it from role_id so the agent cannot pass a mismatching pair. An
    # unknown role_id fails fast as invalid_params.
    try:
        item_kind = adapter.ctrl.template_kind(template_id)
    except KeyError as exc:
        raise RemoteError(ErrorCode.INVALID_PARAMS, str(exc)) from exc
    try:
        adapter.ctrl.create_from_template(item_kind, template_id, name, edits=edits)
    except LibraryEntryInvalid as exc:
        raise RemoteError(
            ErrorCode.INVALID_PARAMS,
            str(exc),
            reason="form_errors",
            data={
                "errors": [
                    {"path": list(path), "message": message}
                    for path, message in exc.errors.messages.items()
                ]
            },
        ) from exc
    except KeyError as exc:
        raise RemoteError(ErrorCode.INVALID_PARAMS, str(exc)) from exc
    return {"created": name}


def h_context_ml_del_module(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    """Delete concrete module params[name]; echo deleted, propagate owner errors."""
    name = str(params["name"])
    parameters = adapter.context_control.read_context()
    adapter.ctrl.apply_library_change(
        parameters, LibraryChange(kind="module", action="delete", path=name)
    )
    return {"deleted": name}


def h_context_ml_rename_module(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    """Rename concrete module params[old/new]; echo renamed, propagate owner errors."""
    old = str(params["old"])
    new = str(params["new"])
    parameters = adapter.context_control.read_context()
    adapter.ctrl.apply_library_change(
        parameters,
        LibraryChange(kind="module", action="rename", path=old, new_path=new),
    )
    return {"renamed": new}


def h_context_ml_rename_waveform(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    """Rename concrete waveform params[old/new]; echo renamed, propagate owner errors."""
    old = str(params["old"])
    new = str(params["new"])
    parameters = adapter.context_control.read_context()
    adapter.ctrl.apply_library_change(
        parameters,
        LibraryChange(kind="waveform", action="rename", path=old, new_path=new),
    )
    return {"renamed": new}


def h_context_ml_del_waveform(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    """Delete concrete waveform params[name]; echo deleted, propagate owner errors."""
    name = str(params["name"])
    parameters = adapter.context_control.read_context()
    adapter.ctrl.apply_library_change(
        parameters, LibraryChange(kind="waveform", action="delete", path=name)
    )
    return {"deleted": name}
