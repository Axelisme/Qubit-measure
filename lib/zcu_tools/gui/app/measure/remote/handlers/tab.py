"""Tab remote handlers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, TypedDict

from pydantic import JsonValue, TypeAdapter

from zcu_tools.format_version import YamlValue
from zcu_tools.gui.app.measure.services.form_contracts import (
    CfgPreconditionError,
    CfgPreconditionReason,
)
from zcu_tools.gui.form import FormCommandError, FormInputRejected, FormPathError
from zcu_tools.gui.form_adapter import LogicalEdit
from zcu_tools.gui.remote.errors import ErrorCode, RemoteError

if TYPE_CHECKING:
    from ..service import RemoteControlAdapter

from ..save_types import project_raw_data_state, project_save_snapshot
from ._common import follow_tab, render_view


def h_tab_new(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    name = str(params["adapter_name"])
    if name not in adapter.ctrl.get_adapter_names():
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown adapter: {name!r}")
    tab_id = adapter.tab_control.new_tab(name)
    return {"tab_id": tab_id}


def h_tab_open_file(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    from ..load_types import project_load_outcome

    name = str(params["adapter_name"])
    if name not in adapter.ctrl.get_adapter_names():
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown adapter: {name!r}")
    return project_load_outcome(
        adapter.tab_control.open_tab_from_file(name, str(params["data_path"]))
    )


def h_tab_close(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    tab_id = str(params["tab_id"])
    if not adapter.tab_control.has_tab(tab_id):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
    from .lifecycle import require_idle, require_saved

    require_idle(adapter, tab_id)
    if not params["discard_unsaved"]:
        require_saved(adapter, [tab_id])
    adapter.tab_control.close_tab(tab_id)
    return {"ok": True}


def h_tab_set_active(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    tab_id = str(params["tab_id"])
    if not adapter.tab_control.has_tab(tab_id):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
    adapter.tab_control.set_active_tab(tab_id)
    return {"ok": True}


def h_tab_list_all(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    running_tab_id = adapter.tab_control.get_running_tab_id()
    tabs = [
        {
            "tab_id": tid,
            "adapter_name": adapter.tab_control.get_tab_adapter_name(tid),
            "is_running": tid == running_tab_id,
        }
        for tid in adapter.tab_control.list_tab_ids()
    ]
    # active_tab_id is a view projection (which tab the user is focused on),
    # sourced from the RenderView snapshot, separate from the status tool.
    active_tab_id = render_view(adapter).get_view_snapshot().get("active_tab_id")
    return {
        "tabs": tabs,
        "active_tab_id": active_tab_id,
        "running_tab_id": running_tab_id,
    }


def tab_operation_state(
    adapter: RemoteControlAdapter, tab_id: str
) -> dict[str, object]:
    snap = adapter.tab_control.get_tab_snapshot(tab_id)
    interaction = snap.interaction
    # Render snapshot always fills the live fields (persist/restore form is the
    # only one that leaves them None, and it never hits the wire).
    assert interaction is not None
    assert snap.run is not None
    assert snap.analysis is not None
    assert snap.post_analysis is not None
    versions = adapter.ctrl.resources_versions()
    return {
        "tab_id": tab_id,
        "adapter_name": adapter.tab_control.get_tab_adapter_name(tab_id),
        "interaction": {
            "global_run_active": bool(interaction.global_run_active),
            "is_running": bool(interaction.is_running),
            "is_analyzing": bool(interaction.is_analyzing),
            "has_context": bool(interaction.has_context),
            "has_active_context": bool(interaction.has_active_context),
            "has_soc": bool(interaction.has_soc),
            "has_run_result": bool(interaction.has_run_result),
            "has_analyze_result": bool(interaction.has_analyze_result),
            "has_post_analyze_result": bool(interaction.has_post_analyze_result),
            "has_figure": bool(interaction.has_figure),
        },
        **project_save_snapshot(snap.artifacts, run_id=snap.run.run_id).model_dump(
            mode="json"
        ),
        "raw_data": project_raw_data_state(
            snap.artifacts, run_id=snap.run.acquisition_run_id
        ).model_dump(mode="json"),
        "result_source_path": snap.run.source_path,
        # Revisions distinguish replacements even when availability and source
        # path stay unchanged. Payload arrays remain with the application owner.
        "result_state": {
            "revision": versions.get(f"tab:{tab_id}:result", 0),
            "available": snap.run.result is not None,
            "source_path": snap.run.source_path,
            "source_operation_id": snap.run.source_operation_id,
        },
        "analysis_state": {
            "revision": versions.get(f"tab:{tab_id}:analyze", 0),
            "available": snap.analysis.result is not None,
            "has_figure": bool(snap.analysis.figures),
            "figure_names": list(snap.analysis.figures or ()),
            "has_writeback_draft": snap.analysis.has_writeback_draft,
        },
        "post_analysis_state": {
            "revision": versions.get(f"tab:{tab_id}:post_analyze", 0),
            "available": snap.post_analysis.result is not None,
            "has_figure": bool(snap.post_analysis.figures),
            "figure_names": list(snap.post_analysis.figures or ()),
            "has_writeback_draft": snap.post_analysis.has_writeback_draft,
        },
    }


def h_tab_snapshot(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    # Always returns {tabs: [...]} (a single tab_id yields a one-element list);
    # no shape-switch, so callers index reply["tabs"] uniformly.
    tab_id_raw = params.get("tab_id")
    if tab_id_raw is None:
        tab_ids = adapter.tab_control.list_tab_ids()
    else:
        tab_id = str(tab_id_raw)
        if not adapter.tab_control.has_tab(tab_id):
            raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
        tab_ids = [tab_id]
    return {"tabs": [tab_operation_state(adapter, tid) for tid in tab_ids]}


class TabParameterRead(TypedDict):
    """Requested values and module references from one current source capture.

    values maps requested logical paths to YAML values. Absent leaves are
    omitted; explicit null is retained. modules maps each requested reference
    to its existing canonical library root, or None for custom/missing roots.
    Both use the tab's fixed Open/Reset parameter/library/alias capture.
    Neither mapping refreshes the live context or operation guards.
    """

    values: dict[str, YamlValue]
    modules: dict[str, str | None]


def _parameter_query_strings(params: Mapping[str, object], name: str) -> list[str]:
    """Validate a requested parameter path/reference array at the wire boundary."""
    items = params.get(name)
    if not isinstance(items, list):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"{name} must be a string array")
    strings: list[str] = []
    for item in items:
        if not isinstance(item, str) or not item.strip():
            raise RemoteError(
                ErrorCode.INVALID_PARAMS, f"{name} must contain non-empty strings"
            )
        strings.append(item)
    return strings


def h_tab_parameter_read(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> TabParameterRead:
    """Read paths and module_refs through fixed sources for the named tab.

    tab_id is an existing GUI tab. paths and module_refs are arrays of non-empty
    strings, with $alias.path distinct from concrete component paths. Reset may
    capture single-use aliases without changing globals; reads never recapture.
    Query only these requested data.
    Missing leaves and explicit null follow TabParameterRead. Module lookup uses
    the parameter owner's role/slot resolution and library membership. Unknown
    roles, malformed paths, unavailable sources and owner I/O errors propagate.
    Without current parameter sources, only empty arrays are accepted; otherwise
    precondition_failed.
    This read does not refresh any live-context or operation observation guard.
    """
    tab_id = params.get("tab_id")
    if not isinstance(tab_id, str) or not adapter.tab_control.has_tab(tab_id):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
    paths = _parameter_query_strings(params, "paths")
    module_refs = _parameter_query_strings(params, "module_refs")
    parameters = adapter.cfg_lookup(tab_id).env.parameters
    if parameters is None:
        if paths or module_refs:
            raise RemoteError(
                ErrorCode.PRECONDITION_FAILED, "Parameter sources are unavailable"
            )
        return {"values": {}, "modules": {}}
    values: dict[str, YamlValue] = {}
    for path in paths:
        result = parameters.lookup(path)
        if result.present:
            values[path] = result.value
    modules: dict[str, str | None] = {}
    module_keys = parameters.library.module_keys() if module_refs else ()
    for reference in module_refs:
        try:
            root = parameters.library_key(reference)
        except AttributeError:
            # library_key reports only unset/dangling captured module slots here.
            modules[reference] = None
        else:
            modules[reference] = root if root in module_keys else None
    return {"values": values, "modules": modules}


def temporary_aliases(params: Mapping[str, object]) -> dict[str, str] | None:
    """Decode optional named component overrides; None and {} mean no override.

    Reject malformed objects or empty/non-string keys and values with
    invalid_aliases. Identifier and source errors belong to the context owner.
    """
    raw = params.get("aliases")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise RemoteError(
            ErrorCode.INVALID_PARAMS,
            "aliases must be an object",
            reason="invalid_aliases",
        )
    aliases: dict[str, str] = {}
    for alias, component in raw.items():
        if (
            not isinstance(alias, str)
            or not alias.strip()
            or not isinstance(component, str)
            or not component.strip()
        ):
            raise RemoteError(
                ErrorCode.INVALID_PARAMS,
                "aliases must map non-empty names to components",
                reason="invalid_aliases",
            )
        aliases[alias] = component
    return aliases or None


def h_tab_get_cfg(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    """Read current Form input/ordinary leaves/errors from fixed sources."""
    from ..form_observation import observe_form

    return dict(observe_form(adapter.cfg_lookup(str(params["tab_id"]))))


def h_tab_describe_cfg(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    """Read one current layer and local observation in the same owner turn.

    params supplies tab_id and path (field-name/index array, [] for root). Malformed
    arrays become invalid_params/form_errors. Core FormPathError becomes
    invalid_params/form_path with data.form_path preserving the requested path.
    Other declaration failures retain the controller_error boundary; resource
    lifetime errors keep their cfg projection. Reads allow busy and observer
    clients. Local errors are Form preview only; use get_cfg or Run admission
    for lowered cfg errors.
    """
    from ..form_description import describe_form
    from ..form_errors import cfg_error_to_remote

    raw = params["path"]
    if not isinstance(raw, list):
        raise RemoteError(
            ErrorCode.INVALID_PARAMS, "path must be an array", reason="form_errors"
        )
    segments: list[str | int] = []
    for part in raw:
        if not isinstance(part, (str, int)) or (isinstance(part, str) and not part):
            raise RemoteError(
                ErrorCode.INVALID_PARAMS,
                "path must contain field names or integer indexes",
                reason="form_errors",
            )
        segments.append(part)
    path = tuple(segments)
    try:
        return dict(describe_form(adapter.cfg_lookup(str(params["tab_id"])), path))
    except CfgPreconditionError as exc:
        raise cfg_error_to_remote(exc) from exc
    except FormPathError as exc:
        raise RemoteError(
            ErrorCode.INVALID_PARAMS,
            str(exc),
            reason="form_path",
            data={"form_path": list(exc.path)},
        ) from exc


def h_tab_refresh_sources(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    """Recapture tab sources on the owner thread, preserving all input.

    params requires tab_id and accepts single-capture aliases overrides. The
    adapter retains resource ownership. Admission failures become remote cfg
    errors; unexpected capture/factory failures propagate to dispatch. Return
    the new input/values/errors projection and follow the Run pane on success.
    """
    from ..form_errors import cfg_error_to_remote
    from ..form_observation import observe_form

    tab_id = str(params["tab_id"])
    try:
        resource = adapter.cfg_lookup(tab_id)
        resource.refresh_sources(aliases=temporary_aliases(params))
    except CfgPreconditionError as exc:
        raise cfg_error_to_remote(exc) from exc
    follow_tab(adapter, tab_id, "run")
    return dict(observe_form(resource))


def h_tab_reset_cfg(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    from ..form_errors import cfg_error_to_remote
    from ..form_observation import observe_form

    tab_id = str(params["tab_id"])
    try:
        resource = adapter.cfg_lookup(tab_id)
        resource.reset(aliases=temporary_aliases(params))
    except CfgPreconditionError as exc:
        raise cfg_error_to_remote(exc) from exc
    follow_tab(adapter, tab_id, "run")
    return dict(observe_form(resource))


def h_tab_run_cfg_command(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    """Dispatch a named active Form action through the same mutation host as edits.

    params contains tab_id, canonical path array and name from description.
    Unknown/inactive paths and unavailable commands reject before installation.
    Success returns current input/values/errors. No remote conversion runs.
    """
    from ..form_errors import cfg_error_to_remote
    from ..form_observation import observe_form

    raw = params["path"]
    if (
        not isinstance(raw, list)
        or not raw
        or any(not isinstance(part, str) or not part for part in raw)
    ):
        raise RemoteError(
            ErrorCode.INVALID_PARAMS,
            "path must contain nonempty strings",
            reason="form_errors",
        )
    path = tuple(str(part) for part in raw)
    tab_id = str(params["tab_id"])
    try:
        resource = adapter.cfg_lookup(tab_id)
        resource.run_command(path, str(params["name"]))
    except FormInputRejected as exc:
        from ..form_errors import input_rejected_to_remote

        raise input_rejected_to_remote(exc) from exc
    except CfgPreconditionError as exc:
        raise cfg_error_to_remote(exc) from exc
    except FormPathError as exc:
        raise RemoteError(
            ErrorCode.INVALID_PARAMS,
            str(exc),
            reason="form_path",
            data={"form_path": list(exc.path)},
        ) from exc
    except FormCommandError as exc:
        raise RemoteError(
            ErrorCode.INVALID_PARAMS, str(exc), reason="form_errors"
        ) from exc
    follow_tab(adapter, tab_id, "run")
    return dict(observe_form(resource))


def h_tab_edit_cfg(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    from ..form_errors import cfg_error_to_remote
    from ..form_observation import edit_path_hints, form_writes, observe_form

    tab_id = str(params["tab_id"])
    batch = form_writes(params["edits"])
    try:
        resource = adapter.cfg_lookup(tab_id)
    except CfgPreconditionError as exc:
        raise cfg_error_to_remote(exc) from exc
    try:
        resource.write(batch)
    except FormInputRejected as exc:
        from ..form_errors import input_rejected_to_remote

        raise input_rejected_to_remote(exc) from exc
    except CfgPreconditionError as exc:
        raise cfg_error_to_remote(exc) from exc
    except ValueError as exc:
        hints = edit_path_hints(resource)
        raise RemoteError(
            ErrorCode.INVALID_PARAMS,
            f"{exc}. Active editable paths: {hints['editable_paths']!r}. "
            f"Selection paths: {hints['selection_paths']!r}. "
            "Use tab.select_recipe_module for module selection with logical paths; "
            "ref descendants require use_inline first. "
            "Use tab.describe_cfg to inspect canonical options and paths.",
            reason="form_errors",
            data=dict(hints),
        ) from exc
    follow_tab(adapter, tab_id, "run")
    return dict(observe_form(resource))


def module_path(raw: object) -> tuple[str | int, ...]:
    """Decode canonical typed fields/indexes without dotted or name fallbacks."""
    if (
        not isinstance(raw, list)
        or not raw
        or any(
            (not isinstance(part, str) or not part or part != part.strip())
            and type(part) is not int
            for part in raw
        )
    ):
        raise RemoteError(
            ErrorCode.INVALID_PARAMS, "field must be a nonempty array of names/indexes"
        )
    return tuple(raw)


def h_tab_read_module(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    """Read detached field-local module values, without source refresh or mutation."""
    from ..form_errors import cfg_error_to_remote

    path = module_path(params["field"])
    try:
        module = adapter.cfg_lookup(str(params["tab_id"])).read_module(path)
    except FormPathError as exc:
        raise RemoteError(
            ErrorCode.INVALID_PARAMS,
            str(exc),
            reason="form_path",
            data={"form_path": list(exc.path)},
        ) from exc
    except CfgPreconditionError as exc:
        raise cfg_error_to_remote(exc) from exc
    except ValueError as exc:
        raise RemoteError(
            ErrorCode.INVALID_PARAMS, str(exc), reason="form_errors"
        ) from exc
    return {"value": module.value, "mode": module.mode, "source": module.source}


def h_tab_use_inline(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    """Convert one module using the fixed capture and return its complete Form view.

    params supplies tab_id and a typed field/index array for a core module field. Resource admission
    runs before reading the public input tag at an exact core module path.
    Ref invokes to_inline through the guarded command host; inline is a no-op
    even for partial input or invalid expressions. Mode detection does not
    evaluate the draft or read sources. Disabled, missing or untagged input and
    unavailable ref commands reject without fallback.
    Busy, revoked and notification-reentrant calls also reject inline no-ops.
    Known form failures become directed RemoteError replies; unexpected failures
    propagate. No source refresh, caller layout construction or retry occurs.
    """
    from ..form_errors import cfg_error_to_remote
    from ..form_observation import observe_form

    path = module_path(params["field"])
    try:
        resource = adapter.cfg_lookup(str(params["tab_id"]))
        resource.use_inline(path)
    except FormInputRejected as exc:
        from ..form_errors import input_rejected_to_remote

        raise input_rejected_to_remote(exc) from exc
    except FormPathError as exc:
        raise RemoteError(
            ErrorCode.INVALID_PARAMS,
            str(exc),
            reason="form_path",
            data={"form_path": list(exc.path)},
        ) from exc
    except CfgPreconditionError as exc:
        if exc.reason is CfgPreconditionReason.NOT_VALID:
            raise cfg_error_to_remote(
                CfgPreconditionError(
                    exc.reason,
                    f"Cannot convert {path!r}: {exc}. "
                    "Select an Inline option to edit this field.",
                )
            ) from exc
        raise cfg_error_to_remote(exc) from exc
    except ValueError as exc:
        raise RemoteError(
            ErrorCode.INVALID_PARAMS,
            f"Cannot convert {path!r}: {exc}. "
            "Select an Inline option to edit this field.",
            reason="form_errors",
        ) from exc
    return dict(observe_form(resource))


def h_tab_read_elements(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    """Forward one readonly aggregate to its owner; never focus or refresh."""
    from ..form_errors import recipe_form_errors
    from ..form_observation import observe_elements

    path = module_path(params["path"])
    with recipe_form_errors():
        raw_fields = params.get("fields", [])
        if not isinstance(raw_fields, list):
            raise RemoteError(
                ErrorCode.INVALID_PARAMS,
                "fields must be an array",
                reason="form_errors",
            )
        fields = tuple(_recipe_path(field) for field in raw_fields)
        return dict(
            observe_elements(
                adapter.cfg_lookup(str(params["tab_id"])), path, fields=fields
            )
        )


def _recipe_path(raw: object, *, root: bool = False) -> tuple[str | int, ...]:
    if not isinstance(raw, list) or (not raw and not root):
        raise RemoteError(
            ErrorCode.INVALID_PARAMS, "path must be a field array", reason="form_errors"
        )
    parts: list[str | int] = []
    for part in raw:
        if (not isinstance(part, str) or not part) and type(part) is not int:
            raise RemoteError(
                ErrorCode.INVALID_PARAMS,
                "path requires nonempty names or integer indexes",
                reason="form_errors",
            )
        parts.append(part)
    return tuple(parts)


def h_tab_read_recipe_field(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    """Read logical metadata/authored input/value/errors in one fixed-source turn.

    path contains str field names and int original list indexes, [] for root. Readonly refs do not materialize drafts
    or grant write authority. Expected failures keep directed logical details.
    """
    from ..form_description import recipe_field_wire
    from ..form_errors import recipe_form_errors

    path = _recipe_path(params["path"], root=True)
    with recipe_form_errors():
        resource = adapter.cfg_lookup(str(params["tab_id"]))
        return dict(recipe_field_wire(resource.read_recipe_field(path)))


def _recipe_edits(raw: object) -> list[LogicalEdit]:
    if not isinstance(raw, list):
        raise RemoteError(
            ErrorCode.INVALID_PARAMS, "edits must be an array", reason="form_errors"
        )
    edits: list[LogicalEdit] = []
    for edit in raw:
        if not isinstance(edit, dict) or set(edit) != {"path", "value"}:
            raise RemoteError(
                ErrorCode.INVALID_PARAMS,
                "edits require path/value",
                reason="form_errors",
            )
        path = _recipe_path(edit["path"])
        try:
            value = TypeAdapter(JsonValue).validate_python(edit["value"], strict=True)
        except ValueError as exc:
            raise RemoteError(
                ErrorCode.INVALID_PARAMS, str(exc), reason="form_errors"
            ) from exc
        edits.append(LogicalEdit(path, value))
    return edits


def h_tab_edit_recipe_fields(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    """Commit one ordered logical batch; invalid ordinary drafts remain installed.

    Structural/source/ref/guard failures reject without a successful prefix.
    Returns changed and complete logical Form preview errors, not cfg validity.
    """
    from ..form_description import RecipeEditResult
    from ..form_errors import recipe_form_errors

    edits = _recipe_edits(params["edits"])
    tab_id = str(params["tab_id"])
    with recipe_form_errors():
        resource = adapter.cfg_lookup(tab_id)
        changed = resource.edit_recipe_fields(edits)
        errors = resource.recipe_errors()
        result: RecipeEditResult = {
            "changed": changed,
            "errors": [
                {"path": list(path), "message": message}
                for path, message in (errors.messages.items() if errors else ())
            ],
        }
    if changed:
        follow_tab(adapter, tab_id, "run")
    return dict(result)


def h_tab_select_recipe_module(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    """Select an actual key, explicitly copy, or disable a logical selector.

    use_library requires a nonblank key; other actions must omit it. Explicit
    null is not omission. Owner admission precedes no-op detection; the reply
    includes a fresh same-turn observation, never a second RPC or cached read.
    """
    from ..form_description import RecipeSelectionResult, recipe_field_wire
    from ..form_errors import recipe_form_errors

    path = _recipe_path(params["path"])
    action = params["action"]
    if action not in ("use_library", "use_inline", "disable"):
        raise RemoteError(
            ErrorCode.INVALID_PARAMS, "Unknown selector action", reason="form_errors"
        )
    key = params.get("key")
    if action == "use_library":
        if not isinstance(key, str) or not key.strip():
            raise RemoteError(
                ErrorCode.INVALID_PARAMS,
                "use_library requires a key",
                reason="form_errors",
            )
    elif "key" in params:
        raise RemoteError(
            ErrorCode.INVALID_PARAMS, "key must be omitted", reason="form_errors"
        )
    tab_id = str(params["tab_id"])
    with recipe_form_errors():
        resource = adapter.cfg_lookup(tab_id)
        if action == "use_library" and isinstance(key, str):
            changed = resource.select_recipe_module(path, "use_library", key=key)
        elif action == "use_inline":
            changed = resource.select_recipe_module(path, "use_inline")
        else:
            changed = resource.select_recipe_module(path, "disable")
        result: RecipeSelectionResult = {
            "changed": changed,
            "field": recipe_field_wire(resource.read_recipe_field(path)),
        }
    if changed:
        follow_tab(adapter, tab_id, "run")
    return dict(result)
