"""Workflow wire commands over the measure session's single headless owner."""

from __future__ import annotations

from base64 import b64encode
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from typing import TYPE_CHECKING, Literal

from pydantic import JsonValue, TypeAdapter, ValidationError

from zcu_tools.experiment.workflows import Actor, InvalidRunState, RunMismatch
from zcu_tools.gui.app.measure.services.workflow_control import (
    WorkflowControlFacet,
    WorkflowTabSnapshot,
)
from zcu_tools.gui.form import FormErrors, FormWritePath
from zcu_tools.gui.remote.errors import ErrorCode, RemoteError

from ..form_description import description_wire
from ..form_errors import recipe_form_errors
from ..form_observation import form_writes
from ..workflow_types import project_record, project_status
from ._wire_values import context_wire_value

if TYPE_CHECKING:
    from ..service import RemoteControlAdapter


@contextmanager
def _workflow_errors() -> Generator[None]:
    try:
        with recipe_form_errors():
            yield
    except RunMismatch as exc:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            str(exc),
            reason="run_mismatch",
            data={"expected": exc.expected, "actual": exc.actual},
        ) from exc
    except InvalidRunState as exc:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            str(exc),
            reason="invalid_run_state",
            data={"action": exc.action, "lifecycle": exc.lifecycle},
        ) from exc


def _tab(control: WorkflowControlFacet, tab_id: str) -> WorkflowTabSnapshot:
    for tab in control.tabs():
        if tab.tab_id == tab_id:
            return tab
    raise RemoteError(ErrorCode.INVALID_PARAMS, f"Unknown workflow tab: {tab_id}")


def _run(control: WorkflowControlFacet, run_id: str) -> WorkflowTabSnapshot:
    for tab in control.tabs():
        if tab.run_id == run_id:
            return tab
    raise RunMismatch(run_id, None)


def _snapshot(tab: WorkflowTabSnapshot) -> dict[str, object]:
    return {
        "tab_id": tab.tab_id,
        "name": tab.name,
        "run_id": tab.run_id,
        "operation_id": tab.operation_id,
        "status": project_status(tab.status),
        "busy": tab.busy,
        "error": tab.error,
        "pending_question": tab.pending_question,
        "last_exchange": None
        if tab.last_exchange is None
        else {
            "question": tab.last_exchange.question,
            "answer": tab.last_exchange.answer,
            "state": tab.last_exchange.state,
        },
    }


def _form(params: Mapping[str, object]) -> Literal["plan", "tunables"]:
    form = params["form"]
    if form not in ("plan", "tunables"):
        raise RemoteError(ErrorCode.INVALID_PARAMS, "form must be plan or tunables")
    return "plan" if form == "plan" else "tunables"


def _batch(
    params: Mapping[str, object], key: str = "batch"
) -> Mapping[FormWritePath, JsonValue] | None:
    value = params.get(key)
    return None if value is None else form_writes(value)


def _path(raw: object) -> FormWritePath:
    if not isinstance(raw, list) or any(
        isinstance(part, bool) or not isinstance(part, (str, int)) or part == ""
        for part in raw
    ):
        raise RemoteError(
            ErrorCode.INVALID_PARAMS, "path must be a typed array; [] is root"
        )
    return tuple(TypeAdapter(list[str | int]).validate_python(raw))


def _valid[T](result: T | FormErrors) -> T:
    if isinstance(result, FormErrors):
        raise RemoteError(
            ErrorCode.INVALID_PARAMS,
            "Invalid workflow form input",
            reason="form_errors",
            data={
                "messages": [
                    {"path": list(path), "message": message}
                    for path, message in result.messages.items()
                ]
            },
        )
    return result


def h_workflow_open(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    """Open params.name and return tab_id/name/tab; unknown names are invalid_params."""
    control = adapter.workflow_control
    name = str(params["name"])
    if name not in control.names():
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"Unknown workflow: {name}")
    tab_id = control.open(name)
    return {"tab_id": tab_id, "name": name, "tab": _snapshot(_tab(control, tab_id))}


def h_workflow_close(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    """Close params.tab_id if idle; return closed=False while the owner is busy."""
    tab_id = str(params["tab_id"])
    _tab(adapter.workflow_control, tab_id)
    return {"tab_id": tab_id, "closed": adapter.workflow_control.close(tab_id)}


def h_workflow_status(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    """Read params.run_id or all tabs/catalog; stale native targets keep run_mismatch."""
    control = adapter.workflow_control
    run_id = params.get("run_id")
    if run_id is None:
        return {
            "names": list(control.names()),
            "tabs": [_snapshot(tab) for tab in control.tabs()],
        }
    with _workflow_errors():
        return _snapshot(_run(control, str(run_id)))


def h_workflow_start(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    """Submit params.tab_id/run_id and optional plan/tunables batches; return pending operation_id."""
    control = adapter.workflow_control
    tab_id, run_id = str(params["tab_id"]), str(params["run_id"])
    _tab(control, tab_id)
    with _workflow_errors():
        operation = _valid(
            control.start(
                tab_id,
                run_id,
                plan_batch=_batch(params, "plan"),
                tunables_batch=_batch(params, "tunables"),
            )
        )
    return {"tab_id": tab_id, "run_id": run_id, "operation_id": operation}


def _observe(
    adapter: RemoteControlAdapter, params: Mapping[str, object], *, describe: bool
) -> dict[str, object]:
    tab_id = str(params["tab_id"])
    _tab(adapter.workflow_control, tab_id)
    with _workflow_errors():
        observation = adapter.workflow_control.observe_form(
            tab_id,
            _form(params),
            batch=_batch(params),
            path=_path(params["path"]) if describe else None,
        )
    preview = observation.preview
    result: dict[str, object] = {
        "tab_id": tab_id,
        "form": params["form"],
        "input": observation.input.data,
        "values": [
            {"path": list(path), "value": context_wire_value(value)}
            for path, value in preview.values.items()
        ],
        "errors": [
            {"path": list(path), "message": message}
            for path, message in (
                preview.errors.messages.items() if preview.errors else ()
            )
        ],
    }
    if observation.description is not None:
        result["description"] = description_wire(observation.description)
    return result


def h_workflow_form(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    """Read params.tab_id/form with optional preview batch, without committing input."""
    return _observe(adapter, params, describe=False)


def h_workflow_describe_form(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    """Observe params.tab_id/form/batch plus one layer at params.path, [] for root."""
    return _observe(adapter, params, describe=True)


def h_workflow_edit_form(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    """Write required params.batch to params.tab_id/form; return installed input/values/errors."""
    tab_id = str(params["tab_id"])
    _tab(adapter.workflow_control, tab_id)
    with _workflow_errors():
        adapter.workflow_control.write_form(
            tab_id, _form(params), form_writes(params["batch"])
        )
    # Read installed draft, not the same batch applied for a second time.
    return _observe(adapter, {"tab_id": tab_id, "form": params["form"]}, describe=False)


def h_workflow_tunables(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    """Apply required params.batch to params.run_id; return native revision/values or form_errors."""
    run_id = str(params["run_id"])
    with _workflow_errors():
        tab = _run(adapter.workflow_control, run_id)
        snapshot = _valid(
            adapter.workflow_control.apply_tunables(
                run_id, form_writes(params["batch"]), actor=Actor("agent", "rpc")
            )
        )
    return {
        "tab_id": tab.tab_id,
        "run_id": run_id,
        "revision": snapshot.revision,
        "values": snapshot.values,
    }


def h_workflow_pause(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    """Request cooperative Pause for params.run_id; project native state, not completion."""
    run_id = str(params["run_id"])
    with _workflow_errors():
        adapter.workflow_control.pause(run_id)
        return _snapshot(_run(adapter.workflow_control, run_id))


def h_workflow_resume(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    """Submit params.run_id continuation in the worker; return its original operation_id."""
    run_id = str(params["run_id"])
    with _workflow_errors():
        operation = adapter.workflow_control.resume(run_id)
        return {
            **_snapshot(_run(adapter.workflow_control, run_id)),
            "operation_id": operation,
        }


def h_workflow_stop(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    """Request Stop for params.run_id; lease release still awaits owner terminal delivery."""
    run_id = str(params["run_id"])
    with _workflow_errors():
        adapter.workflow_control.stop(run_id)
        return _snapshot(_run(adapter.workflow_control, run_id))


def h_workflow_records(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    """Read params.run_id commits strictly after nonnegative params.since, default zero."""
    run_id, since = str(params["run_id"]), int(str(params.get("since", 0)))
    if since < 0:
        raise RemoteError(ErrorCode.INVALID_PARAMS, "since must be nonnegative")
    with _workflow_errors():
        tab = _run(adapter.workflow_control, run_id)
        records = adapter.workflow_control.records(run_id, since=since)
    return {
        "tab_id": tab.tab_id,
        "run_id": run_id,
        "records": [project_record(record) for record in records],
    }


def h_workflow_figures(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    """List params.run_id retained Plots names without rendering, including terminal runs."""
    run_id = str(params["run_id"])
    with _workflow_errors():
        tab = _run(adapter.workflow_control, run_id)
        names = adapter.workflow_control.figures(run_id)
    return {"tab_id": tab.tab_id, "run_id": run_id, "names": list(names)}


def h_workflow_figure(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    """Render params.run_id/name to png_b64; unknown names are invalid_params/unknown_figure."""
    run_id, name = str(params["run_id"]), str(params["name"])
    with _workflow_errors():
        tab = _run(adapter.workflow_control, run_id)
        try:
            png = adapter.workflow_control.figure_png(run_id, name)
        except KeyError as exc:
            raise RemoteError(
                ErrorCode.INVALID_PARAMS,
                f"Unknown figure: {name}",
                reason="unknown_figure",
            ) from exc
    return {
        "tab_id": tab.tab_id,
        "run_id": run_id,
        "name": name,
        "png_b64": b64encode(png).decode("ascii"),
    }


def h_workflow_answer(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    """Submit params.run_id/answer on the retained operation; return its pending receipt.

    Engine/driver JSON validation errors become invalid_params/answer_validation.
    The owner, not this adapter, owns answer typing and lifecycle rules.
    """
    run_id = str(params["run_id"])
    if "answer" not in params:
        raise RemoteError(ErrorCode.INVALID_PARAMS, "answer is required")
    answer = TypeAdapter(JsonValue).validate_python(params["answer"])
    with _workflow_errors():
        tab = _run(adapter.workflow_control, run_id)
        try:
            operation = adapter.workflow_control.answer(run_id, answer)
        except (ValidationError, TypeError) as exc:
            raise RemoteError(
                ErrorCode.INVALID_PARAMS,
                str(exc),
                reason="answer_validation",
            ) from exc
    return {"tab_id": tab.tab_id, "run_id": run_id, "operation_id": operation}
