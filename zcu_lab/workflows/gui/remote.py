"""App-owned wire projection onto the existing T1 Window controls."""

from __future__ import annotations

from base64 import b64encode
from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal, NotRequired, TypedDict

from pydantic import JsonValue
from zcu_tools.experiment.workflows import (
    Actor,
    CommittedRecord,
    InvalidRunState,
    RunMismatch,
    RunStatus,
)
from zcu_tools.gui.app.workflows.samples import SampleExportError, UnsupportedSamples
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.form import FormErrors, FormWritePath
from zcu_tools.gui.remote.control_service import (
    ControlOptions,
    RemoteControlServiceBase,
)
from zcu_tools.gui.remote.errors import ErrorCode, RemoteError
from zcu_tools.gui.remote.method_spec import BoundMethod
from zcu_tools.gui.session.ports import OwnerScheduler

from .remote_methods import GUI_VERSION, METHOD_SPECS, WIRE_VERSION

if TYPE_CHECKING:
    from .window import T1WorkflowWindow


class RunTarget(TypedDict):
    """run_id is the non-empty native identity supplied to Start, never a token."""

    run_id: str


class SampleExportReply(RunTarget):
    """Successful explicit append: written_count counts this call; path is history.

    run_id is the targeted workflow, not the per-row key. written_count includes
    zero only for a supported run without successful points; unsupported fails.
    """

    written_count: int
    path: str


class SampleFailureData(TypedDict):
    """Partial append diagnosis: successful prefix count and original I/O cause.

    written_count includes zero and excludes the failed row. cause is the
    original OSError text; cause_type is its concrete Python exception name.
    Existing rows remain; retry appends a complete batch without rollback.
    """

    written_count: int
    cause: str
    cause_type: str


class FluxdepOutputReply(TypedDict):
    """One current output attempt, not a measurement outcome.

    format names the native data_h5 or Labber writer. path is its allocated
    local destination as text, or null only on failed Labber allocation.
    status is saved/failed; error is null on success and a nonempty diagnostic
    on failure. Earlier calls' successful paths are not merged into this reply.
    """

    format: Literal["data_h5", "labber"]
    path: str | None
    status: Literal["saved", "failed"]
    error: str | None


class FluxdepExportReply(RunTarget):
    """Latest terminal export report for the explicitly targeted run_id.

    outputs lists native then Labber attempts, including allocation failures.
    ledger_error is null after saved-event append, otherwise its diagnostic;
    it does not erase successful file paths or change the workflow outcome.
    """

    outputs: list[FluxdepOutputReply]
    ledger_error: str | None


class RecordsParams(RunTarget):
    """Native records cursor: since is a nonnegative commit_seq, default 0."""

    since: int


class FigureParams(RunTarget):
    """name is an exact nonempty name published by workflow.figures."""

    name: str


class EncodedRecordReply(TypedDict):
    """Native encoding: json/repr mode, JSON/diagnostic value, optional failure."""

    mode: Literal["json", "repr"]
    value: JsonValue
    error: str | None


class RecordReply(TypedDict):
    """One native committed point, with no inferred values or artist inspection.

    commit_seq counts Next commits; call_seq counts invocations; revision is the
    invocation's captured tunables version. encoded_record is the native summary.
    iteration_dir is relative to data_root. Workflow file references belong to
    encoded_record, whose contents the framework does not interpret.
    """

    commit_seq: int
    call_seq: int
    revision: int
    encoded_record: EncodedRecordReply
    iteration_dir: str


class RecordsReply(RunTarget):
    """records contains native commits after since, in native commit order."""

    records: list[RecordReply]


class FiguresReply(RunTarget):
    """names lists the current Plots membership; [] means no figure registered."""

    names: list[str]


class FigureReply(FigureParams):
    """png_base64 contains the PNG of name, rendered at 640 x 480 on the owner."""

    png_base64: str


class FormWrite(TypedDict):
    """One JSON batch edit: nonempty string field path and JSON value.

    path uses model-field segments, not dotted keys, mapping keys or indexes.
    value may be an ordinary value, expression or complex tag. Native FormState
    owns branch ordering, whole-leaf replacement and field/model validation.
    """

    path: list[str]
    value: JsonValue


class StartParams(RunTarget):
    """Start identity and two required partial-input batches.

    plan/tunables contain FormWrite objects; [] uses current captured defaults.
    name is an optional native workflow name; omitted/null selects catalog's T1
    default, independently of the GUI selection. Each path occurs once. Batches
    never edit existing draft before success; selection uses the same app owner.
    """

    plan: list[FormWrite]
    tunables: list[FormWrite]
    name: NotRequired[str | None]


class TunablesParams(RunTarget):
    """External edit of one native run using its current complete parsed input.

    batch contains unique FormWrite paths. [] submits current input, not defaults.
    Actor is app-owned agent/rpc; caller supplies no revision or holder.
    """

    batch: list[FormWrite]


class FormErrorReply(TypedDict):
    """Existing FormErrors projection: relative field/index path and native message.

    Empty path means a model-wide error. No code, identity or other metadata
    is inferred from the native message.
    """

    path: list[str | int]
    message: str


class FormInspectionParams(TypedDict):
    """form selects plan/tunables; run_id targets an attached run or is null.

    batch is null for current draft read, or path/value edits for a non-submitting
    candidate preview. [] previews retained parsed input, not Start defaults.
    """

    form: Literal["plan", "tunables"]
    run_id: str | None
    batch: list[FormWrite] | None


class FormInspectionReply(TypedDict):
    """Same-owner form observation, never submission or a new editing state.

    schema describes accepted partial input in the fixed capture. input retains
    expressions and missing/null in the complete candidate tree. values lists
    successfully decoded ordinary leaves as path/value entries; complex uses
    the existing __complex__ tag, tuples become lists. errors contains native
    path/message entries and, on read only, renderer-local parse errors.
    revision is the attached Engine's tunables revision or null before Start;
    it is not a form version. Values do not establish validity or acceptance.
    """

    schema: dict[str, JsonValue]
    input: dict[str, JsonValue]
    values: list[FormWrite]
    errors: list[FormErrorReply]
    revision: int | None


class ProgressReply(TypedDict):
    """Native display snapshot: name is the bar; total may be null; completed is absolute."""

    name: str
    total: float | None
    completed: float


class NativeStatusReply(TypedDict):
    """JSON projection of RunStatus, with no inferred acceptance or completion.

    run_id/workflow identify the native run and declaration. lifecycle is running,
    pausing, paused, stopping, done, aborted, stopped or failed. call_seq is the
    last invocation or null; committed_seq counts successful Next commits.
    revision is the native numeric tunables version. reason is a native diagnostic
    or null. progress is the list of native display snapshots, not commit proof.
    """

    run_id: str
    workflow: str
    lifecycle: str
    call_seq: int | None
    committed_seq: int
    revision: int
    reason: str | None
    progress: list[ProgressReply]


class WorkflowReply(TypedDict):
    """Same-owner receipt using only existing Controller and Engine fields.

    run_id is the attached Start identity, or null before submission.
    operation_id is the current pending/executing segment token, or null.
    status is native RunStatus projected to JSON, or null before acceptance.
    A non-null operation_id never establishes worker acceptance or completion.
    """

    run_id: str | None
    operation_id: int | None
    status: NativeStatusReply | None


class T1RemoteControlAdapter(RemoteControlServiceBase):
    """RPC face of one T1 Window; it never owns a second run or capture.

    workflow.start submits partial path/value batches through Window.start;
    FormErrors becomes invalid_params/form_errors; data is an object whose native
    messages field contains the list of FormErrorReply path/message entries.
    Batch wire admission rejects malformed or duplicate paths without side effects.
    A successful reply is pending admission, not worker acceptance. GUI drafts are
    untouched on failure; success resets their bases through the existing owner.
    workflow.tunables requires run_id and a path/value batch. It submits complete
    current parsed draft through the same Window tuning panel with agent/rpc actor;
    [] never resets defaults. Success returns WorkflowReply with native revision.
    Invalid input stays in draft and projects FormErrors; accepted input/base and
    native values/revision are unchanged. Pending startup is not editable.
    workflow.form reads schema/input/ordinary values/errors and native revision
    through Window.inspect_form. An optional batch previews a call-local candidate
    without modifying input/base, native state or renderer-local text. An attached
    run requires explicit run_id; pending acceptance is not readable as accepted.
    workflow.status reads the same Controller on the owner. Pause/Resume/Stop
    require explicit run_id and invoke the existing Window control once.
    RunMismatch becomes precondition_failed/run_mismatch with expected/actual;
    workflow.records requires run_id and optional nonnegative integer since
    (default 0). It projects Engine.records, without re-encoding or copying its
    detached summary. workflow.figures lists names without rendering;
    workflow.figure takes run_id/name and returns base64 PNG at 640 x 480.
    Both use this Window's Plots on the GUI owner; terminal plots remain readable
    until replacement/close. Unknown name becomes invalid_params/unknown_figure;
    released plots become precondition_failed/no_plots. Pending native acceptance
    is controller_error, never an empty success. Renderer errors propagate through
    the shared controller_error envelope without retry. status never renders.
    InvalidRunState becomes precondition_failed/invalid_run_state with action/
    lifecycle. Unmarked pending RuntimeError and unexpected errors retain shared
    controller_error classification. Transport timeout/disconnect does not Stop
    a run and gives no retry guarantee. There are no business event projections,
    additional observation policies, holder checks or hidden pre-reads.
    """

    ctrl: T1WorkflowWindow

    def __init__(
        self,
        window: T1WorkflowWindow,
        opts: ControlOptions,
        *,
        bus: BaseEventBus,
        owner_scheduler: OwnerScheduler,
    ) -> None:
        """Bind one live Window, its session bus and its owner scheduler.

        Construct on the Qt owner after window exists. opts controls transport;
        constructor opens no socket. Caller starts/stops this adapter through
        inherited start()/stop(), stops it before destroying window, and must use
        the same session bus/owner as window. Stopping transport does not cancel
        native work. Registry drift and shared transport admission failures
        propagate without retry; no hardware is contacted by construction.
        """
        self._event_bus = bus
        super().__init__(
            window,
            opts,
            owner_scheduler=owner_scheduler,
            wire_version=WIRE_VERSION,
            gui_version=GUI_VERSION,
            server_name="T1WorkflowRemoteServer",
            method_registry={
                "workflow.start": BoundMethod(
                    self._start, METHOD_SPECS["workflow.start"]
                ),
                "workflow.tunables": BoundMethod(
                    self._tunables, METHOD_SPECS["workflow.tunables"]
                ),
                "workflow.form": BoundMethod(self._form, METHOD_SPECS["workflow.form"]),
                "workflow.status": BoundMethod(
                    self._status, METHOD_SPECS["workflow.status"]
                ),
                "workflow.records": BoundMethod(
                    self._records, METHOD_SPECS["workflow.records"]
                ),
                "workflow.figures": BoundMethod(
                    self._figures, METHOD_SPECS["workflow.figures"]
                ),
                "workflow.figure": BoundMethod(
                    self._figure, METHOD_SPECS["workflow.figure"]
                ),
                "workflow.pause": BoundMethod(
                    self._pause, METHOD_SPECS["workflow.pause"]
                ),
                "workflow.resume": BoundMethod(
                    self._resume, METHOD_SPECS["workflow.resume"]
                ),
                "workflow.stop": BoundMethod(self._stop, METHOD_SPECS["workflow.stop"]),
                "workflow.export_samples": BoundMethod(
                    self._export_samples, METHOD_SPECS["workflow.export_samples"]
                ),
                "workflow.export_fluxdep": BoundMethod(
                    self._export_fluxdep, METHOD_SPECS["workflow.export_fluxdep"]
                ),
            },
            event_serializers={},
            wire_event_name=str,
        )

    def _get_bus(self) -> BaseEventBus:
        return self._event_bus

    def _start(
        self,
        _adapter: RemoteControlServiceBase,
        params: StartParams,
    ) -> WorkflowReply:
        plan_batch = _parse_form_batch(params["plan"])
        tunables_batch = _parse_form_batch(params["tunables"])
        name = params.get("name")
        try:
            result = self.ctrl.start(
                params["run_id"],
                plan_batch=plan_batch,
                tunables_batch=tunables_batch,
                name=self.ctrl.default_workflow_name if name is None else name,
            )
        except InvalidRunState as exc:
            raise _project_control_error(exc) from exc
        if isinstance(result, FormErrors):
            messages: list[FormErrorReply] = [
                {"path": list(path), "message": message}
                for path, message in result.messages.items()
            ]
            raise RemoteError(
                ErrorCode.INVALID_PARAMS,
                "Invalid Start form input",
                reason="form_errors",
                data={"messages": messages},
            )
        return self._status(_adapter, {})

    def _tunables(
        self,
        _adapter: RemoteControlServiceBase,
        params: TunablesParams,
    ) -> WorkflowReply:
        batch = _parse_form_batch(params["batch"])
        try:
            result = self.ctrl.apply_tunables(
                params["run_id"],
                batch,
                actor=Actor("agent", "rpc"),
            )
        except (RunMismatch, InvalidRunState) as exc:
            raise _project_control_error(exc) from exc
        if isinstance(result, FormErrors):
            messages: list[FormErrorReply] = [
                {"path": list(path), "message": message}
                for path, message in result.messages.items()
            ]
            raise RemoteError(
                ErrorCode.INVALID_PARAMS,
                "Invalid tunables form input",
                reason="form_errors",
                data={"messages": messages},
            )
        return self._status(_adapter, {})

    def _form(
        self,
        _adapter: RemoteControlServiceBase,
        params: FormInspectionParams,
    ) -> FormInspectionReply:
        batch = params["batch"]
        writes = None if batch is None else _parse_form_batch(batch)
        try:
            result = self.ctrl.inspect_form(
                params["form"], run_id=params["run_id"], batch=writes
            )
        except (RunMismatch, InvalidRunState) as exc:
            raise _project_control_error(exc) from exc
        if isinstance(result, FormErrors):
            messages: list[FormErrorReply] = [
                {"path": list(path), "message": message}
                for path, message in result.messages.items()
            ]
            raise RemoteError(
                ErrorCode.INVALID_PARAMS,
                "Invalid form preview batch",
                reason="form_errors",
                data={"messages": messages},
            )
        return result

    def _status(
        self,
        _adapter: RemoteControlServiceBase,
        _params: Mapping[str, object],
    ) -> WorkflowReply:
        controller = self.ctrl.controller
        return {
            "run_id": controller.run_id,
            "operation_id": controller.operation_id,
            "status": _project_status(controller.status()),
        }

    def _records(
        self, _adapter: RemoteControlServiceBase, params: RecordsParams
    ) -> RecordsReply:
        if params["since"] < 0:
            raise RemoteError(ErrorCode.INVALID_PARAMS, "since must be nonnegative")
        controller = self.ctrl.controller
        try:
            controller.require_run_id(params["run_id"])
        except RunMismatch as exc:
            raise _project_control_error(exc) from exc
        engine = controller.engine
        if engine is None:
            raise RuntimeError("Records require an accepted run")
        return {
            "run_id": params["run_id"],
            "records": [
                _project_record(record)
                for record in engine.records(params["run_id"], since=params["since"])
            ],
        }

    def _figures(
        self, _adapter: RemoteControlServiceBase, params: RunTarget
    ) -> FiguresReply:
        try:
            names = self.ctrl.figure_names(params["run_id"])
        except RunMismatch as exc:
            raise _project_control_error(exc) from exc
        return {"run_id": params["run_id"], "names": list(names)}

    def _figure(
        self, _adapter: RemoteControlServiceBase, params: FigureParams
    ) -> FigureReply:
        try:
            png = self.ctrl.figure_png(params["run_id"], params["name"])
        except RunMismatch as exc:
            raise _project_control_error(exc) from exc
        except KeyError as exc:
            raise RemoteError(
                ErrorCode.INVALID_PARAMS, str(exc), reason="unknown_figure"
            ) from exc
        return {
            "run_id": params["run_id"],
            "name": params["name"],
            "png_base64": b64encode(png).decode("ascii"),
        }

    def _pause(
        self,
        _adapter: RemoteControlServiceBase,
        params: RunTarget,
    ) -> WorkflowReply:
        try:
            status = self.ctrl.pause(params["run_id"])
        except (RunMismatch, InvalidRunState) as exc:
            raise _project_control_error(exc) from exc
        controller = self.ctrl.controller
        return {
            "run_id": controller.run_id,
            "operation_id": controller.operation_id,
            "status": _project_status(status),
        }

    def _resume(
        self,
        _adapter: RemoteControlServiceBase,
        params: RunTarget,
    ) -> WorkflowReply:
        try:
            operation_id = self.ctrl.resume(params["run_id"])
        except (RunMismatch, InvalidRunState) as exc:
            raise _project_control_error(exc) from exc
        controller = self.ctrl.controller
        return {
            "run_id": controller.run_id,
            "operation_id": operation_id,
            "status": _project_status(controller.status()),
        }

    def _export_samples(
        self, _adapter: RemoteControlServiceBase, params: RunTarget
    ) -> SampleExportReply:
        try:
            report = self.ctrl.export_samples(params["run_id"])
        except (RunMismatch, InvalidRunState) as error:
            raise _project_control_error(error) from error
        except UnsupportedSamples as error:
            raise RemoteError(
                ErrorCode.PRECONDITION_FAILED, str(error), reason="unsupported"
            ) from error
        except SampleExportError as error:
            detail = SampleFailureData(
                written_count=error.written_count,
                cause=str(error.cause),
                cause_type=type(error.cause).__name__,
            )
            raise RemoteError(
                ErrorCode.CONTROLLER_ERROR,
                str(error),
                reason="sample_export_failed",
                data=dict(detail),
            ) from error
        return {
            "run_id": report.run_id,
            "written_count": report.written_count,
            "path": str(report.path),
        }

    def _export_fluxdep(
        self, _adapter: RemoteControlServiceBase, params: RunTarget
    ) -> FluxdepExportReply:
        try:
            report = self.ctrl.export_fluxdep(params["run_id"], origin="mcp")
        except (RunMismatch, InvalidRunState) as exc:
            raise _project_control_error(exc) from exc
        return {
            "run_id": report.run_id,
            "outputs": [
                {
                    "format": out.format,
                    "path": None if out.path is None else str(out.path),
                    "status": out.status,
                    "error": out.error,
                }
                for out in report.outputs
            ],
            "ledger_error": report.ledger_error,
        }

    def _stop(
        self,
        _adapter: RemoteControlServiceBase,
        params: RunTarget,
    ) -> WorkflowReply:
        try:
            status = self.ctrl.stop(params["run_id"])
        except (RunMismatch, InvalidRunState) as exc:
            raise _project_control_error(exc) from exc
        controller = self.ctrl.controller
        return {
            "run_id": controller.run_id,
            "operation_id": controller.operation_id,
            "status": _project_status(status),
        }


def _parse_form_batch(
    batch: JsonValue | list[FormWrite],
) -> dict[FormWritePath, JsonValue]:
    """Convert the JSON batch shape only; FormState owns values and field paths."""
    if not isinstance(batch, list):
        raise RemoteError(ErrorCode.INVALID_PARAMS, "Start batch must be an array")
    writes: dict[FormWritePath, JsonValue] = {}
    for edit in batch:
        if not isinstance(edit, dict) or set(edit) != {"path", "value"}:
            raise RemoteError(
                ErrorCode.INVALID_PARAMS, "Start edits require path and value"
            )
        path = edit["path"]
        if not isinstance(path, list) or not path:
            raise RemoteError(
                ErrorCode.INVALID_PARAMS, "Start paths must be nonempty arrays"
            )
        segments: list[str] = []
        for segment in path:
            if not isinstance(segment, str) or not segment:
                raise RemoteError(
                    ErrorCode.INVALID_PARAMS,
                    "Start path segments must be nonempty strings",
                )
            segments.append(segment)
        field_path = tuple(segments)
        if field_path in writes:
            raise RemoteError(ErrorCode.INVALID_PARAMS, "Duplicate Start path")
        writes[field_path] = edit["value"]
    return writes


def _project_record(record: CommittedRecord) -> RecordReply:
    """Project native fields only; keep the detached encoded value unchanged."""
    return {
        "commit_seq": record.commit_seq,
        "call_seq": record.call_seq,
        "revision": record.revision,
        "encoded_record": {
            "mode": record.encoded_record.mode,
            "value": record.encoded_record.value,
            "error": record.encoded_record.error,
        },
        "iteration_dir": record.iteration_dir,
    }


def _project_status(status: RunStatus | None) -> NativeStatusReply | None:
    """Project only native status fields; None preserves pending acceptance."""
    if status is None:
        return None
    return {
        "run_id": status.run_id,
        "workflow": status.workflow,
        "lifecycle": status.lifecycle,
        "call_seq": status.call_seq,
        "committed_seq": status.committed_seq,
        "revision": status.revision,
        "reason": status.reason,
        "progress": [
            {
                "name": progress.name,
                "total": progress.total,
                "completed": progress.completed,
            }
            for progress in status.progress
        ],
    }


def _project_control_error(exc: RunMismatch | InvalidRunState) -> RemoteError:
    """Translate native target/lifecycle failures without classifying other errors."""
    if isinstance(exc, RunMismatch):
        return RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            str(exc),
            reason="run_mismatch",
            data={"expected": exc.expected, "actual": exc.actual},
        )
    return RemoteError(
        ErrorCode.PRECONDITION_FAILED,
        str(exc),
        reason="invalid_run_state",
        data={"action": exc.action, "lifecycle": exc.lifecycle},
    )
