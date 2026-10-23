"""Fixed GUI binding and operation lifetime behind the async recipe author API."""

from __future__ import annotations

import base64
import time
from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, replace
from threading import RLock
from typing import Literal

from pydantic import BaseModel, TypeAdapter

from zcu_tools.experiment import ScriptCancelled
from zcu_tools.experiment.scope import Scope, parse_cancel_scope
from zcu_tools.experiment.script_host import CallContext, CallResult, ScriptHost
from zcu_tools.experiment.stop_signal import (
    CancelResult,
)
from zcu_tools.format_version import YamlValue
from zcu_tools.gui.app.measure.remote.analysis_types import AnalysisPresentation
from zcu_tools.gui.app.measure.remote.form_description import RecipeEditResult
from zcu_tools.mcp.core.images import validated_png
from zcu_tools.mcp.core.reply import PngImage
from zcu_tools.mcp.measure.analysis_execution import (
    AnalysisReadPolicy,
    AnalysisRecovery,
    ExecutionPhase,
    InvalidAnalysisValue,
)
from zcu_tools.mcp.measure.raw_save import RawDataState, SaveReport
from zcu_tools.mcp.measure.recipe_api import (
    AnalysisResult,
    FieldEdit,
    FormError,
    FormField,
    FormPath,
    FormSelector,
    FormSource,
    FormValue,
    RecipeRun,
    RecipeTab,
    RunResult,
    SaveResult,
    WritebackDestination,
    WritebackItem,
    WritebackProposal,
    WritebackReceipt,
)
from zcu_tools.mcp.measure.recipe_contracts import (
    MissingParameter,
    RecipeNeedsParameters,
)
from zcu_tools.mcp.measure.recipe_details import RecipeAnalysisDetail, RecipeRunDetail
from zcu_tools.mcp.measure.recipe_form import (
    RecipeFormError,
    field_observation,
    field_path,
    read_form_field,
)
from zcu_tools.mcp.measure.session import GuiRpcError
from zcu_tools.mcp.measure.tool_context import MeasureToolContext
from zcu_tools.mcp.measure.writeback_recovery import (
    CHECK_WRITEBACK_STATUS,
    begin_writeback,
    writeback_recovery,
)


def _writeback_failure_receipt(
    error: GuiRpcError, identity: str | None
) -> WritebackReceipt:
    """Keep a known submission queryable when a timeout hides the owner outcome."""
    uncertain = identity is not None and error.code == "timeout"
    return WritebackReceipt(
        "unknown" if uncertain else "failed",
        (),
        f"{error}. {CHECK_WRITEBACK_STATUS}" if uncertain else str(error),
        identity,
        writeback_recovery(identity) if identity is not None else None,
    )


@dataclass(frozen=True)
class ScopeProjection:
    """Recipe/GUI path observation, not a cross-process atomic snapshot.

    active_path is the single chain from the root through the first fork.
    live_paths lists every live leaf's full composite path, including the root
    when no descendants live. GUI run roots are omitted from composite paths.
    """

    active_path: str
    live_paths: list[str]


class _ScopeReply(BaseModel):
    active_path: str
    live_paths: list[str]


class _ResultState(BaseModel):
    available: bool
    source_operation_id: str | int | None
    partial_result: bool


class _Interaction(BaseModel):
    is_running: bool
    is_analyzing: bool


class _AppliedPane(BaseModel):
    subtab_id: Literal["analysis", "post_analysis"]
    ids: tuple[str, ...]


class _Acceptance(BaseModel):
    status: Literal["completed", "failed"]
    written: tuple[_AppliedPane, ...]
    error: str | None


class _TabSnapshot(BaseModel):
    tab_id: str
    adapter_name: str
    interaction: _Interaction
    raw_data: RawDataState
    result_state: _ResultState


class _Snapshots(BaseModel):
    tabs: list[_TabSnapshot]


class _Started(AnalysisReadPolicy):
    handle: int
    interactive: bool = False


class _OperationError(BaseModel):
    reason: str
    message: str


class _Completion(BaseModel):
    reason: Literal["completed", "timeout"]
    status: Literal["finished", "cancelled", "failed"] | None = None
    error: str | _OperationError | None = None
    failure_kind: Literal["analysis"] | None = None

    @property
    def error_message(self) -> str:
        if isinstance(self.error, _OperationError):
            return self.error.message
        return self.error or "Operation failed"


class _Candidate(BaseModel):
    id: str
    target_name: str
    actual_path: str
    proposed: YamlValue


class _DestinationContext(BaseModel):
    active_label: str


class _Preview(BaseModel):
    has_draft: bool
    items: tuple[_Candidate, ...]
    destination_context: _DestinationContext


@dataclass(frozen=True)
class PendingAnalysis:
    """A GUI-owned interactive handoff, not another runner or a question.

    tab is its GUI locator. stage is primary/post. op is the opaque MCP session
    operation handle. The same owner operation completes the pending analyze.
    """

    tab: str
    stage: Literal["primary", "post"]
    op: int


class RecipeRuntime:
    """Own one recipe's RPC binding, operation sources and detached run history.

    tools is a bound execution context, host is its shared ScriptHost. This
    adapter concentrates wire decoding and cooperative GUI drain; authors use
    RecipeSession/RecipeTab instead. It never retries mutations or reconnects.
    """

    def __init__(
        self,
        tools: MeasureToolContext,
        host: ScriptHost,
        *,
        on_change: Callable[[], None] | None = None,
    ) -> None:
        self._tools = tools.bound()
        self.host = host
        self._lock = RLock()
        self._runs: list[RecipeRun] = []
        self._details: list[RecipeRunDetail] = []
        self._run_ops: dict[str, int] = {}
        self._analysis_ops: dict[tuple[str, str], int] = {}
        self._candidates: dict[str, tuple[tuple[WritebackItem, str], ...]] = {}
        self._pending_analysis: PendingAnalysis | None = None
        self._on_change = on_change
        self._images: list[PngImage] = []
        self._run_scope: Scope | None = None
        self._run_handle: int | None = None
        self._failures: list[
            tuple[Exception, ExecutionPhase, AnalysisRecovery | None]
        ] = []

    def failure_phase(self, error: Exception) -> ExecutionPhase:
        """Read the phase of this exact runtime failure, or operation for author errors.

        Error identity survives author finally/cleanup and catches. Successful
        cleanup cannot relabel an earlier failure; a new author error is not
        classified using an unrelated, previously caught runtime failure.
        """
        with self._lock:
            for failure, phase, _recovery in reversed(self._failures):
                if failure is error:
                    return phase
            return "operation"

    def failure_recovery(self, error: Exception) -> AnalysisRecovery | None:
        """Return the read-only route for this exact failure, without querying GUI."""
        with self._lock:
            for failure, _phase, recovery in reversed(self._failures):
                if failure is error:
                    return deepcopy(recovery)
            return None

    @contextmanager
    def _failure_phase(
        self, phase: ExecutionPhase, *, recovery: AnalysisRecovery | None = None
    ) -> Generator[None]:
        try:
            yield
        except Exception as error:
            # Preserve the original exception so author-side recovery still works.
            with self._lock:
                self._failures.append((error, phase, deepcopy(recovery)))
            raise

    def runs(self) -> tuple[RecipeRun, ...]:
        """Read immutable acquisition/save facts in execution order, without RPCs."""
        with self._lock:
            return tuple(self._runs)

    def details(self) -> tuple[RecipeRunDetail, ...]:
        """Read detached per-Run owner evidence; never query, analyze, write or save."""
        with self._lock:
            return deepcopy(tuple(self._details))

    def scope_status(self) -> ScopeProjection:
        """Read the local tree and project descendants of its exact GUI Run.

        Local binding/path reads share the host control sequence. The GUI
        observes its own tree under its lock. Released handles contribute no
        tail; transport errors propagate without reconnecting or inventing paths.
        """

        def local() -> tuple[int | None, str | None, ScopeProjection]:
            return (
                self._run_handle,
                None if self._run_scope is None else self._run_scope.path,
                ScopeProjection(self.host.active_path, self.host.live_paths),
            )

        try:
            handle, prefix, paths = self.host.control(local)
        except RuntimeError:
            return ScopeProjection(self.host.active_path, self.host.live_paths)
        if handle is None or prefix is None:
            return paths
        reply = _ScopeReply.model_validate(
            self._tools.gui.read_internal(
                "operation.status_scope", {}, operation_handle=handle
            )
        )
        leaves: list[str] = []
        for path in paths.live_paths:
            if path == prefix and reply.live_paths:
                leaves.extend(
                    self._scope_path(prefix, leaf) for leaf in reply.live_paths
                )
            else:
                leaves.append(path)
        active = paths.active_path
        if active == prefix:
            active = self._scope_path(prefix, reply.active_path)
        return ScopeProjection(active, leaves)

    @property
    def active_path(self) -> str:
        """Read the unique composite chain through the first live fork."""
        return self.scope_status().active_path

    def cancel(
        self,
        scope: str = "recipe",
        *,
        expected: int | None = None,
        op: int | None = None,
    ) -> CancelResult:
        """Cancel a local scope or delegate fine/full-path selectors to its Run.

        run keeps Run-only admission. Fine selectors and composite full paths
        bind to the current tab.run handle; an explicit op must match it.
        Preserve GUI refusal reasons and project paths/candidates, without
        claiming that either acknowledgement proves cleanup has completed.
        """
        label, _sequence = parse_cancel_scope(scope, expected)
        full_path = "/" in label
        if label not in {"schedule", "acquire", "run"} and not full_path:
            return self.host.cancel(scope, expected=expected)

        def local() -> CancelResult | tuple[int | None, str, str]:
            if op is not None and self._run_handle != op:
                return CancelResult(False, "scope not active", self.host.active_path)
            run = self._run_scope
            if run is None:
                if full_path:
                    return self.host.cancel(scope, expected=expected)
                return CancelResult(False, "no run in progress", self.host.active_path)
            if label == "run":
                return self.host.cancel(scope, expected=expected)
            if full_path and not scope.startswith(run.path + "/"):
                return self.host.cancel(scope, expected=expected)
            return self._run_handle, run.path, self.host.active_path

        try:
            target = self.host.control(local)
        except RuntimeError:
            return CancelResult(False, "no run in progress", self.host.active_path)
        if isinstance(target, CancelResult):
            if op is not None and target.reason == "scope not active":
                return replace(target, active_path=self.active_path)
            return target
        handle, prefix, local_path = target
        if handle is None:
            return CancelResult(False, "no run in progress", local_path)
        selector = "run/" + scope[len(prefix) + 1 :] if full_path else scope
        reply = self._tools.gui.read_internal(
            "operation.cancel_scope",
            {"scope": selector, "expected": expected},
            operation_handle=handle,
        )
        result = TypeAdapter(CancelResult).validate_python(reply)
        active = (
            self._scope_path(prefix, result.active_path)
            if local_path == prefix
            else local_path
        )
        return replace(
            result,
            active_path=active,
            candidates=[self._scope_path(prefix, path) for path in result.candidates],
        )

    @staticmethod
    def _scope_path(prefix: str, gui_path: str) -> str:
        tail = gui_path.partition("/")[2]
        return f"{prefix}/{tail}" if tail else prefix

    def _retain_run(self, scope: Scope, handle: int | None = None) -> None:
        self._run_scope = scope
        self._run_handle = handle

    def images(self) -> tuple[PngImage, ...]:
        """Read explicit previews retained in memory; this never captures new PNGs."""
        return tuple(self._images)

    def open_tab(
        self,
        adapter: str,
        *,
        aliases: Mapping[str, str | None] | None = None,
        reuse: str | None = None,
    ) -> RecipeTab:
        """Create or validate an idle matching tab, then Reset its fixed sources."""
        if not adapter.strip() or (reuse is not None and not reuse.strip()):
            raise ValueError("adapter and reuse must be nonblank")
        if reuse is None:
            tab = TypeAdapter(str).validate_python(
                self._tools.send_gui_rpc("tab.new", {"adapter_name": adapter})["tab_id"]
            )
        else:
            tab = reuse
            snapshot = self._snapshot(tab)
            if snapshot.adapter_name != adapter:
                raise GuiRpcError(
                    "Tab has a different adapter", reason="wrong_experiment"
                )
            if snapshot.interaction.is_running or snapshot.interaction.is_analyzing:
                raise GuiRpcError("Tab is busy", reason="tab_busy")
        params: dict[str, object] = {"tab_id": tab}
        overrides = {
            key: value for key, value in (aliases or {}).items() if value is not None
        }
        if overrides:
            params["aliases"] = overrides
        self._tools.send_gui_rpc("tab.reset_cfg", params)
        return RecipeTab(self, tab, adapter)

    def read_source(self, source_path: str, *, missing_ok: bool = False) -> YamlValue:
        """Read one registered session-global value source, without a tab capture.

        source_path is the owner's value.read key (for example
        device.flux.value), not a Form path or expression. With missing_ok=True,
        return None only for an unregistered source. Other source/transport/type
        errors propagate. This never refreshes tab.read sources.
        """
        if not source_path.strip():
            raise ValueError("Source path must be nonblank")
        try:
            reply = self._tools.send_gui_rpc("value.read", {"key": source_path})
        except GuiRpcError as exc:
            if (
                missing_ok
                and exc.code == "invalid_params"
                and exc.reason == "unregistered_source"
            ):
                return None
            raise
        return TypeAdapter(YamlValue).validate_python(reply["value"])

    @property
    def pending_analysis(self) -> PendingAnalysis | None:
        """Read the explicit interactive owner handoff, or None without one."""
        with self._lock:
            return self._pending_analysis

    def _publish_analysis(self, pending: PendingAnalysis | None) -> None:
        with self._lock:
            self._pending_analysis = pending
        if self._on_change is not None:
            self._on_change()

    def read_field(self, tab: str, path: FormPath) -> FormField:
        """Decode the owner's atomic logical field observation without guessing layout."""
        field_path(path)
        reply = self._tools.send_gui_rpc(
            "tab.read_recipe_field", {"tab_id": tab, "path": list(path)}
        )
        observed = read_form_field(field_observation(reply), path)
        selector = observed.selector
        source = observed.source
        return FormField(
            observed.value,
            observed.value_status,
            tuple(FormError(error.path, error.message) for error in observed.errors),
            FormSource(source["key"], source["source_kind"], tuple(source["spec_path"]))
            if source is not None
            else None,
            FormSelector(
                "ref" if selector["mode"] == "key" else selector["mode"],
                selector["key"],
            )
            if selector is not None
            else None,
            observed.locked,
            observed.fields,
        )

    def set_field(self, tab: str, path: FormPath, value: FormValue) -> None:
        """Send one logical native edit; None keeps input without a mutation RPC."""
        field_path(path)
        if value is None:
            return
        payload = TypeAdapter(FormValue).dump_python(value, mode="json")
        reply = self._tools.send_gui_rpc(
            "tab.edit_recipe_fields",
            {"tab_id": tab, "edits": [{"path": list(path), "value": payload}]},
        )
        if reply["errors"]:
            raise GuiRpcError(
                "Owner retained an invalid cfg draft", reason="invalid_cfg"
            )

    def apply_fields(
        self, tab: str, edits: Sequence[FieldEdit], *, require_valid: bool = False
    ) -> None:
        """Send one native batch; validated rejection never installs its candidate.

        None-valued edits retain input. Invalid labels/paths fail before RPC.
        require_valid delegates complete cfg admission to the GUI owner; its
        errors raise RecipeNeedsParameters. Ordinary invalid drafts raise
        invalid_cfg after installation. No rollback or retry occurs.
        """
        changes: list[dict[str, object]] = []
        for edit in edits:
            if not edit.source.strip():
                raise ValueError("Edit source must be nonblank")
            field_path(edit.path)
            if edit.value is not None:
                changes.append(
                    {
                        "path": list(edit.path),
                        "value": TypeAdapter(FormValue).dump_python(
                            edit.value, mode="json"
                        ),
                    }
                )
        if not changes and not require_valid:
            return
        reply = TypeAdapter(RecipeEditResult).validate_python(
            self._tools.send_gui_rpc(
                "tab.edit_recipe_fields",
                {"tab_id": tab, "edits": changes, "require_valid": require_valid},
            ),
            strict=True,
        )
        errors = tuple(
            RecipeFormError(tuple(error["path"]), error["message"])
            for error in reply["errors"]
        )
        if require_valid:
            self._require_cfg(errors)
        elif errors:
            raise GuiRpcError(
                "Owner retained an invalid cfg draft", reason="invalid_cfg"
            )

    @staticmethod
    def _require_cfg(errors: Sequence[RecipeFormError]) -> None:
        if errors:
            missing = tuple(
                MissingParameter(
                    "".join(
                        f"[{part}]"
                        if isinstance(part, int)
                        else ("." if index else "") + part
                        for index, part in enumerate(error.path)
                    )
                    or "$",
                    error.message,
                )
                for error in errors
            )
            raise RecipeNeedsParameters(missing, parameter_errors=tuple(errors))

    def select(
        self, tab: str, path: FormPath, *, ref: str | None, inline: bool
    ) -> None:
        """Dispatch an owner selector command, preserving actual keys and sources."""
        field_path(path)
        if ref is not None and inline:
            raise ValueError("Select ref or inline, not both")
        if ref is None and not inline:
            return
        if ref is not None and not ref.strip():
            raise ValueError("Reference key must be nonblank")
        params: dict[str, object] = {
            "tab_id": tab,
            "path": list(path),
            "action": "use_inline" if inline else "use_library",
        }
        if ref is not None:
            params["key"] = ref
        self._tools.send_gui_rpc("tab.select_recipe_module", params)

    def call[T](
        self, tab: str, method: str, params: BaseModel, *, result: type[T]
    ) -> T:
        """Decode one RPC's reply as the caller's named result model/dataclass."""
        values = params.model_dump(mode="json")
        if method.startswith("tab."):
            values["tab_id"] = tab
        return TypeAdapter(result).validate_python(
            self._tools.send_gui_rpc(method, values)
        )

    async def run(self, tab: str, adapter: str) -> RunResult:
        """Retain Run completion before root cancellation raises at the await.

        tab is the opened GUI locator; adapter is its registered catalog name.
        Descendant interruption returns partial data. Root cancellation prevents
        new acquisition; in-flight acquisition drains first. Genuine errors raise.
        """
        with self._lock:
            index = len(self._runs)
            self._runs.append(RecipeRun(tab, adapter, None))
            self._details.append(RecipeRunDetail(tab, adapter))

        def retain(completion: CallResult[RunResult]) -> None:
            # The author-worker hook runs before the host raises root cancellation.
            result = completion.result
            with self._lock:
                self._runs[index] = replace(
                    self._runs[index],
                    run_id=result.run_id,
                    partial_result=result.partial_result,
                )
                self._details[index] = replace(
                    self._details[index], run_id=result.run_id
                )

        try:
            completion = await self.host.call(
                lambda ctx: self._acquire(tab, index, ctx),
                on_completed=retain,
            )
        except ScriptCancelled:
            raise
        except Exception:
            with self._lock:
                self._runs[index] = replace(self._runs[index], status="failed")
            raise
        finally:
            self._run_scope = None
            self._run_handle = None
        return completion.result

    async def analyze[T](
        self,
        tab: str,
        stage: Literal["primary", "post"],
        result: type[T],
        *,
        interactive: bool,
        updates: Mapping[str, YamlValue] | None = None,
    ) -> AnalysisResult[T]:
        """Await one owner analysis with named native-unit parameter overrides.

        Omitted updates retain owner parameters. The owner validates names and
        values before starting; admission/validation/transport errors propagate.
        Interactive completion uses the same tracked operation. Validate its
        exact public summary as T without saving or answering questions.
        """
        if stage not in ("primary", "post"):
            raise ValueError("stage must be primary or post")
        try:
            completion = await self.host.call(
                lambda ctx: self._analyze(
                    tab, stage, ctx, interactive=interactive, updates=updates
                )
            )
        finally:
            self._publish_analysis(None)
        with self._failure_phase("result_read"):
            analysis = completion.result
            if analysis.status == "failed":
                return AnalysisResult(None, status="failed", reason=analysis.reason)
            return AnalysisResult(TypeAdapter(result).validate_python(analysis.result))

    def save(self, tab: str) -> SaveResult:
        """Save fixed Run raw then all completed artifacts and retain the whole attempt."""
        with self._failure_phase("image_save"):
            op = self._run_ops[tab]
            reports: list[SaveReport] = []
            try:
                for method, extra in (
                    ("tab.save_data", {}),
                    ("tab.save_artifacts", {"artifacts": "all"}),
                ):
                    reply = self._tools.gui.send_gui_rpc(
                        method, {"tab_id": tab, **extra}, run_operation_handle=op
                    )
                    reports.append(TypeAdapter(SaveReport).validate_python(reply))
            except Exception:
                if reports:
                    self._record_save(tab, reports, failed=True)
                raise
            return self._record_save(tab, reports)

    def _record_save(
        self, tab: str, reports: list[SaveReport], *, failed: bool = False
    ) -> SaveResult:
        if any(report.run_id != reports[0].run_id for report in reports):
            raise GuiRpcError(
                "Save reports identify different Runs", reason="result_superseded"
            )
        outputs = tuple(
            output
            for report in reports
            for section in report.artifacts.values()
            for artifact in section.values()
            for members in artifact.members.values()
            for output in members
        )
        saved = SaveResult(
            "failed"
            if failed or any(output.status == "failed" for output in outputs)
            else "completed",
            reports[0].run_id,
            outputs,
            tuple(reports),
        )
        with self._lock:
            index = next(
                index
                for index in range(len(self._runs) - 1, -1, -1)
                if self._runs[index].tab == tab
            )
            run = self._runs[index]
            self._runs[index] = replace(run, saves=(*run.saves, saved))
        return saved

    def preview(self, tab: str, pane: Literal["run", "analysis", "post"]) -> None:
        """Capture one explicitly selected PNG into execution reply memory."""
        with self._failure_phase("figure_read"):
            if pane not in ("run", "analysis", "post"):
                raise ValueError("Unknown preview pane")
            subtab = "post_analysis" if pane == "post" else pane
            reply = self._tools.gui.send_gui_rpc(
                "tab.get_figure",
                {"tab_id": tab, "subtab_id": subtab},
                run_operation_handle=self._run_ops[tab],
            )
            encoded = TypeAdapter(str).validate_python(reply["png_b64"])
            self._images.append(validated_png(base64.b64decode(encoded, validate=True)))

    def writeback_proposal(self, tab: str) -> WritebackProposal:
        """Capture candidates from completed stages; labels are preview-time display only."""
        with self._failure_phase("writeback_read"):
            retained: list[tuple[WritebackItem, str]] = []
            for stage, pane in (("primary", "analysis"), ("post", "post_analysis")):
                op = self._analysis_ops.get((tab, stage))
                if op is None:
                    continue
                preview = _Preview.model_validate(
                    self._tools.gui.send_gui_rpc(
                        "tab.writeback_preview",
                        {"tab_id": tab, "subtab_id": pane},
                        operation_handle=op,
                    )
                )
                for candidate in preview.items if preview.has_draft else ():
                    item = WritebackItem(
                        candidate.target_name,
                        WritebackDestination(
                            candidate.actual_path,
                            preview.destination_context.active_label,
                        ),
                        candidate.proposed,
                        None,
                        "primary" if stage == "primary" else "post",
                    )
                    retained.append((item, candidate.id))
            # Author-visible YamlValue mappings must not edit retained acceptance evidence.
            self._candidates[tab] = deepcopy(tuple(retained))
            proposal = WritebackProposal(tuple(item for item, _ in retained))
            with self._lock:
                index = self._detail_index(tab)
                self._details[index] = replace(
                    self._details[index], writeback=deepcopy(proposal)
                )
            return proposal

    def accept_writeback(
        self, tab: str, items: Sequence[WritebackItem]
    ) -> WritebackReceipt:
        """Select captured IDs, then write current owner draft values/destinations.

        Receipt items are the original captured candidates for confirmed IDs,
        not snapshots of actual written values. Preserve confirmed stage prefixes.
        """
        captured = self._candidates.get(tab, ())
        chosen = tuple(items)
        if len({item.name for item in chosen}) != len(chosen) or any(
            item not in tuple(item for item, _ in captured) for item in chosen
        ):
            raise ValueError("Unknown, duplicate or modified writeback candidate")
        stages: list[dict[str, object]] = []
        for stage, pane in (("primary", "analysis"), ("post", "post_analysis")):
            ids = [
                identity
                for item, identity in captured
                if item.stage == stage and item in chosen
            ]
            if ids:
                stages.append(
                    {
                        "subtab_id": pane,
                        "ids": ids,
                    }
                )
        identity: str | None = None
        try:
            identity = begin_writeback(self._tools, tab)
            reply = _Acceptance.model_validate(
                self._tools.gui.send_gui_rpc(
                    "tab.writeback_accept",
                    {"tab_id": tab, "stages": stages, "submission_id": identity},
                )
            )
        except GuiRpcError as error:
            receipt = _writeback_failure_receipt(error, identity)
        else:
            confirmed = {
                (row.subtab_id, identity)
                for row in reply.written
                for identity in row.ids
            }
            applied = tuple(
                item
                for item, identity in captured
                if item in chosen
                and (
                    "analysis" if item.stage == "primary" else "post_analysis",
                    identity,
                )
                in confirmed
            )
            receipt = WritebackReceipt(
                reply.status,
                deepcopy(applied),
                reply.error,
                identity,
                writeback_recovery(identity),
            )
        with self._lock:
            index = self._detail_index(tab)
            detail = self._details[index]
            self._details[index] = replace(
                detail,
                writeback_receipts=(*detail.writeback_receipts, deepcopy(receipt)),
            )
        return receipt

    def _snapshot(self, tab: str) -> _TabSnapshot:
        snapshots = _Snapshots.model_validate(
            self._tools.send_gui_rpc("tab.snapshot", {"tab_id": tab})
        ).tabs
        if len(snapshots) != 1 or snapshots[0].tab_id != tab:
            raise GuiRpcError("Requested tab was not found", reason="unknown_tab")
        return snapshots[0]

    def _await(
        self, op: int, ctx: CallContext, *, analysis: bool = False
    ) -> _Completion:
        stopped = False
        while True:
            began = time.monotonic()
            if ctx.scope.stop.is_set() and not stopped:
                self._tools.gui.read_internal(
                    "operation.cancel", {}, operation_handle=op
                )
                stopped = True
            reply = self._tools.gui.send_gui_rpc(
                "operation.await", {"timeout": 0.25}, 2.25, operation_handle=op
            )
            completion = _Completion.model_validate(reply)
            if completion.reason == "completed":
                if completion.status == "failed":
                    if analysis and completion.failure_kind == "analysis":
                        return completion
                    raise GuiRpcError(
                        completion.error_message,
                        reason="operation_failed",
                    )
                if completion.status not in ("finished", "cancelled"):
                    raise GuiRpcError(
                        "Missing terminal operation status", reason="incompatible_wire"
                    )
                return completion
            delay = max(0.0, 0.25 - (time.monotonic() - began))
            if stopped:
                time.sleep(delay)
            else:
                ctx.scope.stop.event.wait(delay)

    def _detail_index(self, tab: str) -> int:
        return next(
            index
            for index in range(len(self._details) - 1, -1, -1)
            if self._details[index].tab == tab
        )

    def _acquire(self, tab: str, index: int, ctx: CallContext) -> RunResult:
        self.host.control(lambda: self._retain_run(ctx.scope))
        conditions = field_observation(
            self._tools.send_gui_rpc(
                "tab.read_recipe_field", {"tab_id": tab, "path": []}
            )
        )
        with self._lock:
            self._details[index] = replace(self._details[index], conditions=conditions)
        self._require_cfg(
            tuple(
                RecipeFormError(tuple(error["path"]), error["message"])
                for error in conditions["errors"]
            )
        )
        started = _Started.model_validate(
            self._tools.send_gui_rpc("tab.run_start", {"tab_id": tab})
        )
        self._run_ops[tab] = started.handle
        self.host.control(lambda: self._retain_run(ctx.scope, started.handle))
        with self._lock:
            self._details[index] = replace(self._details[index], op=started.handle)
        self._analysis_ops.pop((tab, "primary"), None)
        self._analysis_ops.pop((tab, "post"), None)
        completion = self._await(started.handle, ctx)
        snapshot = self._snapshot(tab)
        source = snapshot.result_state.source_operation_id
        if (
            source is not None
            and self._tools.gui.expose_operation(source) != started.handle
        ):
            raise GuiRpcError("Tab contains another Run", reason="result_superseded")
        if completion.status == "finished" and not snapshot.result_state.available:
            raise GuiRpcError(
                "Run published no usable data", reason="run_result_unavailable"
            )
        return RunResult(
            "completed",
            snapshot.raw_data.run_id,
            snapshot.result_state.partial_result,
        )

    def _analyze(
        self,
        tab: str,
        stage: str,
        ctx: CallContext,
        *,
        interactive: bool,
        updates: Mapping[str, YamlValue] | None,
    ) -> AnalysisResult[object]:
        if ctx.scope.cancelled:
            return AnalysisResult(None, status="failed", reason="Analysis cancelled")
        run = self._run_ops[tab]
        primary = self._analysis_ops[(tab, "primary")] if stage == "post" else None
        started = _Started.model_validate(
            self._tools.gui.send_gui_rpc(
                "tab.analyze" if stage == "primary" else "tab.post_analyze",
                {
                    "tab_id": tab,
                    "updates": dict(updates) if updates is not None else {},
                },
                run_operation_handle=run,
                operation_handle=primary,
            )
        )
        if started.interactive and not interactive:
            self._tools.gui.read_internal(
                "operation.cancel", {}, operation_handle=started.handle
            )
            self._await(started.handle, ctx)
            raise GuiRpcError(
                "Interactive analysis requires interactive=True",
                reason="interaction_required",
            )
        if started.interactive:
            pending = PendingAnalysis(
                tab, "primary" if stage == "primary" else "post", started.handle
            )
            self.host.control(lambda: self._publish_analysis(pending))
        completion = self._await(started.handle, ctx, analysis=True)
        if completion.status == "failed":
            return AnalysisResult(
                None, status="failed", reason=completion.error_message
            )
        if completion.status == "cancelled":
            return AnalysisResult(None, status="failed", reason="Analysis cancelled")
        recovery: AnalysisRecovery | None = (
            {**started.recovery, "op": started.handle}
            if stage == "post" and started.recovery is not None
            else None
        )
        with self._failure_phase("result_read", recovery=recovery):
            reply = self._tools.gui.send_gui_rpc(
                "tab.get_analyze_result"
                if stage == "primary"
                else "tab.get_post_analyze_result",
                {"tab_id": tab},
                timeout_seconds=started.result_read_timeout_seconds
                if stage == "post"
                else None,
                operation_handle=started.handle,
            )
            summary = TypeAdapter(YamlValue).validate_python(reply["summary"])
            invalid = TypeAdapter(tuple[InvalidAnalysisValue, ...]).validate_python(
                reply["invalid"]
            )
            captured = RecipeAnalysisDetail(
                "primary" if stage == "primary" else "post",
                started.handle,
                summary,
                invalid,
                presentation=TypeAdapter(AnalysisPresentation | None).validate_python(
                    reply.get("presentation")
                ),
            )
            with self._lock:
                index = self._detail_index(tab)
                detail = self._details[index]
                self._details[index] = replace(
                    detail, analyses=(*detail.analyses, captured)
                )
            self._analysis_ops[(tab, stage)] = started.handle
            return AnalysisResult(summary)
