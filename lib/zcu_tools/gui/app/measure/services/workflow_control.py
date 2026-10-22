"""Public workflow commands over headless per-tab owners, shared by GUI and MCP."""

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from pydantic import JsonValue

from zcu_tools.core.script_protocol import Answer
from zcu_tools.experiment.stop_signal import CancelResult
from zcu_tools.experiment.workflows import (
    Actor,
    CommittedRecord,
    RunMismatch,
    RunStatus,
    TunablesSnapshot,
    WorkflowPlots,
)
from zcu_tools.gui.form import (
    FormDescription,
    FormEnv,
    FormErrors,
    FormInput,
    FormModel,
    FormPreview,
    FormState,
    FormWritePath,
)
from zcu_tools.gui.plotting.figure_export import render_figure_png
from zcu_tools.gui.session.services.build import SessionServices
from zcu_tools.gui.session.state import SessionState
from zcu_tools.plotting.plots import PlotHost

from .workflow_attachment import Attachment
from .workflow_run import PendingQuestion, QuestionExchange
from .workflow_tab import FormBatch, FormKind, WorkflowTab, WorkflowTabPort

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkflowTabSnapshot:
    """tab_id is stable per open; name may repeat; run_id/token are absent before Start.

    status is native acceptance, not submission. busy includes pending owner
    delivery and paused lease ownership. error is the last producer failure text,
    or None when no failure was delivered. pending_question is the detached
    answerable Ask envelope, or None outside a delivered unanswered Ask pause.
    last_exchange retains the latest per-run question and accepted answer facts.
    """

    tab_id: str
    name: str
    run_id: str | None
    operation_id: int | None
    status: RunStatus | None
    busy: bool
    error: str | None
    pending_question: PendingQuestion | None = None
    last_exchange: QuestionExchange | None = None


@dataclass(frozen=True)
class WorkflowFormSnapshot:
    """One detached draft/candidate observation from the tab's fixed capture.

    input retains authored values and selections. preview holds ordinary leaves
    and complete Form errors, without lowering native cfg. description is one
    requested layer, or None when no path was requested. No field grants admission.
    """

    input: FormInput[FormModel]
    preview: FormPreview
    description: FormDescription | None


class WorkflowControlFacet:
    """Own the tab registry and route commands to headless workflow owners.

    Use on the session owner thread. configure supplies caller registrations;
    open creates independent drafts, never widgets. Run commands target native
    run_id; tab/form commands target tab_id. Unknown names/tabs raise KeyError,
    stale runs raise RunMismatch. Subscribers observe registry/delivery changes
    on-owner and must unsubscribe before destroying children. This facet never
    closes hardware or drains the caller's session. Ask continuations use the
    retained worker/operation; this owner never executes author code inline.
    """

    def __init__(
        self,
        state: SessionState,
        services: SessionServices,
    ) -> None:
        """Borrow one existing session/state pair."""
        self._state = state
        self._services = services
        self._factories: dict[str, Callable[[], WorkflowTabPort]] = {}
        self._tabs: dict[str, WorkflowTabPort] = {}
        self._next_id = 0
        self._listeners: list[Callable[[], None]] = []

    def configure[C](self, registrations: Sequence[Attachment[C]]) -> None:
        """Replace idle catalog; reject open tabs, empty/duplicate declarations.

        Native generic pairing is captured in factories, not erased by casts.
        No widgets, files or hardware are created. Existing tabs must first close.
        """
        if self._tabs:
            raise RuntimeError("Close workflow tabs before replacing registrations")
        if not registrations:
            raise ValueError("Workflow registrations must be nonempty")
        factories: dict[str, Callable[[], WorkflowTabPort]] = {}
        for registration in registrations:
            if registration.name in factories:
                raise ValueError(f"Duplicate workflow name: {registration.name}")
            factories[registration.name] = self._factory(registration)
        self._factories = factories
        self._notify()

    def _factory[C](self, attachment: Attachment[C]) -> Callable[[], WorkflowTabPort]:
        return lambda: WorkflowTab(
            attachment, self._state, self._services, self._notify
        )

    def names(self) -> tuple[str, ...]:
        """Return registration names in caller order; empty until configure."""
        return tuple(self._factories)

    def open(self, name: str) -> str:
        """Create independent captured drafts for name and return a stable tab_id."""
        tab = self._factories[name]()
        self._next_id += 1
        tab_id = f"workflow-{self._next_id}"
        self._tabs[tab_id] = tab
        self._notify()
        return tab_id

    def close(self, tab_id: str) -> bool:
        """Release an idle tab; return False without mutation while pending/paused.

        The pane must retain itself on False. After True it can dispose its own
        callbacks/timers/editors before deleting children. No session drain,
        cancel, disconnect or Qt deletion occurs. Release errors retain the tab.
        """
        tab = self._tabs[tab_id]
        if tab.busy:
            return False
        tab.release()
        del self._tabs[tab_id]
        self._notify()
        return True

    def tabs(self) -> tuple[WorkflowTabSnapshot, ...]:
        """Project opened tabs in insertion order without rendering or hardware I/O."""
        return tuple(
            WorkflowTabSnapshot(
                key,
                tab.name,
                tab.run.run_id,
                tab.run.operation_id,
                tab.run.status(),
                tab.busy,
                tab.error,
                None
                if tab.run.run_id is None
                else tab.run.pending_question(tab.run.run_id),
                tab.run.last_exchange,
            )
            for key, tab in self._tabs.items()
        )

    def subscribe(self, listener: Callable[[], None]) -> Callable[[], None]:
        """Observe committed tab/control changes; return an idempotent unsubscribe."""
        self._listeners.append(listener)

        def unsubscribe() -> None:
            if listener in self._listeners:
                self._listeners.remove(listener)

        return unsubscribe

    def _notify(self) -> None:
        for listener in tuple(self._listeners):
            try:
                listener()
            except Exception:
                logger.exception("Workflow observer failed")

    def _run_tab(self, run_id: str) -> WorkflowTabPort:
        for tab in self._tabs.values():
            if tab.run.run_id == run_id:
                return tab
        raise RunMismatch(run_id, None)

    def start(
        self,
        tab_id: str,
        run_id: str,
        *,
        plan_batch: FormBatch | None = None,
        tunables_batch: FormBatch | None = None,
        plan_errors: FormErrors | None = None,
        tunables_errors: FormErrors | None = None,
    ) -> int | FormErrors:
        """Submit fixed captured forms; return pending token or directed form errors.

        run_id must be path-safe and unique across tabs. None batches read draft
        plus optional renderer errors; explicit batches use fresh Start defaults.
        The same session RUN gate arbitrates experiments and every workflow tab.
        Native/source/admission failures propagate without accepting draft bases.
        """
        tab = self._tabs[tab_id]
        if any(
            other is not tab and other.run.run_id == run_id
            for other in self._tabs.values()
        ):
            raise ValueError(f"Duplicate workflow run_id: {run_id}")
        return tab.start(
            run_id,
            plan_batch=plan_batch,
            tunables_batch=tunables_batch,
            plan_errors=plan_errors,
            tunables_errors=tunables_errors,
        )

    def status(self, run_id: str) -> RunStatus | None:
        """Read native state; None means this target is still pending acceptance."""
        return self._run_tab(run_id).run.status()

    def pause(self, run_id: str) -> RunStatus:
        """Request cooperative Pause while retaining the operation and RUN lease."""
        result = self._run_tab(run_id).run.pause(run_id)
        self._notify()
        return result

    def resume(self, run_id: str) -> int:
        """Continue a delivered paused run with its original operation token."""
        result = self._run_tab(run_id).run.resume(run_id)
        self._notify()
        return result

    def cancel(
        self, run_id: str, scope: str = "script", *, expected: int | None = None
    ) -> CancelResult:
        """Match one scope in the explicit native run, acknowledging without drain."""
        result = self._run_tab(run_id).run.cancel(run_id, scope, expected=expected)
        self._notify()
        return result

    def stop(self, run_id: str) -> RunStatus | None:
        """Request Stop; pending work releases ownership only on owner delivery.

        Control I/O failure ends as failed teardown, not cancellation. The original
        error propagates; worker drain runs author cleanup before terminal delivery.
        """
        result = self._run_tab(run_id).run.stop(run_id)
        self._notify()
        return result

    def form_state(self, tab_id: str, form: FormKind) -> FormState[FormModel]:
        """Borrow the unique draft for FormEditor; never retain it past tab close."""
        return self._tabs[tab_id].form_state(form)

    def form_env(self, tab_id: str, form: FormKind) -> FormEnv:
        """Read that draft's fixed source capture, not a fresh session snapshot."""
        return self._tabs[tab_id].form_env(form)

    def prepare_forms(self, tab_id: str) -> None:
        """Recapture idle form sources; failures preserve drafts and old capture."""
        self._tabs[tab_id].prepare_forms()

    def write_form(self, tab_id: str, form: FormKind, batch: FormBatch) -> None:
        """Write parsed edits to one draft; reject busy Start-plan edits before writes."""
        tab = self._tabs[tab_id]
        if tab.busy and form == "plan":
            raise RuntimeError("Workflow plan is locked during a run")
        tab.form_state(form).write(batch, tab.form_env(form))
        self._notify()

    def inspect_form(self, tab_id: str, form: FormKind) -> FormPreview:
        """Preview ordinary values/errors in the same captured draft, without writes."""
        tab = self._tabs[tab_id]
        return tab.form_state(form).preview(tab.form_env(form))

    def describe_form(
        self, tab_id: str, form: FormKind, path: FormWritePath = ()
    ) -> FormDescription:
        """Describe one canonical path using the shared core Form contract."""
        tab = self._tabs[tab_id]
        return tab.form_state(form).describe(path, tab.form_env(form))

    def observe_form(
        self,
        tab_id: str,
        form: FormKind,
        *,
        batch: FormBatch | None = None,
        path: FormWritePath | None = None,
    ) -> WorkflowFormSnapshot:
        """Read or preview one candidate and optional metadata without publication.

        None batch reads the parsed shared draft; even an empty batch previews
        that draft, never fresh defaults. path=None omits metadata, () describes
        root. All observations use the same candidate and fixed FormEnv. Form
        structural/source/path failures propagate; ordinary invalid input stays
        readable via preview.errors. No input/base, run or capture is changed.
        """
        tab = self._tabs[tab_id]
        env = tab.form_env(form)
        candidate = FormState(tab.form_state(form).input)
        if batch is not None:
            candidate.write(batch, env)
        description = None if path is None else candidate.describe(path, env)
        return WorkflowFormSnapshot(
            candidate.input, candidate.preview(env), description
        )

    def tunables(self, run_id: str) -> TunablesSnapshot:
        """Read detached native values and revision after acceptance."""
        return self._run_tab(run_id).tunables(run_id)

    def apply_tunables(
        self,
        run_id: str,
        batch: FormBatch | None = None,
        *,
        actor: Actor,
        local_errors: FormErrors | None = None,
    ) -> TunablesSnapshot | FormErrors:
        """Submit the shared draft or external edits with native revision handling."""
        result = self._run_tab(run_id).run.apply_tunables(
            run_id,
            batch,
            actor=actor,
            local_errors=local_errors,
        )
        self._notify()
        return result

    def records(self, run_id: str, *, since: int = 0) -> tuple[CommittedRecord, ...]:
        """Read committed summaries strictly after since; never scan saved files."""
        return self._run_tab(run_id).records(run_id, since)

    def set_plot_host(self, tab_id: str, host: PlotHost) -> None:
        """Attach a driving PlotHost while idle; caller retains it through delivery."""
        self._tabs[tab_id].set_plot_host(host)

    def plots(self, tab_id: str) -> WorkflowPlots | None:
        """Read retained native presentation for a driving widget; None before Start."""
        return self._tabs[tab_id].plots

    def figures(self, run_id: str) -> tuple[str, ...]:
        """Read retained names for an accepted run; no rendering or hardware I/O."""
        tab = self._run_tab(run_id)
        tab.tunables(run_id)
        return () if tab.plots is None else tuple(tab.plots.figures())

    def figure_png(self, run_id: str, name: str) -> bytes:
        """Render a named retained figure on owner; unknown name raises KeyError."""
        tab = self._run_tab(run_id)
        tab.tunables(run_id)
        if tab.plots is None:
            raise RuntimeError("Run plots have been released")
        return render_figure_png(tab.plots.figures()[name])

    def pending_question(self, run_id: str) -> PendingQuestion | None:
        """Read a detached question/type/schema envelope, or None outside Ask.

        The original worker remains pending while the question is answerable.
        Reads do not consume it, execute author code or change selection. Stale
        run targets raise RunMismatch.
        """
        return self._run_tab(run_id).run.pending_question(run_id)

    def answer(
        self, run_id: str, answer: Answer | JsonValue, *, request_id: str
    ) -> int:
        """Deliver the rendered question ID under the original token and RUN lease.

        Engine matches request_id before parsing JSON fields. Stale or absent
        questions raise QuestionMismatch with the current pending ID. Invalid
        fields/model types raise ValidationError/TypeError without consumption.
        Stale run targets raise RunMismatch. Return the same pending token as
        Start, not terminal completion. Author code stays on its original worker;
        no new Resume/Execute or Runner continuation follows.
        """
        result = self._run_tab(run_id).run.answer(run_id, answer, request_id=request_id)
        self._notify()
        return result
