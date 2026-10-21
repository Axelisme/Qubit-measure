"""Headless form capture, native lowering and run resources for one workflow tab."""

from collections.abc import Callable, Mapping
from typing import Literal, Protocol

from pydantic import JsonValue
from pydantic_core import PydanticCustomError
from qick import QickConfig

from zcu_tools.core.script_protocol import Answer
from zcu_tools.datafile import RunSnapshot
from zcu_tools.experiment.acquisition import capture_software_provenance
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.stop_signal import StopSignal
from zcu_tools.experiment.workflows import (
    Actor,
    CommittedRecord,
    EnginePorts,
    InvalidRunState,
    RunIdentity,
    RunPaths,
    RunStatus,
    TunablesSnapshot,
)
from zcu_tools.experiment.workflows.clock import UtcClock
from zcu_tools.gui.form import FormEnv, FormErrors, FormModel, FormState, FormWritePath
from zcu_tools.gui.session.services.build import SessionServices
from zcu_tools.gui.session.state import SessionState
from zcu_tools.plotting.plots import NonPresentingHost, PlotHost, Plots
from zcu_tools.progress_bar.backend.tqdm import TQDMProgressBar
from zcu_tools.resources.entry.save_layout import workflow_roots

from .workflow_attachment import Attachment
from .workflow_devices import SessionDevices
from .workflow_run import PendingQuestion, QuestionExchange, WorkflowController

FormKind = Literal["plan", "tunables"]
type FormBatch = Mapping[FormWritePath, JsonValue]


class WorkflowRunPort(Protocol):
    """Native-independent run control used to route heterogeneous registrations."""

    @property
    def run_id(self) -> str | None: ...
    @property
    def operation_id(self) -> int | None: ...
    @property
    def last_exchange(self) -> QuestionExchange | None: ...
    @property
    def has_pending_segment(self) -> bool: ...
    def status(self) -> RunStatus | None: ...
    def require_run_id(self, run_id: str) -> None: ...
    def pause(self, run_id: str) -> RunStatus: ...
    def resume(self, run_id: str) -> int: ...
    def stop(self, run_id: str) -> RunStatus | None: ...
    def pending_question(self, run_id: str) -> PendingQuestion | None: ...
    def answer(self, run_id: str, answer: Answer | JsonValue) -> int: ...
    def apply_tunables(
        self,
        run_id: str,
        batch: FormBatch | None = None,
        *,
        actor: Actor,
        local_errors: FormErrors | None = None,
    ) -> TunablesSnapshot | FormErrors: ...


class WorkflowTabPort(Protocol):
    """Erase only native type pairing, retaining each tab's owning commands.

    The generic tab keeps its declaration, context and Engine paired internally.
    No pane or frontend implements this port. Calls run on the session owner.
    """

    @property
    def name(self) -> str: ...
    @property
    def run(self) -> WorkflowRunPort: ...
    @property
    def plots(self) -> Plots | None: ...
    @property
    def busy(self) -> bool: ...
    @property
    def error(self) -> str | None: ...
    def form_state(self, form: FormKind) -> FormState[FormModel]: ...
    def form_env(self, form: FormKind) -> FormEnv: ...
    def prepare_forms(self) -> None: ...
    def set_plot_host(self, host: PlotHost) -> None: ...
    def start(
        self,
        run_id: str,
        *,
        plan_batch: FormBatch | None = None,
        tunables_batch: FormBatch | None = None,
        plan_errors: FormErrors | None = None,
        tunables_errors: FormErrors | None = None,
    ) -> int | FormErrors: ...
    def records(self, run_id: str, since: int) -> tuple[CommittedRecord, ...]: ...
    def tunables(self, run_id: str) -> TunablesSnapshot: ...
    def release(self) -> None: ...


class WorkflowTab[C]:
    """Own a tab's form drafts, fixed Start capture, run and retained Plots.

    Construct with an attachment from the caller's catalog and the exact shared
    session/state pair. changed notifies owner-thread observers after delivery.
    This owner never creates widgets or closes/drains session resources.
    """

    def __init__(
        self,
        attachment: Attachment[C],
        state: SessionState,
        services: SessionServices,
        changed: Callable[[], None],
    ) -> None:
        self._attachment = attachment
        self._state = state
        self._services = services
        self._changed = changed
        self._host: PlotHost = NonPresentingHost()
        self.plots: Plots | None = None
        self.error: str | None = None
        self.run = WorkflowController[FormModel, C](
            services.runner, f"workflow:{id(self)}", on_finished=self._finished
        )
        self._env = (
            self._capture() if state.session_env.parameters is not None else FormEnv()
        )
        self._plan = FormState(attachment.plan_model.default_input(self._env))
        self._tunables = FormState(attachment.knobs_model.default_input(self._env))
        self._forms_from_run = False

    @property
    def name(self) -> str:
        """Return this tab's immutable registration name."""
        return self._attachment.name

    @property
    def busy(self) -> bool:
        """True until terminal owner delivery, including all paused ownership."""
        return self.run.has_pending_segment or self.run.operation_id is not None

    def _ensure_idle(self) -> None:
        if self.busy:
            status = self.run.status()
            raise InvalidRunState(
                "start", "running" if status is None else status.lifecycle
            )

    def _capture(self) -> FormEnv:
        parameters = self._services.context.read_context()
        return FormEnv(
            parameters=parameters,
            library=parameters.library,
            devices=self._services.device.capture_native_device_info(),
        )

    def form_state(self, form: FormKind) -> FormState[FormModel]:
        """Return the unique draft to render; tunables switches to accepted binding."""
        if form == "plan":
            return self._plan
        if form != "tunables":
            raise ValueError(f"Unknown workflow form: {form}")
        binding = self.run.binding
        return (
            self._tunables
            if binding is None or not self._forms_from_run
            else binding.state
        )

    def form_env(self, form: FormKind) -> FormEnv:
        """Return draft's fixed capture, including terminal run tunables."""
        self.form_state(form)
        binding = self.run.binding
        return self._env if binding is None or not self._forms_from_run else binding.env

    def prepare_forms(self) -> None:
        """Refresh idle source capture without dropping authored input or selections."""
        self._ensure_idle()
        env = self._capture()
        tunables = FormState(self.form_state("tunables").input)
        self._env = env
        self._tunables = tunables
        self._forms_from_run = False
        # The historical binding stays available to explicit run-id queries.
        self._changed()

    def set_plot_host(self, host: PlotHost) -> None:
        """Attach a driving plot adapter before Start; reject while run is owned."""
        self._ensure_idle()
        self._host = host

    def _prepare(
        self,
        form: FormKind,
        batch: FormBatch | None,
        env: FormEnv,
        errors: FormErrors | None,
    ) -> tuple[FormModel, FormState[FormModel]] | FormErrors:
        if batch is None:
            if errors is not None and errors.messages:
                return errors
            draft = FormState(self.form_state(form).input)
        else:
            model = (
                self._attachment.plan_model
                if form == "plan"
                else self._attachment.knobs_model
            )
            try:
                draft = FormState(model.default_input(env))
                draft.write(batch, env)
            except PydanticCustomError as error:
                if error.type != "form_field" or error.context is None:
                    raise
                return FormErrors({error.context["path"]: error.context["message"]})
            except ValueError as error:
                return FormErrors({(): str(error)})
        value = draft.result(env)
        return value if isinstance(value, FormErrors) else (value, draft)

    def start(
        self,
        run_id: str,
        *,
        plan_batch: FormBatch | None = None,
        tunables_batch: FormBatch | None = None,
        plan_errors: FormErrors | None = None,
        tunables_errors: FormErrors | None = None,
    ) -> int | FormErrors:
        """Capture sources and lower forms once, then submit under the shared RUN lease.

        None batches use parsed drafts and optional renderer-local errors. Explicit
        batches use fresh captured defaults. Invalid models return FormErrors;
        structure errors return FormErrors. Source, SoC, path and admission failures
        propagate. No draft is
        accepted on rejection. The operation token is not native acceptance.
        """
        self._ensure_idle()
        env = self._capture()
        parameters = env.parameters
        if parameters is None:
            raise ValueError("Workflow Start requires captured parameters")
        storage = self._services.context.capture_storage(parameters)
        plan = self._prepare("plan", plan_batch, env, plan_errors)
        knobs = self._prepare("tunables", tunables_batch, env, tunables_errors)
        if isinstance(plan, FormErrors):
            return plan
        if isinstance(knobs, FormErrors):
            return knobs
        snapshot = RunSnapshot(
            entry_id=storage.entry_id,
            entry_name=storage.entry_name,
            point=storage.point,
            description=storage.description,
            roles=storage.roles,
        )
        prepared = self._attachment.prepare(
            plan[0], knobs[0], knobs[1].input, env, snapshot
        )
        if isinstance(prepared, FormErrors):
            return prepared
        session = self._state.session_env
        if session.soc is None or not isinstance(session.soccfg, QickConfig):
            raise ValueError(
                "Workflow Start requires connected SoC and native QickConfig"
            )
        roots = workflow_roots(
            result_path=storage.result_path,
            database_path=storage.database_path,
            workflow=self.name,
            run_id=run_id,
        )
        plots = Plots(self._host)
        context = RunContext(
            session.soc,
            session.soccfg,
            plots,
            self._services.device.get_connected_devices(),
            StopSignal(),
        )
        identity = RunIdentity(
            run_id,
            capture_software_provenance(context=context),
            str(storage.entry_id),
            storage.point,
        )
        ports = EnginePorts(
            plots=plots,
            progress=TQDMProgressBar,
            clock=UtcClock(),
            context=prepared.context,
            soc=session.soc,
            soccfg=session.soccfg,
            devices=SessionDevices(self._services.device),
        )
        self.release()
        self.plots = plots
        self.error = None
        try:
            token = prepared.submit(
                self.run,
                ports,
                RunPaths(metadata_root=roots.result, data_root=roots.database),
                identity,
                env,
            )
        except Exception:
            plots.finish(present=False)
            plots.release()
            self.plots = None
            raise
        self._env = env
        self._forms_from_run = True
        self._plan.reset(plan[1].input)
        self._tunables.reset(knobs[1].input)
        self._changed()
        return token

    def _finished(self, status: RunStatus | None, error: Exception | None) -> None:
        self.error = None if error is None else str(error)
        if self.plots is not None and (status is None or status.lifecycle != "paused"):
            self.plots.finish()
        self._changed()

    def records(self, run_id: str, since: int) -> tuple[CommittedRecord, ...]:
        """Read native committed summaries after target and acceptance validation."""
        self.run.require_run_id(run_id)
        engine = self.run.engine
        if engine is None:
            raise RuntimeError("Records require native acceptance")
        return engine.records(run_id, since=since)

    def tunables(self, run_id: str) -> TunablesSnapshot:
        """Read current detached values/revision; reject pending native acceptance."""
        self.run.require_run_id(run_id)
        engine = self.run.engine
        if engine is None:
            raise RuntimeError("Tunables require native acceptance")
        return engine.tunables(run_id)

    def release(self) -> None:
        """Release retained plot presentation only; never disconnect or drain session."""
        self._ensure_idle()
        if self.plots is not None:
            self.plots.release()
            self.plots = None
