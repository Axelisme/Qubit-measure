"""App-local native/form pairing; Controller remains the only run owner."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel

from zcu_tools.datafile import RunSnapshot
from zcu_tools.experiment.workflows import EnginePorts, RunIdentity, RunPaths
from zcu_tools.experiment.workflows.declaration import WorkflowScript, workflow_definition
from zcu_tools.gui.form import FormEnv, FormErrors, FormInput, FormModel, FormState

from .workflow_run import WorkflowController
from .workflow_tunables import TunablesHost


@dataclass(frozen=True)
class PreparedWorkflow[C]:
    """A validated native/form pairing ready for Window's captured Start.

    context is the workflow-defined seed of type C, or None if unused. submit
    calls the supplied fixed Controller with typed native inputs and a pending
    Host over accepted input. Call submit(controller, ports, paths, identity, env)
    once after host admission. ports.context must contain this context; paths are
    the new run's metadata/data roots, identity is its native run identity, and env
    is the same fixed form capture used to prepare. Return a pending operation
    token, not proof of native acceptance. Controller admission/native failures
    propagate. No capture/admission policy is duplicated here. No callable owns a run,
    lease or live source.
    """

    context: C | None
    submit: Callable[
        [
            WorkflowController[FormModel, C],
            EnginePorts[C],
            RunPaths,
            RunIdentity,
            FormEnv,
        ],
        int,
    ]


class Attachment[C](Protocol):
    """Common app view of a coherent typed native/form attachment.

    name is the native declaration name. Models are the current independent form
    declarations; panels render those models in a fixed capture. prepare accepts
    their validated ordinary instances plus accepted knobs expression input and
    a host snapshot. It returns directed errors or typed submission closures.
    This interface hides native type parameters from the shared Window, not
    lifecycle, source capture, storage, operation or Controller ownership.
    """

    @property
    def name(self) -> str:
        """Return the declared native workflow name used for selection and RPC."""
        ...

    @property
    def plan_model(self) -> type[FormModel]:
        """Return the current independent Start plan form declaration."""
        ...

    @property
    def knobs_model(self) -> type[FormModel]:
        """Return the current independent Start/accepted tunables form declaration."""
        ...

    def prepare(
        self,
        plan: FormModel,
        knobs: FormModel,
        accepted_input: FormInput[FormModel],
        env: FormEnv,
        snapshot: RunSnapshot,
    ) -> PreparedWorkflow[C] | FormErrors:
        """Prepare native inputs from validated current plan_model/knobs_model values.

        accepted_input is the exact knobs expression input producing knobs in the
        fixed env; snapshot captures historical entry/point/roles, not live state.
        Return consuming-field FormErrors or a PreparedWorkflow for one Start.
        Model mismatches raise TypeError; native converter/seed errors propagate.
        No resource capture, storage write, admission or native execution occurs.
        """
        ...


@dataclass(frozen=True)
class WorkflowAttachment[
    P: BaseModel,
    T: BaseModel,
    Summary: BaseModel | None,
    FP: FormModel,
    FT: FormModel,
    C,
]:
    """Register a workflow with independent plan/tunables forms for the generic host.

    P/T/Summary are native plan, tunables and optional summary types. C is the
    opaque workflow context; the host only passes the seed into EnginePorts.
    FP/FT are independent FormModel declarations, not native numeric models.

    script is a coroutine already declared with @workflow. plan_model/knobs_model
    are its current plan/tunables form classes. plan_to_cfg(form, env, snapshot)
    converts an FP instance using only the fixed FormEnv and historical
    RunSnapshot; return native P or consuming-field FormErrors. knobs_to_cfg(form)
    converts validated FT into native T for both Start and every Apply.
    seed(native_plan, plan_form) returns C or
    None for a context-independent workflow. Converters and seed must not read
    live sources or acquire resources; their failures propagate at prepare/preview.
    These fields must all describe one coherent source revision. Constructor does
    not evaluate forms, invoke converters or acquire resources.

    Build WorkflowAttachment(script=my_script, plan_model=PlanForm,
    knobs_model=TunablesForm, plan_to_cfg=PlanForm.to_cfg,
    knobs_to_cfg=TunablesForm.to_cfg, seed=my_seed), then pass a nonempty ordered
    list of these registrations to WorkflowControlFacet.configure(...).
    Different native plan/tunables/summary types remain paired inside prepare's
    closures; registrations in one host share C. No export/presentation callback,
    run, storage owner or live source is retained.
    """

    script: WorkflowScript[P, T, Summary, C]
    plan_model: type[FP]
    knobs_model: type[FT]
    plan_to_cfg: Callable[[FP, FormEnv, RunSnapshot], P | FormErrors]
    knobs_to_cfg: Callable[[FT], T]
    seed: Callable[[P, FP], C | None]

    @property
    def name(self) -> str:
        """Return the name from the existing public native declaration."""
        return workflow_definition(self.script).name

    def _plan(
        self, form: FormModel, env: FormEnv, snapshot: RunSnapshot
    ) -> P | FormErrors:
        if not isinstance(form, self.plan_model):
            raise TypeError("Plan form does not belong to this attachment")
        return self.plan_to_cfg(form, env, snapshot)

    def _knobs(self, form: FormModel) -> T:
        if not isinstance(form, self.knobs_model):
            raise TypeError("Tunables form does not belong to this attachment")
        return self.knobs_to_cfg(form)

    def prepare(
        self,
        plan: FormModel,
        knobs: FormModel,
        accepted_input: FormInput[FormModel],
        env: FormEnv,
        snapshot: RunSnapshot,
    ) -> PreparedWorkflow[C] | FormErrors:
        """Resolve forms and retain native type pairing in submission closures.

        plan/knobs must belong to this current attachment; mismatches raise
        TypeError. Native conversion errors propagate or return FormErrors.
        accepted_input is the exact knobs expression input producing knobs in
        fixed env, not a second draft. snapshot is historical entry/point/roles.
        No sources/storage/hardware are read or captured beyond supplied env.
        The returned submit delegates directly to Controller.start once.
        """
        if not isinstance(plan, self.plan_model) or not isinstance(
            knobs, self.knobs_model
        ):
            raise TypeError("Forms do not belong to this attachment")
        native_plan = self.plan_to_cfg(plan, env, snapshot)
        if isinstance(native_plan, FormErrors):
            return native_plan
        native_tunables = self._knobs(knobs)
        context = self.seed(native_plan, plan)

        def submit(
            controller: WorkflowController[FormModel, C],
            ports: EnginePorts[C],
            paths: RunPaths,
            identity: RunIdentity,
            capture: FormEnv,
        ) -> int:
            """Submit these prepared inputs once to the fixed host controller.

            ports/paths/identity/capture follow PreparedWorkflow's contract.
            Create one Host over accepted_input and return its pending token.
            Native/admission failures propagate without retry or new capture.
            """
            binding = TunablesHost[FormModel, C](
                FormState(accepted_input), capture, identity.run_id, to_cfg=self._knobs
            )
            return controller.start(
                self.script,
                plan=native_plan,
                tunables=native_tunables,
                paths=paths,
                identity=identity,
                ports=ports,
                binding=binding,
            )

        return PreparedWorkflow(context, submit)
