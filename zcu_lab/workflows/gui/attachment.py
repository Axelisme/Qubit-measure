"""App-local native/form pairing; Controller remains the only run owner."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel
from qtpy.QtWidgets import QWidget
from zcu_tools.datafile import RunSnapshot
from zcu_tools.experiment.workflows import (
    CommittedRecord,
    EnginePorts,
    FluxdepExportSnapshot,
    PreparedFluxdepExport,
    RunIdentity,
    RunPaths,
)
from zcu_tools.experiment.workflows.declaration import WorkflowStep, workflow_definition
from zcu_tools.experiment.workflows.exporting import SampleRow, SampleRun
from zcu_tools.gui.app.workflows.controller import WorkflowController
from zcu_tools.gui.app.workflows.exporting import (
    FluxdepExportBinding,
    WorkflowExportBindings,
)
from zcu_tools.gui.app.workflows.samples import SampleExportBinding
from zcu_tools.gui.app.workflows.tunables import TunablesHost
from zcu_tools.gui.form import FormEnv, FormErrors, FormInput, FormModel, FormState
from zcu_tools.resources.entry import PointStorageBinding, WorkflowRoots
from zcu_tools.resources.entry.storage import bind_sample_table

from ..calibration import Calibrated

if TYPE_CHECKING:
    from .panel import WorkflowFormPanel, WorkflowPlanPanel


class WorkflowExportPreparer[P: BaseModel](Protocol):
    """Science callback for one paired native plan type and committed records.

    plan is Start's detached native plan; data_root is its exact acquisition root.
    records are journal-authoritative Engine evidence. Return a pure payload or
    raise before output writes; no fitting, hardware or host admission occurs.
    """

    def __call__(
        self, *, plan: P, data_root: Path, records: tuple[CommittedRecord, ...]
    ) -> PreparedFluxdepExport:
        """Prepare native/Labber data without writing or recapturing sources."""
        ...


@dataclass(frozen=True)
class PreparedWorkflow:
    """A validated native/form pairing ready for Window's captured Start.

    context is T1's scalar seed or None for context-independent science. submit
    calls the supplied fixed Controller with typed native inputs and a pending
    Host over accepted input. Window supplies its ports, paths, identity, env and
    optional export binding; no capture/admission policy is duplicated here.
    bind_export fixes this pairing's native plan/science callback with Window's
    captured storage, identity and roots, or returns None if unsupported.
    bind_samples captures the exact table/component/point and artifact producer
    or explicit unsupported reason. bind_exports groups these two capabilities
    for Start identity checking. No callable owns a run, lease or live source.
    """

    context: Calibrated | None
    submit: Callable[
        [
            WorkflowController[FormModel, Calibrated],
            EnginePorts[Calibrated],
            RunPaths,
            RunIdentity,
            FormEnv,
            WorkflowExportBindings,
        ],
        int,
    ]
    bind_export: Callable[
        [PointStorageBinding, RunIdentity, WorkflowRoots], FluxdepExportBinding | None
    ]
    bind_samples: Callable[
        [PointStorageBinding, RunIdentity, WorkflowRoots], SampleExportBinding
    ]

    def bind_exports(
        self, storage: PointStorageBinding, identity: RunIdentity, roots: WorkflowRoots
    ) -> WorkflowExportBindings:
        """Capture both export capabilities from this prepared Start's fixed inputs.

        storage/identity/roots are the host capture, never current source selection.
        Delegates to the paired bind_export/bind_samples closures without writes;
        returns their grouped capabilities for Controller.start identity checking.
        """
        return WorkflowExportBindings(
            self.bind_export(storage, identity, roots),
            self.bind_samples(storage, identity, roots),
        )


class Attachment(Protocol):
    """Common app view of a coherent typed native/form attachment.

    name is the native declaration name. Models are the current independent form
    declarations; panels render those models in a fixed capture. prepare accepts
    their validated ordinary instances plus accepted knobs expression input and
    a host snapshot. It returns directed errors or typed submission closures.
    This interface hides native type parameters from the shared Window, not
    lifecycle, source capture, storage, operation or Controller ownership.
    """

    @property
    def name(self) -> str: ...
    @property
    def samples_unsupported_reason(self) -> str | None:
        """Return the current science reason, or None when samples are supported."""
        ...

    @property
    def plan_model(self) -> type[FormModel]: ...
    @property
    def knobs_model(self) -> type[FormModel]: ...

    def plan_panel(self, env: FormEnv, parent: QWidget | None) -> WorkflowPlanPanel:
        """Build a partial plan view without capture or native execution."""
        ...

    def knobs_panel(
        self, state: FormState[FormModel], env: FormEnv, parent: QWidget | None
    ) -> WorkflowFormPanel[FormModel]:
        """Render the supplied unique draft without copying or resetting it."""
        ...

    def prepare(
        self,
        plan: FormModel,
        knobs: FormModel,
        accepted_input: FormInput[FormModel],
        env: FormEnv,
        snapshot: RunSnapshot,
    ) -> PreparedWorkflow | FormErrors:
        """Convert current validated forms into native typed submission closures."""
        ...


@dataclass(frozen=True)
class WorkflowAttachment[
    P: BaseModel,
    T: BaseModel,
    S,
    R,
    FP: FormModel,
    FT: FormModel,
]:
    """Pair a native step with independent plan/knobs forms and converters.

    P/T/S/R pair the step with native plan, tunables, state and record. FP pairs
    plan_model with plan_to_cfg and seed; FT pairs knobs_model with knobs_to_cfg.
    step is already declared via workflow. plan_to_cfg resolves fixed captured
    modules and snapshot, returning consuming-field errors. knobs_to_cfg converts
    ordinary validated values. seed returns T1 scalar calibration or None without
    live reads. prepare_export is the matching science callback or None for
    unsupported workflows. prepare_samples is a pure artifact callback or None;
    samples_reason explains unsupported science, ignored when a producer exists.
    All fields must be a coherent pairing supplied by the
    app builder. Panels use the shared FormModel view, but native types stay paired
    inside prepare's closures. No run, storage owner or resource is retained.
    """

    step: WorkflowStep[P, T, S, R, Calibrated]
    plan_model: type[FP]
    knobs_model: type[FT]
    plan_to_cfg: Callable[[FP, FormEnv, RunSnapshot], P | FormErrors]
    knobs_to_cfg: Callable[[FT], T]
    seed: Callable[[P, FP], Calibrated | None]
    prepare_export: WorkflowExportPreparer[P] | None
    prepare_samples: Callable[[SampleRun[P]], tuple[SampleRow, ...]] | None = None
    samples_reason: str = "Workflow has no sample science fields"

    @property
    def samples_unsupported_reason(self) -> str | None:
        """Return the declared unsupported reason, or None for a samples producer."""
        return self.samples_reason if self.prepare_samples is None else None

    @property
    def name(self) -> str:
        """Return the name from the existing public native declaration."""
        return workflow_definition(self.step).name

    def _plan(
        self, form: FormModel, env: FormEnv, snapshot: RunSnapshot
    ) -> P | FormErrors:
        if not isinstance(form, self.plan_model):
            raise TypeError("Plan form does not belong to this attachment")
        return self.plan_to_cfg(form, env, snapshot)

    def plan_panel(self, env: FormEnv, parent: QWidget | None) -> WorkflowPlanPanel:
        """Build this model's partial draft with its paired native preview."""
        from .panel import WorkflowPlanPanel

        return WorkflowPlanPanel(self.plan_model, self._plan, env, parent)

    def knobs_panel(
        self, state: FormState[FormModel], env: FormEnv, parent: QWidget | None
    ) -> WorkflowFormPanel[FormModel]:
        """Render the supplied unique knobs draft in its fixed capture."""
        from .panel import WorkflowFormPanel

        return WorkflowFormPanel(state, env, parent)

    def prepare(
        self,
        plan: FormModel,
        knobs: FormModel,
        accepted_input: FormInput[FormModel],
        env: FormEnv,
        snapshot: RunSnapshot,
    ) -> PreparedWorkflow | FormErrors:
        """Resolve forms and retain native type pairing in submission closures.

        plan/knobs must belong to this current attachment; mismatches raise
        TypeError. Native conversion errors propagate or return FormErrors.
        accepted_input is the exact knobs expression input, not a second draft.
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
        native_tunables = self.knobs_to_cfg(knobs)
        context = self.seed(native_plan, plan)

        def submit(
            controller: WorkflowController[FormModel, Calibrated],
            ports: EnginePorts[Calibrated],
            paths: RunPaths,
            identity: RunIdentity,
            capture: FormEnv,
            export_bindings: WorkflowExportBindings,
        ) -> int:
            binding = TunablesHost[FormModel, Calibrated](
                FormState(accepted_input), capture, identity.run_id
            )
            return controller.start(
                self.step,
                plan=native_plan,
                tunables=native_tunables,
                paths=paths,
                identity=identity,
                ports=ports,
                binding=binding,
                export_bindings=export_bindings,
            )

        def bind_export(
            storage: PointStorageBinding, identity: RunIdentity, roots: WorkflowRoots
        ) -> FluxdepExportBinding | None:
            if self.prepare_export is None:
                return None
            fixed_plan = type(native_plan).model_validate_json(
                native_plan.model_dump_json(), by_name=True, by_alias=False
            )
            paths = RunPaths(metadata_root=roots.result, data_root=roots.database)
            return FluxdepExportBinding(
                FluxdepExportSnapshot(
                    identity, paths, snapshot, datetime.now(UTC), None
                ),
                partial(
                    self.prepare_export, plan=fixed_plan, data_root=paths.data_root
                ),
                storage.result_path,
                storage.database_path,
                storage.ledger,
            )

        def bind_samples(
            storage: PointStorageBinding, identity: RunIdentity, roots: WorkflowRoots
        ) -> SampleExportBinding:
            fixed_plan = type(native_plan).model_validate_json(
                native_plan.model_dump_json(), by_name=True, by_alias=False
            )
            producer = self.prepare_samples

            def prepare_rows(
                records: tuple[CommittedRecord, ...],
            ) -> tuple[SampleRow, ...]:
                if producer is None:
                    raise ValueError("Unsupported samples producer")
                return producer(
                    SampleRun(identity.run_id, fixed_plan, roots.database, records)
                )

            return SampleExportBinding(
                identity.run_id,
                snapshot.point,
                bind_sample_table(storage.result_path, snapshot.roles["qubit"]),
                None if producer is None else prepare_rows,
                self.samples_unsupported_reason,
            )

        return PreparedWorkflow(context, submit, bind_export, bind_samples)
