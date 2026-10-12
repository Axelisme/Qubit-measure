"""App-owned current attachments; native Registry only validates declarations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from zcu_tools.experiment.workflows.declaration import WorkflowRegistry
from zcu_tools.gui.expected_error import InvalidInputError

from .attachment import Attachment, WorkflowAttachment

if TYPE_CHECKING:
    from ..calibration import Calibrated
    from ..t1 import T1Plan
    from .plan import T1PlanForm


@dataclass(frozen=True)
class WorkflowCatalog:
    """Current coherent attachments in selector order, without run/draft state.

    attachments pairs native science with independent current forms. The first
    attachment is the default for Start without name (shipped T1_fluxdep).
    Names must be unique and at least one attachment must exist. The app selector
    and optional Start name consume this same pairing; Registry has no lookup.
    New Starts use current callables; Controller's existing export binding stays
    fixed across catalog replacement. No live resources are owned here.
    """

    attachments: tuple[Attachment, ...]

    def __post_init__(self) -> None:
        """Reject empty catalogs or duplicate declared names before publication."""
        if not self.attachments:
            raise ValueError("Workflow catalog must contain an attachment")
        names = self.names
        if len(names) != len(set(names)):
            raise ValueError("Workflow catalog names must be unique")

    @property
    def names(self) -> tuple[str, ...]:
        """Return selector names in app-declared order, without Registry lookup."""
        return tuple(attachment.name for attachment in self.attachments)

    def select(self, name: str | None = None) -> Attachment:
        """Return a current pairing; None selects default, unknown name raises InvalidInputError."""
        if name is None:
            return self.attachments[0]
        for attachment in self.attachments:
            if attachment.name == name:
                return attachment
        raise InvalidInputError(
            f"Unknown workflow: {name}", reason_code="unknown_workflow"
        )


def calibrated_seed(plan: T1Plan, form: T1PlanForm) -> Calibrated:
    """Build T1 scalar context from fixed modules and required form pi estimate.

    No calibration or live source read occurs. Non-scalar module frequency/gain
    raises ValueError; pi estimate is in microseconds and already form-validated.
    """
    from .calibration_modules import calibration_seed

    return calibration_seed(plan.qub_pulse, plan.readout, form.pi_length_us)


def build_catalog() -> WorkflowCatalog:
    """Import current native/form declarations and their coherent typed attachments.

    Use on the app owner, inside bounded source reload. Fresh Registry validates
    both declarations without publishing global state or adding lookup. T1 is
    first/default; overnight is context-independent and has no fluxdep exporter.
    Import/declaration errors propagate; no session, Engine or device is built.
    """
    from ..export import (
        prepare_mist_fluxdep_export,
        prepare_t1_fluxdep_export,
        prepare_t2echo_fluxdep_export,
        prepare_t2ramsey_fluxdep_export,
    )
    from ..mist import mist_fluxdep
    from ..overnight import overnight
    from ..samples import (
        export_t1_samples,
        export_t2echo_samples,
        export_t2ramsey_samples,
    )
    from ..t1 import t1_fluxdep
    from ..t2 import t2echo_fluxdep, t2ramsey_fluxdep
    from .calibration_modules import calibration_seed
    from .mist import MISTPlanForm, MISTTunablesForm
    from .overnight import OvernightPlanForm, OvernightTunablesForm
    from .plan import T1PlanForm
    from .t2 import T2EchoTunablesForm, T2PlanForm, T2RamseyTunablesForm
    from .tunables import T1TunablesForm

    registry = WorkflowRegistry()
    registry.add(t1_fluxdep)
    registry.add(overnight)
    registry.add(t2echo_fluxdep)
    registry.add(t2ramsey_fluxdep)
    registry.add(mist_fluxdep)
    t1 = WorkflowAttachment(
        step=t1_fluxdep,
        plan_model=T1PlanForm,
        knobs_model=T1TunablesForm,
        plan_to_cfg=T1PlanForm.to_cfg,
        knobs_to_cfg=T1TunablesForm.to_cfg,
        seed=calibrated_seed,
        prepare_export=prepare_t1_fluxdep_export,
        prepare_samples=export_t1_samples,
    )
    monitoring = WorkflowAttachment(
        step=overnight,
        plan_model=OvernightPlanForm,
        knobs_model=OvernightTunablesForm,
        plan_to_cfg=OvernightPlanForm.to_cfg,
        knobs_to_cfg=OvernightTunablesForm.to_cfg,
        seed=lambda _plan, _form: None,
        prepare_export=None,
        samples_reason="Overnight monitors fixed-flux rounds, not a flux scan",
    )
    echo = WorkflowAttachment(
        step=t2echo_fluxdep,
        plan_model=T2PlanForm,
        knobs_model=T2EchoTunablesForm,
        plan_to_cfg=T2PlanForm.to_cfg,
        knobs_to_cfg=T2EchoTunablesForm.to_cfg,
        seed=lambda plan, form: calibration_seed(
            plan.qub_pulse, plan.readout, form.pi_length_us
        ),
        prepare_export=prepare_t2echo_fluxdep_export,
        prepare_samples=export_t2echo_samples,
    )
    ramsey = WorkflowAttachment(
        step=t2ramsey_fluxdep,
        plan_model=T2PlanForm,
        knobs_model=T2RamseyTunablesForm,
        plan_to_cfg=T2PlanForm.to_cfg,
        knobs_to_cfg=T2RamseyTunablesForm.to_cfg,
        seed=lambda plan, form: calibration_seed(
            plan.qub_pulse, plan.readout, form.pi_length_us
        ),
        prepare_export=prepare_t2ramsey_fluxdep_export,
        prepare_samples=export_t2ramsey_samples,
    )
    mist = WorkflowAttachment(
        step=mist_fluxdep,
        plan_model=MISTPlanForm,
        knobs_model=MISTTunablesForm,
        plan_to_cfg=MISTPlanForm.to_cfg,
        knobs_to_cfg=MISTTunablesForm.to_cfg,
        seed=lambda plan, form: calibration_seed(
            plan.qub_pulse, plan.readout, form.pi_length_us
        ),
        prepare_export=prepare_mist_fluxdep_export,
        samples_reason="MIST has calibrated frequency/success, but no sample science fields",
    )
    return WorkflowCatalog((t1, monitoring, echo, ramsey, mist))
