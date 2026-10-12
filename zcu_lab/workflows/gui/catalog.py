"""App-owned current attachments; native Registry only validates declarations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from zcu_tools.experiment.workflows.declaration import WorkflowRegistry
from zcu_tools.gui.expected_error import InvalidInputError

from .attachment import Attachment, WorkflowAttachment

if TYPE_CHECKING:
    from ..calibration import Calibrated
    from ..coherence import CoherencePlan
    from .plan import CoherencePlanForm


@dataclass(frozen=True)
class WorkflowCatalog:
    """Current coherent attachments in selector order, without run/draft state.

    attachments pairs native science with independent current forms. The first
    attachment is the default for Start without name (shipped coherence_fluxdep).
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


def calibrated_seed(plan: CoherencePlan, form: CoherencePlanForm) -> Calibrated:
    """Build Coherence scalar context from fixed modules and required form pi estimate.

    No calibration or live source read occurs. Non-scalar module frequency/gain
    raises ValueError; pi estimate is in microseconds and already form-validated.
    """
    from .calibration_modules import calibration_seed

    return calibration_seed(plan.qub_pulse, plan.readout, form.pi_length_us)


def build_catalog() -> WorkflowCatalog:
    """Pair current native workflows/forms without constructing live resources.

    Coherence is first/default; MIST is a gain scan; overnight is fixed-pair
    monitoring without feedback. Registry validates the declarations locally.
    Source reload uses this builder; current run bindings remain unchanged.
    Coherence has no temporary export bridge; independent tools own conversion.
    """
    from ..coherence import coherence_fluxdep
    from ..export import prepare_mist_fluxdep_export
    from ..mist import mist_fluxdep
    from ..overnight import overnight
    from .calibration_modules import calibration_seed
    from .coherence import CoherenceTunablesForm
    from .mist import MISTPlanForm, MISTTunablesForm
    from .overnight import OvernightPlanForm, OvernightTunablesForm
    from .plan import CoherencePlanForm

    registry = WorkflowRegistry()
    registry.add(coherence_fluxdep)
    registry.add(mist_fluxdep)
    registry.add(overnight)
    coherence = WorkflowAttachment(
        step=coherence_fluxdep,
        plan_model=CoherencePlanForm,
        knobs_model=CoherenceTunablesForm,
        plan_to_cfg=CoherencePlanForm.to_cfg,
        knobs_to_cfg=CoherenceTunablesForm.to_cfg,
        seed=calibrated_seed,
        prepare_export=None,
        samples_reason="Coherence conversion is provided by independent run tools",
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
        samples_reason="MIST has no lifetime sample science fields",
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
    return WorkflowCatalog((coherence, mist, monitoring))
