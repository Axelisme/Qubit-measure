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
    from ..calibration import Calibrated

    frequency = plan.qub_pulse.freq
    readout = plan.readout.pulse_cfg
    if (
        not isinstance(frequency, float)
        or not isinstance(readout.freq, float)
        or not isinstance(readout.gain, float)
    ):
        raise ValueError("T1 initial calibration requires scalar module frequency/gain")
    return Calibrated(frequency, form.pi_length_us, readout.freq, readout.gain)


def build_catalog() -> WorkflowCatalog:
    """Import current native/form declarations and their coherent typed attachments.

    Use on the app owner, inside bounded source reload. Fresh Registry validates
    both declarations without publishing global state or adding lookup. T1 is
    first/default; overnight is context-independent and has no fluxdep exporter.
    Import/declaration errors propagate; no session, Engine or device is built.
    """
    from ..export import prepare_t1_fluxdep_export
    from ..overnight import overnight
    from ..t1 import t1_fluxdep
    from .overnight import OvernightPlanForm, OvernightTunablesForm
    from .plan import T1PlanForm
    from .tunables import T1TunablesForm

    registry = WorkflowRegistry()
    registry.add(t1_fluxdep)
    registry.add(overnight)
    t1 = WorkflowAttachment(
        step=t1_fluxdep,
        plan_model=T1PlanForm,
        knobs_model=T1TunablesForm,
        plan_to_cfg=T1PlanForm.to_cfg,
        knobs_to_cfg=T1TunablesForm.to_cfg,
        seed=calibrated_seed,
        prepare_export=prepare_t1_fluxdep_export,
    )
    monitoring = WorkflowAttachment(
        step=overnight,
        plan_model=OvernightPlanForm,
        knobs_model=OvernightTunablesForm,
        plan_to_cfg=OvernightPlanForm.to_cfg,
        knobs_to_cfg=OvernightTunablesForm.to_cfg,
        seed=lambda _plan, _form: None,
        prepare_export=None,
    )
    return WorkflowCatalog((t1, monitoring))
