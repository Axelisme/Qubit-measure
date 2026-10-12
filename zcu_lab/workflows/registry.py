"""Lab workflow registrations consumed by the generic GUI host."""

from zcu_tools.experiment.workflows.declaration import WorkflowRegistry
from zcu_tools.gui.app.workflows.attachment import Attachment, WorkflowAttachment

from zcu_lab.workflows.coherence_fluxdep.form import CoherencePlanForm
from zcu_lab.workflows.coherence_fluxdep.workflow import CoherencePlan
from zcu_lab.workflows.common.calibration import Calibrated


def calibrated_seed(plan: CoherencePlan, form: CoherencePlanForm) -> Calibrated:
    """Build Coherence scalar context from fixed modules and required form pi estimate.

    No calibration or live source read occurs. Non-scalar module frequency/gain
    raises ValueError; pi estimate is in microseconds and already form-validated.
    """
    from zcu_lab.workflows.common.calibration_modules import calibration_seed

    return calibration_seed(plan.qub_pulse, plan.readout, form.pi_length_us)


def build_registry() -> tuple[Attachment[Calibrated], ...]:
    """Pair current native workflows/forms without constructing live resources.

    Coherence is first/default; MIST is a gain scan; overnight is fixed-pair
    monitoring without feedback. Registry validates the declarations locally.
    Source reload uses this builder; current run bindings remain unchanged.
    Coherence has no temporary export bridge; independent tools own conversion.
    """
    from zcu_lab.workflows.coherence_fluxdep.form import (
        CoherencePlanForm,
        CoherenceTunablesForm,
    )
    from zcu_lab.workflows.coherence_fluxdep.workflow import coherence_fluxdep
    from zcu_lab.workflows.common.calibration_modules import calibration_seed
    from zcu_lab.workflows.mist_fluxdep.form import MISTPlanForm, MISTTunablesForm
    from zcu_lab.workflows.mist_fluxdep.workflow import mist_fluxdep
    from zcu_lab.workflows.overnight.form import (
        OvernightPlanForm,
        OvernightTunablesForm,
    )
    from zcu_lab.workflows.overnight.workflow import overnight

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
    )
    monitoring = WorkflowAttachment(
        step=overnight,
        plan_model=OvernightPlanForm,
        knobs_model=OvernightTunablesForm,
        plan_to_cfg=OvernightPlanForm.to_cfg,
        knobs_to_cfg=OvernightTunablesForm.to_cfg,
        seed=lambda _plan, _form: None,
    )
    return (coherence, mist, monitoring)
