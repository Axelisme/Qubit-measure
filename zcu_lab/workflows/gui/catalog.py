"""Fixed typed attachment builder; imports reloadable declarations at call time."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from qtpy.QtWidgets import QWidget
from zcu_tools.experiment.workflows import CommittedRecord, PreparedFluxdepExport
from zcu_tools.experiment.workflows.declaration import WorkflowRegistry, WorkflowStep
from zcu_tools.gui.form import FormEnv

if TYPE_CHECKING:
    from ..calibration import Calibrated
    from ..t1 import T1Plan, T1Point, T1State, T1Tunables
    from .panel import T1PlanPanel, T1TunablesPanel
    from .plan import T1PlanForm
    from .tunables import T1TunablesForm


class WorkflowExportPreparer(Protocol):
    """Science callback shared by T1 and the other fluxdep attachment builders.

    plan is Start's detached native plan, data_root its exact run directory and
    records the Engine's committed evidence. Return one native/Labber payload;
    invalid/no usable evidence raises before either writer. No host admission,
    ledger, live source, hardware, fitting or file writes belong in this callback.
    """

    def __call__(
        self, *, plan: T1Plan, data_root: Path, records: tuple[CommittedRecord, ...]
    ) -> PreparedFluxdepExport:
        """Prepare the fixed plan and committed records without output writes."""
        ...


@dataclass(frozen=True)
class WorkflowCatalog:
    """One current typed T1 declaration and its matching GUI attachments.

    step is the declared native callable; plan_model/knobs_model are its input
    FormModel types, with explicit native to_cfg methods. plan_panel/knobs_panel
    build the corresponding Qt views in the supplied fixed FormEnv and parent.
    seed builds the scalar initial context from native plan and pi length in us.
    prepare_export is the same revision's science callback; None explicitly marks
    an unsupported workflow. Each Start fixes this callback with its detached
    native plan/root, so later reload cannot rebind an existing run's export.
    Builders must supply a coherent pairing; Registry validates names/declarations
    but does not provide a heterogeneous callable lookup. No live resources or
    mutable draft are owned by this frozen catalog.
    """

    step: WorkflowStep[T1Plan, T1Tunables, T1State, T1Point, Calibrated]
    plan_model: type[T1PlanForm]
    knobs_model: type[T1TunablesForm]
    plan_panel: Callable[[FormEnv, QWidget | None], T1PlanPanel]
    knobs_panel: type[T1TunablesPanel]
    seed: Callable[[T1Plan, float], Calibrated]
    prepare_export: WorkflowExportPreparer | None


def _calibrated_seed(plan: T1Plan, pi_length_us: float) -> Calibrated:
    from ..calibration import Calibrated

    frequency = plan.qub_pulse.freq
    readout = plan.readout.pulse_cfg
    if (
        not isinstance(frequency, float)
        or not isinstance(readout.freq, float)
        or not isinstance(readout.gain, float)
    ):
        raise ValueError("T1 initial calibration requires scalar module frequency/gain")
    return Calibrated(frequency, pi_length_us, readout.freq, readout.gain)


def build_catalog() -> WorkflowCatalog:
    """Import current native science/forms/panels and return their typed pairing.

    Run on the app owner thread, inside SourceCatalogLoader's bounded import for
    reload. Fresh Registry registration validates the native declaration, without
    duplicating lookup or publishing global state. Import/declaration errors
    propagate; no session, device, Engine or background work is constructed.
    """
    from ..export import prepare_t1_fluxdep_export
    from ..t1 import t1_fluxdep
    from .panel import T1PlanPanel, T1TunablesPanel
    from .plan import T1PlanForm
    from .tunables import T1TunablesForm

    registry = WorkflowRegistry()
    registry.add(t1_fluxdep)
    return WorkflowCatalog(
        step=t1_fluxdep,
        plan_model=T1PlanForm,
        knobs_model=T1TunablesForm,
        plan_panel=T1PlanPanel,
        knobs_panel=T1TunablesPanel,
        seed=_calibrated_seed,
        prepare_export=prepare_t1_fluxdep_export,
    )
