"""Current typed attachments and Start's science callback binding.

Imports reloadable declarations at call time. Binding captures Start's detached
plan/root and storage for the Controller, without retaining any run/live resource
in this module or catalog, and without orchestrating writers.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from qtpy.QtWidgets import QWidget
from zcu_tools.datafile import SoftwareProvenance
from zcu_tools.experiment.workflows import (
    CommittedRecord,
    FluxdepExportSnapshot,
    PreparedFluxdepExport,
    RunIdentity,
    RunPaths,
)
from zcu_tools.experiment.workflows.declaration import WorkflowRegistry, WorkflowStep
from zcu_tools.gui.app.workflows.exporting import FluxdepExportBinding
from zcu_tools.gui.form import FormEnv
from zcu_tools.resources.entry import PointStorageBinding, WorkflowRoots

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


def build_export_binding(
    plan: T1Plan,
    storage: PointStorageBinding,
    run_id: str,
    provenance: SoftwareProvenance,
    roots: WorkflowRoots,
    catalog: WorkflowCatalog,
) -> FluxdepExportBinding | None:
    """Detach native plan and bind this Start's catalog callback and exact storage.

    App-internal assembly helper, not an export entry. plan is the validated
    native plan; storage is Start's exact entry/point paths and scoped ledger.
    run_id/provenance identify this run; roots are its exact workflow directories.
    catalog is the current science/attachment pairing. None prepare_export leaves
    export unsupported. Fix identity/paths/UTC start time and historical snapshot
    without writes or source/hardware reads. Native
    validation errors propagate; Controller owns the returned run binding. Later
    reload/draft changes cannot rebind this callback's plan/root or ledger.
    """
    if catalog.prepare_export is None:
        return None
    identity = RunIdentity(run_id, provenance, str(storage.entry_id), storage.point)
    paths = RunPaths(metadata_root=roots.result, data_root=roots.database)
    fixed_plan = type(plan).model_validate_json(
        plan.model_dump_json(),
        by_name=True,
        by_alias=False,
    )
    return FluxdepExportBinding(
        FluxdepExportSnapshot(
            identity, paths, fixed_plan.snapshot, datetime.now(UTC), None
        ),
        partial(catalog.prepare_export, plan=fixed_plan, data_root=paths.data_root),
        storage.result_path,
        storage.database_path,
        storage.ledger,
    )
