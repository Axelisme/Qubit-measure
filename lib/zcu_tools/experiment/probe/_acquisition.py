"""Shared acquisition boundary for the independent and legacy offline probes."""

from datetime import datetime, timezone
from typing import TypeVar

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.plotting.plots import Plots

from ..acquisition import (
    capture_device_snapshot,
    capture_software_provenance,
    new_run_id,
)
from ..base import Experiment
from ..context import RunContext
from ..metadata import RunMetadata, RunSnapshot
from ..records import RunRecord
from ..scope import Scope
from ..simulation import SimulationEnvironment

_Cfg = TypeVar("_Cfg", bound=ConfigBase)
_Result = TypeVar("_Result")


def acquire_record(
    experiment: Experiment[_Result, _Cfg],
    cfg: _Cfg,
    environment: SimulationEnvironment,
    plots: Plots,
) -> RunRecord[_Cfg, _Result]:
    """Acquire one record with fresh Scope and caller-owned hardware and plots.

    experiment must declare AXES_SPEC and run cfg's concrete model type.
    Pass a deep cfg copy to run, preserving the caller's cfg in the returned
    RunRecord. Capture software provenance, connected device settings and UTC
    start/finish times. Completion is stopped when this run's Scope is cancelled,
    otherwise complete. Missing AXES_SPEC raises ValueError.
    environment supplies soc, soccfg and devices; never close or rewire them.
    plots receives live figures; its caller owns finish/release. Scope is released
    on return or failure. Capture/run failures propagate without making a record.
    No shape validation, persistence, analysis or figure export occurs here.
    """
    with Scope("run") as scope:
        context = RunContext(
            environment.soc,
            environment.soccfg,
            plots,
            environment.devices.get_all_devices(),
            scope,
        )
        provenance = capture_software_provenance(context=context)
        devices = capture_device_snapshot(context.devices)
        started = datetime.now(timezone.utc)
        result = experiment.run(cfg.model_copy(deep=True), context=context)
        finished = datetime.now(timezone.utc)
        spec = experiment.AXES_SPEC
        if spec is None:
            raise ValueError("Probe requires experiment.AXES_SPEC")
        return RunRecord(
            cfg,
            result,
            RunMetadata(
                run_id=new_run_id(started),
                experiment=spec.tag,
                started_at=started.isoformat(),
                finished_at=finished.isoformat(),
                completion="stopped" if scope.cancelled else "complete",
                snapshot=RunSnapshot(roles={}),
                provenance=provenance,
                device_snapshot=devices,
            ).model_dump(mode="json"),
        )

