"""coherence_fluxdep saving helpers for lab workflows."""

from pathlib import Path

from zcu_tools.datafile import RunSnapshot
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.workflows import Completed

from zcu_lab.v2.twotone.time_domain.t2ramsey.core import (
    T2RamseyCfg,
    T2RamseyExp,
    T2RamseyResult,
)
from zcu_lab.workflows.coherence_fluxdep.acquire import T2RamseyAcquisition
from zcu_lab.workflows.common.saving import save_native


def save_t2ramsey(
    source: Completed[T2RamseyCfg, T2RamseyAcquisition],
    destination: Path,
    *,
    snapshot: RunSnapshot,
) -> None:
    """Write Ramsey IQ and fixed evidence to the workflow's exact native path.

    source supplies the completed cfg, rounded microsecond delays, complex IQ
    and acquisition/device evidence; snapshot supplies historical entry/roles.
    Use native twotone/ge/t2ramsey declaration and seconds-on-disk coordinates.
    Parent must exist and destination must not. Native validation, collision
    and I/O errors propagate; no capture, layout choice or retry occurs.
    """
    record: RunRecord[T2RamseyCfg, T2RamseyResult] = RunRecord(
        source.cfg, source.result
    )
    save_native(
        record,
        destination,
        spec=T2RamseyExp().persistence_spec(),
        snapshot=snapshot,
        evidence=source.result.evidence,
    )
