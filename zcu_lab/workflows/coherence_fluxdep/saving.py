"""Save native t2ramsey records with fixed workflow operation facts."""

from pathlib import Path

from zcu_tools.datafile import RunSnapshot
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.workflows import Completed, RunIdentity

from zcu_lab.v2.twotone.time_domain.t2ramsey.core import (
    T2RamseyCfg,
    T2RamseyExp,
    T2RamseyResult,
)
from zcu_lab.workflows.common.saving import save_native


def save_t2ramsey(
    source: Completed[RunRecord[T2RamseyCfg, T2RamseyResult]],
    destination: Path,
    *,
    snapshot: RunSnapshot,
    identity: RunIdentity,
) -> None:
    """Preserve raw t2ramsey IQ and its native rounded coordinate.

    source is env.run's completed core record and operation facts. Pass the same
    run's env.identity and historical snapshot. destination is an exact new
    native filename; its parent must exist. Schema, collision and I/O errors
    propagate without recapture, fitting or retry.
    """
    save_native(
        source,
        destination,
        spec=T2RamseyExp().persistence_spec(),
        snapshot=snapshot,
        identity=identity,
    )
