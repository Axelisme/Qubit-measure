"""Save native mist records with fixed workflow operation facts."""

from pathlib import Path

from zcu_tools.experiment.metadata import RunSnapshot
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.workflows import Completed, RunIdentity

from zcu_lab.v2.mist.power_dep.single_trace.core import (
    PowerDepCfg,
    PowerDepExp,
    PowerDepResult,
)
from zcu_lab.workflows.common.saving import save_native


def save_mist(
    source: Completed[RunRecord[PowerDepCfg, PowerDepResult]],
    destination: Path,
    *,
    snapshot: RunSnapshot,
    identity: RunIdentity,
) -> None:
    """Preserve raw mist IQ and its native rounded coordinate.

    source is env.run's completed core record and operation facts. Pass the same
    run's env.identity and historical snapshot. destination is an exact new
    native filename; its parent must exist. Schema, collision and I/O errors
    propagate without recapture, fitting or retry.
    """
    save_native(
        source,
        destination,
        spec=PowerDepExp().persistence_spec(),
        snapshot=snapshot,
        identity=identity,
    )
