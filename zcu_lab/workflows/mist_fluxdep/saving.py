"""mist_fluxdep saving helpers for lab workflows."""

from pathlib import Path

from zcu_tools.datafile import RunSnapshot
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.workflows import Completed

from zcu_lab.v2.mist.power_dep.single_trace.core import (
    PowerDepCfg,
    PowerDepExp,
    PowerDepResult,
)
from zcu_lab.workflows.common.saving import save_native
from zcu_lab.workflows.mist_fluxdep.acquire import MISTAcquisition


def save_mist(
    source: Completed[PowerDepCfg, MISTAcquisition],
    destination: Path,
    *,
    snapshot: RunSnapshot,
) -> None:
    """Write raw MIST gain trace with native mist/gain schema to exact path.

    source owns actual gains/IQ/cfg and acquisition/device evidence; snapshot is
    the historical Start entry/roles capture. Parent must exist and destination
    must not. Native schema, collision and I/O errors propagate. This writer
    does not transform IQ, fit thresholds, choose paths or recapture metadata.
    """
    record: RunRecord[PowerDepCfg, PowerDepResult] = RunRecord(
        source.cfg, source.result
    )
    save_native(
        record,
        destination,
        spec=PowerDepExp().persistence_spec(),
        snapshot=snapshot,
        evidence=source.result.evidence,
    )
