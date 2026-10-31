"""Save workflow acquisitions using core records and framework operation facts."""

from pathlib import Path
from typing import Literal

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment.axes_spec import AxesSpec, GroupedAxesSpec
from zcu_tools.experiment.metadata import RunSnapshot
from zcu_tools.experiment.native_persistence import save_run
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.workflows import Completed, RunIdentity, record_with_metadata


def save_native[Cfg: ConfigBase, Result](
    source: Completed[RunRecord[Cfg, Result]],
    destination: Path,
    *,
    spec: AxesSpec[Result, Cfg] | GroupedAxesSpec[Result, Cfg],
    snapshot: RunSnapshot,
    identity: RunIdentity,
    completion: Literal["complete", "partial", "stopped", "unknown"],
) -> None:
    """Save a completed core record with its fixed workflow operation facts.

    source.value supplies cfg/result; source.facts supplies UTC boundaries,
    iteration and attempt. source.value.metadata retains the pre-core device
    snapshot. completion is the caller's explicit acquisition completeness
    (complete, partial, stopped or unknown), never inferred from control bits.
    identity is env.identity for that workflow run; snapshot is the caller's
    historical entry/point capture. spec supplies the experiment tag and native
    array mapping. Compose existing RunMetadata without capturing new facts.
    destination is an exact new filename whose parent exists. Native schema,
    metadata, collision and I/O errors propagate; no overwrite or retry occurs.
    """
    record = record_with_metadata(
        source,
        identity=identity,
        experiment_tag=spec.tag,
        snapshot=snapshot,
        completion=completion,
    )
    save_run(record, destination, spec=spec)
