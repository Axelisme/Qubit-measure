"""Native history for tests that isolate cfg projection or presentation."""

from dataclasses import replace
from pathlib import Path
from uuid import UUID

from zcu_tools.datafile import save
from zcu_tools.experiment.metadata import RunMetadata, RunSnapshot, SoftwareProvenance
from zcu_tools.experiment.native_persistence import prepare_run_data
from zcu_tools.experiment.records import RunRecord

from tests.experiment._saving_support import SavingCfg, SavingCore


def write_native_file(
    path: Path,
    *,
    tag: str = "test/load",
    gain: float = 1.0,
    cfg_schema_version: str | None = None,
) -> RunMetadata:
    """Write offline three-point native data with fixed history at an exact path."""
    core = SavingCore()
    cfg = SavingCfg(gain=gain)
    spec = replace(core.AXES_SPEC, tag=tag)
    if cfg_schema_version is not None:
        spec = replace(spec, cfg_schema_version=cfg_schema_version)
    prepared = prepare_run_data(
        RunRecord(cfg, core.run(cfg, context=None)),
        spec=spec,
        comment="history",
    )
    metadata = historical_metadata().model_copy(update={"experiment": tag})
    save(path, prepared.to_container(metadata))
    return metadata


def historical_metadata() -> RunMetadata:
    """Return fixed native history, without storage or live context capture."""
    return RunMetadata(
        run_id="20261008T000000Z-a1b2c3",
        experiment="test/load",
        started_at="2026-10-08T00:00:00+00:00",
        finished_at="2026-10-08T00:01:00+00:00",
        completion="complete",
        snapshot=RunSnapshot(
            entry_id=UUID("00000000-0000-4000-8000-000000000001"),
            point="original-point",
            roles={},
        ),
        provenance=SoftwareProvenance(
            software_versions={},
            git_commit=None,
            git_dirty=None,
            qick_version=None,
            soc_fingerprint=None,
            hostname=None,
        ),
    )
