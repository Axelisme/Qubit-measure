"""Native history for tests that isolate cfg projection or presentation."""

from uuid import UUID

from zcu_tools.datafile import RunMetadata, RunSnapshot, SoftwareProvenance


def historical_metadata() -> RunMetadata:
    """Return fixed native history, without storage or live context capture."""
    return RunMetadata(
        run_id="20261008T000000Z-a1b2c3",
        experiment="test/load",
        started_at="2026-10-08T00:00:00+00:00",
        finished_at="2026-10-08T00:01:00+00:00",
        completion="complete",
        labber_path=None,
        snapshot=RunSnapshot(
            entry_id=UUID("00000000-0000-4000-8000-000000000001"),
            entry_name="historical",
            point="original-point",
            description="",
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
