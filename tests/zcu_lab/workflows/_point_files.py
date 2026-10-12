"""Test assertions over workflow-owned, explicitly named point file references."""

from pydantic import JsonValue
from zcu_tools.experiment.workflows import CommittedRecord


def filename(value: JsonValue) -> tuple[str, ...]:
    """Require a supplied relative reference, or retain a missing-file outcome."""
    if value is None:
        return ()
    assert isinstance(value, str)
    return (value,)


def spectroscopy_files(value: JsonValue) -> tuple[str, ...]:
    """Read all explicitly named Completed attempts in their recorded order."""
    assert isinstance(value, list)
    return tuple(name for item in value for name in filename(item))


def aggregate_files(point: dict[str, JsonValue]) -> tuple[str, ...]:
    """References in the scalar carrier still consumed by old conversion tests."""
    value = point.get("calibration_files")
    if value is None:
        return ()
    assert isinstance(value, dict)
    return (
        spectroscopy_files(value["frequency"])
        + filename(value["rabi"])
        + filename(value["readout"])
    )


def stage_files(point: dict[str, JsonValue]) -> tuple[str, ...]:
    """References in Coherence/MIST's independently named calibration results."""
    if "frequency" not in point:
        return ()
    frequency = point["frequency"]
    assert isinstance(frequency, dict)
    refs = spectroscopy_files(frequency["files"])
    for kind in ("rabi", "readout"):
        stage = point[kind]
        assert isinstance(stage, dict)
        refs += filename(stage["file"])
    return refs


def lifetime_files(point: dict[str, JsonValue]) -> tuple[str, ...]:
    """References in the ordered enabled lifetime tuple, without discovery."""
    members = point.get("measurements", [])
    assert isinstance(members, list)
    refs: tuple[str, ...] = ()
    for member in members:
        assert isinstance(member, dict)
        refs += filename(member["file"])
    return refs


def point_files(record: CommittedRecord) -> tuple[str, ...]:
    """Collect references from declared point fields for native-reader assertions.

    This reads only encoded scalar summaries, never the filesystem. Calibration
    attempts precede decay/gain/pair members for assertion convenience; this is
    not a framework discovery or storage contract.
    """
    point = record.encoded_record.value
    assert isinstance(point, dict)
    refs = aggregate_files(point) + stage_files(point) + lifetime_files(point)
    for purpose in ("decay_file", "gain_file", "t1_file", "t2_file"):
        refs += filename(point.get(purpose))
    return refs
