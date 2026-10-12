"""Test assertions over workflow-owned, explicitly named point file references."""

from zcu_tools.experiment.workflows import CommittedRecord


def point_files(record: CommittedRecord) -> tuple[str, ...]:
    """Collect named native references from the shipped point schemas for tests.

    This reads only encoded scalar summaries, never the filesystem. Calibration
    attempts precede the named decay/gain/pair members for assertion convenience;
    this ordering is not a framework storage or discovery contract.
    """
    point = record.encoded_record.value
    assert isinstance(point, dict)
    result: list[str] = []
    if "calibration_files" in point and point["calibration_files"] is not None:
        calibration = point["calibration_files"]
        assert isinstance(calibration, dict)
        frequency = calibration["frequency"]
        assert isinstance(frequency, list)
        for filename in frequency:
            assert isinstance(filename, str)
            result.append(filename)
        for stage in ("rabi", "readout"):
            filename = calibration[stage]
            if filename is not None:
                assert isinstance(filename, str)
                result.append(filename)
    for purpose in ("decay_file", "gain_file", "t1_file", "t2_file"):
        if purpose in point and point[purpose] is not None:
            filename = point[purpose]
            assert isinstance(filename, str)
            result.append(filename)
    return tuple(result)
