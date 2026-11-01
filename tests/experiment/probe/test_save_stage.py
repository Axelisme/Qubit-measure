"""Save-stage contracts from independently constructed typed records."""

from pathlib import Path

import numpy as np
from zcu_tools.experiment import probe
from zcu_tools.experiment.native_persistence import load_run_with_metadata
from zcu_tools.experiment.records import RunRecord

from tests._native_support import native_metadata
from tests.experiment._saving_support import SavingCfg, SavingCore, SavingResult


def test_save_returns_restored_record_and_native_roundtrip_without_run(tmp_path: Path):
    result = SavingResult(
        frequency=np.array([1000.123456789, 1001.987654321]),
        signal=np.array([2 + 4j, 6 - 8j]),
    )
    spec = SavingCore.AXES_SPEC
    record = RunRecord(
        SavingCfg(gain=2), result, native_metadata(spec.tag).model_dump(mode="json")
    )
    destination = tmp_path / "nested"
    outcome = probe.save_stage(record, spec, destination)
    assert outcome.native_path == destination / "data.h5"
    assert outcome.roundtrip_equal is True
    assert outcome.cfg_issue is None
    assert outcome.record is not record
    assert outcome.record.cfg == record.cfg
    assert outcome.record.metadata == record.metadata
    np.testing.assert_allclose(outcome.record.result.frequency, [1000.123456789, 1001.987654321], rtol=1e-15, atol=0)
    np.testing.assert_array_equal(outcome.record.result.signal, [2 + 4j, 6 - 8j])
    loaded = load_run_with_metadata(outcome.native_path, spec=spec)
    np.testing.assert_array_equal(loaded.result.signal, outcome.record.result.signal)
