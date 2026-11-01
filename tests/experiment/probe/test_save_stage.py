"""Save-stage contracts from independently constructed typed records."""

from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from pydantic import Field
from zcu_tools.experiment import AxesSpec, Axis, ZSpec, native_persistence, probe
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
    np.testing.assert_allclose(
        outcome.record.result.frequency,
        [1000.123456789, 1001.987654321],
        rtol=1e-15,
        atol=0,
    )
    np.testing.assert_array_equal(outcome.record.result.signal, [2 + 4j, 6 - 8j])
    loaded = load_run_with_metadata(outcome.native_path, spec=spec)
    np.testing.assert_array_equal(loaded.result.signal, outcome.record.result.signal)


def test_save_reports_a_differing_result_field_through_public_comparison(
    tmp_path: Path,
):
    @dataclass
    class AuxiliaryResult(SavingResult):
        auxiliary: NDArray[np.float64] = field(default_factory=lambda: np.array([0.0]))

    original = SavingCore.AXES_SPEC
    spec = AxesSpec(
        axes=original.axes,
        z=original.z,
        result_type=AuxiliaryResult,
        cfg_type=SavingCfg,
        tag=original.tag,
    )
    record = RunRecord(
        SavingCfg(),
        AuxiliaryResult(np.array([1.0]), np.array([2 + 3j]), np.array([7.0])),
        native_metadata(spec.tag).model_dump(mode="json"),
    )
    outcome = probe.save_stage(record, spec, tmp_path)
    assert outcome.roundtrip_equal is False
    np.testing.assert_array_equal(outcome.record.result.auxiliary, [0.0])
    assert (
        probe.compare_roundtrip(record.result, outcome.record.result, spec=spec)
        is False
    )
    # A correctly supplied field must compare equal through the same public seam.
    outcome.record.result.auxiliary[:] = 7.0
    assert (
        probe.compare_roundtrip(record.result, outcome.record.result, spec=spec) is True
    )


def test_save_preserves_native_cfg_diagnostic(tmp_path: Path):
    class PositiveCfg(SavingCfg):
        gain: float = Field(default=1.0, gt=0)

    original = SavingCore.AXES_SPEC
    spec = AxesSpec(
        axes=original.axes,
        z=original.z,
        result_type=SavingResult,
        cfg_type=PositiveCfg,
        tag=original.tag,
    )
    record = RunRecord(
        PositiveCfg.model_construct(gain=-1.0),
        SavingResult(np.array([1.0]), np.array([2 + 3j])),
        native_metadata(spec.tag).model_dump(mode="json"),
    )
    with pytest.warns(UserWarning, match="gain"):
        outcome = probe.save_stage(record, spec, tmp_path)
    assert outcome.record.cfg is None
    assert outcome.cfg_issue is not None and "gain" in outcome.cfg_issue
    assert outcome.roundtrip_equal is True


def test_save_propagates_native_io_failure(tmp_path: Path):
    spec = SavingCore.AXES_SPEC
    record = RunRecord(
        SavingCfg(),
        SavingResult(np.array([1.0]), np.array([2 + 3j])),
        native_metadata(spec.tag).model_dump(mode="json"),
    )
    (tmp_path / "data.h5").mkdir()
    with pytest.raises(IsADirectoryError):
        probe.save_stage(record, spec, tmp_path)


def test_save_propagates_original_native_restore_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    spec = SavingCore.AXES_SPEC
    record = RunRecord(
        SavingCfg(),
        SavingResult(np.array([1.0]), np.array([2 + 3j])),
        native_metadata(spec.tag).model_dump(mode="json"),
    )
    error = ValueError("native restore failed")

    def failing_restore(_stored, *, spec):
        raise error

    monkeypatch.setattr(native_persistence, "restore_run", failing_restore)
    with pytest.raises(ValueError, match="native restore failed") as caught:
        probe.save_stage(record, spec, tmp_path)
    assert caught.value is error


@dataclass
class _RunOnlyScalarResult:
    """Coordinate and signal arrays plus a scalar estimate only run produces."""

    coordinate: NDArray[np.float64]
    estimate: float | None
    signal: NDArray[np.complex128]


@pytest.fixture
def run_only_scalar_spec() -> AxesSpec[_RunOnlyScalarResult, SavingCfg]:
    return AxesSpec(
        axes=(Axis("coordinate", "coordinate", "s"),),
        z=ZSpec("signal", "signal", "V"),
        result_type=_RunOnlyScalarResult,
        cfg_type=SavingCfg,
        tag="test/run_only_scalar",
    )


def test_scalar_field_restored_as_none_reports_dtype_and_keeps_comparing(
    run_only_scalar_spec: AxesSpec[_RunOnlyScalarResult, SavingCfg],
    caplog: pytest.LogCaptureFixture,
):
    original = _RunOnlyScalarResult(
        coordinate=np.array([1.0, 2.0]),
        estimate=0.5,
        signal=np.array([1 + 2j, 3 - 4j]),
    )
    restored = replace(original, estimate=None, signal=np.array([1 + 2j, 5 - 4j]))
    assert (
        probe.compare_roundtrip(original, restored, spec=run_only_scalar_spec) is False
    )
    warnings = [
        record.message
        for record in caplog.records
        if record.name == "zcu_tools.experiment.probe" and record.levelname == "WARNING"
    ]
    assert len(warnings) == 2
    estimate_warning, signal_warning = warnings
    assert "field=estimate" in estimate_warning
    assert "dtype float64 != object" in estimate_warning
    assert "field=signal" in signal_warning
    assert "value" in signal_warning.lower()


def test_scalar_field_none_on_both_sides_is_equal(
    run_only_scalar_spec: AxesSpec[_RunOnlyScalarResult, SavingCfg],
    caplog: pytest.LogCaptureFixture,
):
    original = _RunOnlyScalarResult(
        coordinate=np.array([1.0, 2.0]),
        estimate=None,
        signal=np.array([1 + 2j, 3 - 4j]),
    )
    restored = replace(original, signal=original.signal.copy())
    assert (
        probe.compare_roundtrip(original, restored, spec=run_only_scalar_spec) is True
    )
    assert not [
        record
        for record in caplog.records
        if record.name == "zcu_tools.experiment.probe"
    ]
