"""Pure successful flux-point samples from committed scalar/native artifacts."""

from math import isfinite

from pydantic import TypeAdapter
from zcu_tools.datafile import load_run_data
from zcu_tools.experiment.workflows import CommittedRecord
from zcu_tools.experiment.workflows.exporting import SampleField, SampleRow, SampleRun

from .calibration import CalibrationFailed
from .t1 import T1Plan, T1Point
from .t2 import T2Plan, T2Point


def export_t1_samples(run: SampleRun[T1Plan]) -> tuple[SampleRow, ...]:
    """Prepare T1 rows from a fixed accepted run, without writing or fitting.

    run supplies the detached native plan, exact data_root and committed records.
    Each accepted point contributes freq_mhz, t1_us/t1_err_us and dev_value/unit
    from the LAST native acquisition's flux_dev snapshot, not the plan or live
    device. Failed points are skipped. Keys are <run_id>/<zero-based index>.
    Invalid indices/scalars, degraded summaries, missing snapshot/device/native
    acquisition or escaped paths raise ValueError before returning any batch.
    Native reader I/O/schema failures propagate. No source, catalog or hardware
    is consulted and native units are preserved without conversion.
    """
    points = tuple(_decode(record, TypeAdapter(T1Point)) for record in run.records)
    return _rows(run, points, "t1")


def export_t2echo_samples(run: SampleRun[T2Plan]) -> tuple[SampleRow, ...]:
    """Prepare Echo freq_mhz/t2echo_us/t2echo_err_us rows from fixed artifacts.

    Uses export_t1_samples' artifact/failure semantics: one accepted flux point
    per key, last native flux_dev snapshot, no fit/conversion/write/live capture.
    Malformed evidence raises before any batch is returned; native I/O propagates.
    """
    points = tuple(_decode(record, TypeAdapter(T2Point)) for record in run.records)
    return _rows(run, points, "t2echo")


def export_t2ramsey_samples(run: SampleRun[T2Plan]) -> tuple[SampleRow, ...]:
    """Prepare Ramsey freq_mhz/t2ramsey_us/t2ramsey_err_us from fixed artifacts.

    Uses export_t1_samples' artifact/failure semantics: one accepted flux point
    per key, last native flux_dev snapshot, no fit/conversion/write/live capture.
    Malformed evidence raises before any batch is returned; native I/O propagates.
    """
    points = tuple(_decode(record, TypeAdapter(T2Point)) for record in run.records)
    return _rows(run, points, "t2ramsey")


def _decode[R: T1Point | T2Point](
    record: CommittedRecord, adapter: TypeAdapter[R]
) -> R:
    if record.encoded_record.mode != "json":
        raise ValueError("Samples require structured committed point summaries")
    return adapter.validate_python(record.encoded_record.value)


def _rows(
    run: SampleRun[T1Plan] | SampleRun[T2Plan],
    points: tuple[T1Point | T2Point, ...],
    quantity: str,
) -> tuple[SampleRow, ...]:
    root = run.data_root.resolve()
    seen: set[int] = set()
    rows: list[SampleRow] = []
    for record, point in zip(run.records, points, strict=True):
        if (
            point.index < 0
            or point.index >= len(run.plan.fluxes)
            or point.index in seen
        ):
            raise ValueError("Sample indices must be unique and inside the fixed plan")
        seen.add(point.index)
        if (
            isinstance(point.calibration, CalibrationFailed)
            or point.reason is not None
            or point.lifetime_us is None
        ):
            continue
        if point.error_us is None or not isfinite(point.error_us) or point.error_us < 0:
            raise ValueError("Accepted sample error must be finite and nonnegative")
        if not isfinite(point.lifetime_us) or point.lifetime_us <= 0:
            raise ValueError("Accepted sample lifetime must be finite and positive")
        frequency = point.calibration.f01_mhz
        if not isfinite(frequency) or frequency <= 0:
            raise ValueError("Accepted sample frequency must be finite and positive")
        if not record.run_files:
            raise ValueError("Accepted sample requires a native acquisition snapshot")
        path = (root / record.iteration_dir / record.run_files[-1]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Sample acquisition path escapes data_root")
        snapshot = load_run_data(path).metadata.device_snapshot
        if snapshot is None or run.plan.flux_dev not in snapshot:
            raise ValueError(f"Native snapshot lacks flux device {run.plan.flux_dev}")
        device = snapshot[run.plan.flux_dev]
        value, unit = device.get("value"), device.get("value_unit")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(value)
        ):
            raise ValueError(
                "Native snapshot device value must be a finite real number"
            )
        if not isinstance(unit, str) or not unit:
            raise ValueError("Native snapshot device value_unit must be nonempty")
        rows.append(
            SampleRow(
                f"{run.run_id}/{point.index}",
                (
                    SampleField(name="dev_value", value=value),
                    SampleField(name="dev_unit", value=unit),
                    SampleField(name="freq_mhz", value=frequency),
                    SampleField(name=f"{quantity}_us", value=point.lifetime_us),
                    SampleField(name=f"{quantity}_err_us", value=point.error_us),
                ),
            )
        )
    return tuple(rows)
