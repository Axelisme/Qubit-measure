"""One record row per committed flux point, including rejected/skipped stages."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import JsonValue
from zcu_tools.datafile import JsonObject
from zcu_tools.experiment.workflows.saved_run import SavedRecord

from zcu_lab.v2.support.flux import record_flux_frame
from zcu_lab.workflows.common.samples import (
    SampleExportError as SampleExportError,
)
from zcu_lab.workflows.common.samples import (
    SampleExportReport as SampleExportReport,
)
from zcu_lab.workflows.common.samples import (
    SampleRow,
    append_sample,
    lifetime_fields,
    native_devices,
    sample_row,
)
from zcu_lab.workflows.common.samples import (
    append_records as append_records,
)

if TYPE_CHECKING:
    from zcu_lab.workflows.coherence_fluxdep.stages.ge import SingleshotResult
    from zcu_lab.workflows.coherence_fluxdep.workflow import (
        CoherencePlan,
        CoherencePoint,
    )
    from zcu_lab.workflows.common.calibration_results import (
        FrequencyResult,
        RabiResult,
        ReadoutResult,
    )


def _native_files(point: CoherencePoint) -> tuple[str, ...]:
    files = list(point.frequency.files if point.frequency else ())
    if point.rabi is not None:
        files.extend(
            f
            for f in (point.rabi.file, point.rabi.amp.file if point.rabi.amp else None)
            if f
        )
    if point.readout and point.readout.file:
        files.append(point.readout.file)
    if point.singleshot and point.singleshot.file:
        files.append(point.singleshot.file)
    for item in (point.t1, point.t2ramsey, point.t2echo):
        if item and item.file:
            files.append(item.file)
    return tuple(files)


def _frequency_fields(result: FrequencyResult | None) -> JsonObject:
    fields: JsonObject = {
        "status": result.status if result else "skipped",
        "reason": result.reason if result else "not attempted",
        "freq_mhz": None,
        "err_mhz": None,
        "fwhm_mhz": None,
        "probe_gain": None,
    }
    if result is None or result.status != "accepted":
        return fields
    fields["freq_mhz"] = result.frequency_mhz
    if result.probe_gains:
        fields["probe_gain"] = result.probe_gains[-1]
    if result.scans:
        diagnostics = result.scans[-1].diagnostics
        fields["err_mhz"] = diagnostics.center_error_mhz
        fields["fwhm_mhz"] = diagnostics.fwhm_mhz
    return fields


def _rabi_mode(result: RabiResult | None) -> str | None:
    # Describe the saved stages, not an unrecorded requested mode.
    if result is None:
        return None
    if result.amp is not None:
        return "len_then_amp" if result.len_status else "amp_only"
    return "len_only" if result.file else None


def _rabi_fields(result: RabiResult | None, amp: JsonValue) -> JsonObject:
    fields: JsonObject = {
        "status": result.status if result else "skipped",
        "reason": result.reason if result else "not attempted",
        "mode": _rabi_mode(result),
        "amp": amp,
        "pi_len_us": None,
        "pi_gain": None,
        "pi2_len_us": None,
        "pi2_gain": None,
    }
    if result is None or result.status != "accepted":
        return fields
    fields["pi_len_us"] = result.pi_length_us
    fields["pi_gain"] = result.drive_gain
    fields["pi2_len_us"] = result.pi2_length_us
    if result.pi2_length_us is not None:
        fields["pi2_gain"] = (
            result.pi2_drive_gain
            if result.pi2_drive_gain is not None
            else result.drive_gain
        )
    return fields


def _readout_fields(result: ReadoutResult | None) -> JsonObject:
    fields: JsonObject = {
        "status": result.status if result else "skipped",
        "reason": result.reason if result else "not attempted",
        "freq_mhz": None,
        "gain": None,
    }
    if result is not None and result.status == "accepted":
        fields["freq_mhz"] = result.frequency_mhz
        fields["gain"] = result.gain
    return fields


def _singleshot_fields(result: SingleshotResult | None) -> JsonObject:
    if result is None:
        return {
            "status": "skipped",
            "reason": "disabled or not attempted",
            "file": None,
        }
    return {"status": result.status, "reason": result.reason, "file": result.file}


def point_sample(
    run_id: str, plan: CoherencePlan, point: CoherencePoint, directory: Path
) -> SampleRow:
    """Prepare one live/offline row from the point and its explicit native files.

    Only accepted primary values are included. The last saved acquisition supplies
    all devices; no acquisition means null, not a live or plan snapshot. Flux
    records the commanded setpoint separately. Missing stages are skipped with a
    reason. Models preserve both fit diagnostics, including secondary lifetimes.
    Rabi mode describes saved stages, or null when no Rabi acquisition is recorded.
    A present snapshot must supply finite, non-bool values for every captured
    frame device; otherwise ValueError names it before any row is appended.
    """
    encoded = point.model_dump(mode="json")
    devices = native_devices(directory, _native_files(point))
    fields: JsonObject = {
        "index": point.index,
        "status": point.status,
        "flux": {"device": plan.flux_dev, "value": point.flux, "unit": plan.flux_unit},
        "flux_frame": {
            name: dict(calibration)
            for name, calibration in record_flux_frame(plan.flux_frame, devices).items()
        },
        "devices": None
        if devices is None
        else {name: dict(info) for name, info in devices.items()},
        "frequency": _frequency_fields(point.frequency),
        "rabi": _rabi_fields(
            point.rabi, encoded["rabi"]["amp"] if point.rabi else None
        ),
        "readout": _readout_fields(point.readout),
        "singleshot": _singleshot_fields(point.singleshot),
    }
    # Point's JSON preserves dataclass model diagnostics without another mirror schema.
    for name, item in (
        ("t1", point.t1),
        ("t2ramsey", point.t2ramsey),
        ("t2echo", point.t2echo),
    ):
        models = encoded[name]["models"] if item is not None else []
        fields[name] = lifetime_fields(
            item.status if item else "skipped",
            item.reason if item else "disabled or not attempted",
            item.lifetime_us if item else None,
            item.error_us if item else None,
            models,
            detune_mhz=item.detune_mhz if item else None,
        )
    return sample_row(f"{run_id}/{point.index}", fields)


def append_point_sample(
    run_id: str, plan: CoherencePlan, point: CoherencePoint, directory: Path
) -> None:
    """Append immediately after commit; no entry destination means no auto write.

    Append errors propagate after the point is already committed. Offline
    --result can replay it using the same point_sample projection and stable key.
    """
    if plan.result_path is None:
        return
    qubit = plan.snapshot.roles["qubit"]
    if not isinstance(qubit, str):
        raise ValueError("Sample export requires one qubit name")
    append_sample(
        plan.result_path,
        qubit,
        run_id,
        plan.snapshot.point,
        point_sample(run_id, plan, point, directory),
    )


def prepare_samples(
    run_id: str, plan: CoherencePlan, points: tuple[SavedRecord[CoherencePoint], ...]
) -> tuple[SampleRow, ...]:
    """Prepare every committed point with exactly the live sample projection.

    Validate the whole batch before writing; native input failures propagate.
    """
    return tuple(point_sample(run_id, plan, p.record, p.directory) for p in points)
