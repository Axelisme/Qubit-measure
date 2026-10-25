"""Fixed-flux monitoring rows, shared by live append and manual re-export."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from zcu_tools.datafile import JsonObject

from zcu_lab.workflows.common.samples import (
    SampleRow,
    append_sample,
    lifetime_fields,
    native_devices,
    sample_row,
)

if TYPE_CHECKING:
    from zcu_lab.workflows.overnight.workflow import OvernightPlan, OvernightPoint
    from zcu_lab.workflows.overnight_mux.workflow import OvernightMuxQubit


def pair_fields(point: OvernightPoint | OvernightMuxQubit) -> JsonObject:
    """Project a monitoring pair's statuses, accepted us values and both models."""
    encoded = point.model_dump(mode="json")
    return {
        "t1": lifetime_fields(
            point.t1_status,
            point.t1_reason,
            point.t1_us,
            point.t1_error_us,
            encoded["t1_models"],
        ),
        "t2echo": lifetime_fields(
            point.t2_status,
            point.t2_reason,
            point.t2_us,
            point.t2_error_us,
            encoded["t2_models"],
            detune_mhz=0.0,
        ),
    }


def point_sample(
    run_id: str, plan: OvernightPlan, point: OvernightPoint, directory: Path
) -> SampleRow:
    """Build one row per round, preserving fixed pulse/readout settings and UTC times.

    Devices come from the final saved T1/Echo acquisition, or null if neither
    completed. Flux is the commanded setpoint. Primary rejected values stay null.
    """
    fields = pair_fields(point)
    fields.update(
        {
            "round_index": point.round_index,
            "status": point.status,
            "scheduled_start": point.scheduled_start.isoformat(),
            "actual_start": point.actual_start.isoformat()
            if point.actual_start
            else None,
            "flux": {
                "device": plan.flux_dev,
                "value": point.flux,
                "unit": plan.flux_unit,
            },
            "devices": native_devices(
                directory, tuple(f for f in (point.t1_file, point.t2_file) if f)
            ),
            "t1_file": point.t1_file,
            "t2echo_file": point.t2_file,
            "echo_relax_delay_us": point.echo_relax_delay_us,
            "echo_t1_source": point.echo_t1_source,
            "settings": {
                "qub_pulse": plan.qub_pulse.model_dump(mode="json"),
                "readout": plan.readout.model_dump(mode="json"),
            },
        }
    )
    return sample_row(f"{run_id}/round/{point.round_index}", fields)


def append_point_sample(
    run_id: str, plan: OvernightPlan, point: OvernightPoint, directory: Path
) -> None:
    """Append after commit to the captured entry, or skip when result_path is None.

    I/O errors propagate after commit; manual --result can replay the stable key.
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
