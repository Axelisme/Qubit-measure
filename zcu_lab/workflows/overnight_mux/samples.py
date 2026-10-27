"""One sample per monitoring round and participant, including rejected fits."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from zcu_lab.v2.support.flux import record_flux_frame
from zcu_lab.workflows.common.samples import (
    SampleRow,
    append_sample,
    native_devices,
    sample_row,
)
from zcu_lab.workflows.overnight.samples import pair_fields

if TYPE_CHECKING:
    from zcu_lab.workflows.overnight_mux.workflow import (
        OvernightMuxPlan,
        OvernightMuxPoint,
    )


def point_samples(
    run_id: str, plan: OvernightMuxPlan, point: OvernightMuxPoint, directory: Path
) -> tuple[tuple[str, SampleRow], ...]:
    """Prepare all participant rows before append; retain full device snapshots.

    Component names come from each recorded qubit. Shared T1/Echo raw references
    and timestamps identify the round. No commanded flux exists; callers can
    select any coordinate device from devices without guessing a flux source.
    Missing device values contribute zero. Present invalid values raise
    ValueError naming the device before any row is appended.
    """
    devices = native_devices(
        directory, tuple(f for f in (point.t1_file, point.t2_file) if f)
    )
    rows: list[tuple[str, SampleRow]] = []
    for qubit, index in zip(point.qubits, plan.participants, strict=True):
        fields = pair_fields(qubit)
        fields.update(
            {
                "round_index": point.round_index,
                "qubit": qubit.name,
                "status": point.status,
                "scheduled_start": point.scheduled_start.isoformat(),
                "actual_start": point.actual_start.isoformat()
                if point.actual_start
                else None,
                "flux": None,
                "flux_frame": record_flux_frame(plan.flux_frames[qubit.name], devices),
                "devices": None
                if devices is None
                else {name: dict(info) for name, info in devices.items()},
                "t1_file": point.t1_file,
                "t2echo_file": point.t2_file,
                "echo_relax_delay_us": point.echo_relax_delay_us,
                "echo_t1_source": point.echo_t1_source,
                "settings": {
                    "tone": plan.tones[index].model_dump(mode="json"),
                    "readout": plan.readout.model_dump(mode="json"),
                },
            }
        )
        rows.append(
            (qubit.name, sample_row(f"{run_id}/round/{point.round_index}", fields))
        )
    return tuple(rows)


def append_point_samples(
    run_id: str, plan: OvernightMuxPlan, point: OvernightMuxPoint, directory: Path
) -> None:
    """Append after commit to each participant's captured-entry table.

    No entry path means no auto write. Errors propagate; earlier participant
    appends remain and manual --result can replay the same latest-wins keys.
    """
    if plan.result_path is None:
        return
    for qubit, row in point_samples(run_id, plan, point, directory):
        append_sample(plan.result_path, qubit, run_id, plan.snapshot.point, row)
