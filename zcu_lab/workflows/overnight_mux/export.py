"""Re-export mux monitoring samples: python -m ...overnight_mux.export RUN --result ENTRY_RESULT."""

import argparse
from pathlib import Path

from zcu_tools.experiment.workflows.saved_run import read_manifest, read_records
from zcu_tools.resources.entry.storage import bind_record_table

from zcu_lab.workflows.common.samples import (
    SampleExportReport,
    SampleRow,
    append_records,
)
from zcu_lab.workflows.overnight_mux.samples import point_samples
from zcu_lab.workflows.overnight_mux.workflow import OvernightMuxPlan, OvernightMuxPoint


def export_run(run: Path, *, result: Path) -> tuple[SampleExportReport, ...]:
    """Append committed rounds to each participant's explicit entry Result table.

    Prepare all rows before writing. Use the live projection and run/round keys;
    retries replace whole logical rows through latest-wins. Append errors retain
    earlier writes, including earlier components. No hardware, refit or Labber.
    """
    manifest = read_manifest(run, "overnight_mux")
    plan = OvernightMuxPlan.model_validate(manifest.plan)
    if str(plan.snapshot.entry_id) != manifest.identity.entry_id:
        raise ValueError(
            f"{run}/manifest.json: Plan snapshot entry and identity differ"
        )
    rows: dict[str, list[SampleRow]] = {}
    for saved in read_records(run, OvernightMuxPoint):
        for qubit, row in point_samples(
            manifest.identity.run_id, plan, saved.record, saved.directory
        ):
            rows.setdefault(qubit, []).append(row)
    return tuple(
        append_records(
            manifest.identity.run_id,
            rows=tuple(items),
            table=bind_record_table(result, qubit),
            point=plan.snapshot.point,
        )
        for qubit, items in rows.items()
    )


def main(argv: list[str] | None = None) -> int:
    """Re-export RUN --result ENTRY_RESULT; argv excludes the executable name.

    Return zero on success. Invalid input/native data and append failures raise.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args(argv)
    for report in export_run(args.run, result=args.result):
        print(f"samples: {report.written_count}: {report.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
