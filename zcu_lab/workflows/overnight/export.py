"""Re-export fixed-flux monitoring samples: python -m ...overnight.export RUN --result ENTRY_RESULT."""

import argparse
from pathlib import Path

from zcu_tools.experiment.workflows.saved_run import read_manifest, read_records
from zcu_tools.resources.entry.storage import bind_record_table

from zcu_lab.workflows.common.samples import SampleExportReport, append_records
from zcu_lab.workflows.overnight.samples import point_sample
from zcu_lab.workflows.overnight.workflow import OvernightPlan, OvernightPoint


def export_run(run: Path, *, result: Path) -> SampleExportReport:
    """Append committed monitoring rounds to an explicit entry Result directory.

    Use the live projection and stable run/round keys. Validate all rows before
    appending. Missing/malformed native references raise; append OSError retains
    the successful prefix in SampleExportError. No hardware, refit or Labber write.
    """
    manifest = read_manifest(run, "overnight")
    plan = OvernightPlan.model_validate(manifest.plan)
    if str(plan.snapshot.entry_id) != manifest.identity.entry_id:
        raise ValueError(
            f"{run}/manifest.json: Plan snapshot entry and identity differ"
        )
    qubit = plan.snapshot.roles["qubit"]
    if not isinstance(qubit, str):
        raise ValueError("Sample export requires one qubit name")
    points = read_records(run, OvernightPoint)
    rows = tuple(
        point_sample(manifest.identity.run_id, plan, p.record, p.directory)
        for p in points
    )
    return append_records(
        manifest.identity.run_id,
        rows=rows,
        table=bind_record_table(result, qubit),
        point=plan.snapshot.point,
    )


def main(argv: list[str] | None = None) -> int:
    """Re-export RUN --result ENTRY_RESULT; argv excludes the executable name.

    Return zero on success. Invalid input/native data and append failures raise.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args(argv)
    report = export_run(args.run, result=args.result)
    print(f"samples: {report.written_count}: {report.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
