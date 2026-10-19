"""Publish explicit legacy CSV conversions through the storage manifest."""

from dataclasses import replace
from pathlib import Path

from zcu_tools.resources.entry import ResultEntry
from zcu_tools.resources.entry.storage import bind_sample_table
from zcu_tools.resources.sample_table import JsonlSampleTable

from migrations.sample_csv import plan_sample_csv

from .models import KeyMappingItem, MigrationMapping
from .paths import contained_path
from .state import MigrationSession, record_pending


def convert_samples(
    session: MigrationSession, entry: ResultEntry | None, mapping: MigrationMapping
) -> None:
    """Plan and publish root samples.csv for the unique migrated qubit.

    Source CSV is hashed against the complete backup. Missing/ambiguous qubit
    identity stays located pending. The CSV owner alone interprets columns,
    units and rows using independent operator declarations. Invalid rows stay
    pending at the owner's source path/row/reason, with a complete byte-identical
    copy at entry/samples.csv; valid rows publish once to entry.samples(component), through a
    prepared JSONL temp and the same hash/recovery protocol as other artifacts.
    Dry-run only plans, without binding an entry or creating any output.
    A completed publication is hash-checked on resume, never appended again.
    Input, source/hash/destination conflicts and I/O failures propagate.
    """
    root = session.manifest.report.source.result_path
    source = root / "samples.csv"
    if not source.is_file():
        return
    source = contained_path(source, root)
    session.baseline(source)
    declarations = session.manifest.report.declarations
    plan = plan_sample_csv(
        [source],
        dev_value_column=declarations.sample_value_column,
        source_unit=declarations.sample_source_unit,
    )
    for pending in plan.pending:
        record_pending(
            session, pending.source_path, f"row {pending.row_number}", pending.reason
        )
    if plan.pending:
        preserved = session.publish(
            source,
            session.manifest.report.destination.result_path / "samples.csv",
            operation="copy",
        )
        session.completed(preserved)
    names = [
        name
        for name, seed in mapping.components.items()
        if isinstance(seed.get("kind"), str) and str(seed["kind"]).startswith("qubit/")
    ]
    if len(names) != 1:
        record_pending(
            session, source, "samples.csv", "No unique migrated qubit component"
        )
        return
    target = (
        entry.samples(names[0]).path
        if entry is not None and not session.dry_run
        else bind_sample_table(
            session.manifest.report.destination.result_path, names[0]
        ).path
    )
    report = session.manifest.report
    row = KeyMappingItem(
        old_file=source,
        old_key="samples.csv",
        new_file=target,
        new_path=None,
        action="value",
        reason=f"CSV plan: {len(plan.rows)} converted rows, {len(plan.pending)} pending rows; explicit independent sample declarations",
    )
    session.report(
        replace(
            report,
            key_mappings=(
                *(
                    item
                    for item in report.key_mappings
                    if (item.old_file, item.old_key) != (source, "samples.csv")
                ),
                row,
            ),
        )
    )
    session.baseline(source)
    if not plan.rows:
        return

    def build(path: Path) -> None:
        table = JsonlSampleTable(path)
        for sample in plan.rows:
            table.append(
                run_id=sample.run_id,
                point=None,
                source="manual",
                at=None,
                dev_value=sample.dev_value,
                dev_unit=sample.dev_unit,
                **sample.fields,
            )

    state = session.publish(source, target, operation="samples", build=build)
    session.completed(state)
