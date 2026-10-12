"""Fixed terminal samples binding and sequential append receipts."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from zcu_tools.experiment.workflows import CommittedRecord
from zcu_tools.experiment.workflows.exporting import SampleRow
from zcu_tools.resources.sample_table import JsonlSampleTable


class UnsupportedSamples(ValueError):
    """The accepted workflow cannot produce scientific sample rows.

    The exception text is the specific attachment's reason. This is distinct
    from a nonterminal lifecycle or stale target; no append was attempted.
    """


class SampleExportError(RuntimeError):
    """Sequential append failed after a retained prefix, not a successful export.

    written_count is the number of rows successfully appended by THIS call,
    including zero; it is not the number of latest keys or existing history rows.
    cause is the original OSError, also retained as __cause__ when raised by the
    host. The message preserves its text and prefix count. Previous appends are
    not rolled back. Retry appends the whole batch; the owner reads latest keys.
    """

    def __init__(self, written_count: int, cause: OSError) -> None:
        if written_count < 0:
            raise ValueError("written_count must be nonnegative")
        self.written_count = written_count
        self.cause = cause
        super().__init__(f"Sample export failed after {written_count} rows: {cause}")


@dataclass(frozen=True)
class SampleExportReport:
    """Receipt of one successful explicit append, without ledger/lifecycle effects.

    run_id is the targeted workflow identity. written_count counts this call's
    appended rows, including zero for a supported run with no successful points.
    path is the owner-bound history path, not a CSV or workflow data destination.
    """

    run_id: str
    written_count: int
    path: Path


@dataclass(frozen=True)
class SampleExportBinding:
    """Per-Start captured owner and science callback, fixed across source changes.

    run_id is the accepted identity; point is the captured entry point (nullable
    under the sample owner's contract). table is bound to the captured concrete
    component and exact entry. prepare consumes committed records and validates
    ALL rows before returning a batch; None means unsupported. unsupported_reason
    is nonempty only when prepare is None. Invalid combinations raise ValueError.
    It never recaptures catalog/draft/device state or performs admission itself.
    """

    run_id: str
    point: str | None
    table: JsonlSampleTable
    prepare: Callable[[tuple[CommittedRecord, ...]], tuple[SampleRow, ...]] | None
    unsupported_reason: str | None

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("Sample binding requires a run_id")
        if (self.prepare is None) != bool(self.unsupported_reason):
            raise ValueError("Unsupported samples require an explicit reason")


def append_samples(
    run_id: str,
    *,
    rows: tuple[SampleRow, ...],
    table: JsonlSampleTable,
    point: str | None,
) -> SampleExportReport:
    """Append a fully science-validated batch through its captured sample owner.

    run_id is the workflow identity for source=workflow:<run_id>. rows are prepared
    named-field artifacts, not raw/unvalidated producer data. table is the fixed
    entry/component capability; point is Start's historical point. No admission,
    accepted/ledger event, native save, fitting, field merging or rollback occurs.
    On OSError, raise SampleExportError with successful-prefix count and original
    cause; prior rows remain. Retry appends every row, leaving latest-wins to the
    table owner. Other programming/validation errors propagate unchanged.
    """
    if not run_id:
        raise ValueError("Sample export requires a run_id")
    written = 0
    for row in rows:
        try:
            table.append(
                run_id=row.key,
                point=point,
                source=f"workflow:{run_id}",
                at=None,
                **{field.name: field.value for field in row.fields},
            )
        except OSError as cause:
            raise SampleExportError(written, cause) from cause
        written += 1
    return SampleExportReport(run_id, written, table.path)
