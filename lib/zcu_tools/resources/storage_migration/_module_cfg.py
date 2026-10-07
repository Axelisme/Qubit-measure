"""Point-local typed legacy library conversion in the parameter migration step."""

from pathlib import Path

from zcu_tools.resources.entry import ResultEntry

from .models import MigrationMapping
from .state import MigrationSession


def convert_module_cfg(
    session: MigrationSession,
    *,
    entry: ResultEntry | None,
    label: str | None,
    source: Path,
    mapping: MigrationMapping,
) -> None:
    """Prepare/report one legacy library and commit a point's complete Spec file.

    session owns report, source baselines and manifest checkpoints. source is
    the absolute legacy module_cfg.yaml under its result root; an absent file is
    a no-op. label is the validated point label, or None for root metadata with
    no point destination. entry is the matching handle, or None only in dry-run.
    mapping supplies explicit module/waveform source keys and canonical targets.
    Root entries remain located pending with new_file=None and do not set slots.
    Child slots retain first-present module-rule priority. Unsupported expressions,
    Specs and unmapped child references remain located pending without evaluation.
    Document/schema or I/O commit failures are recorded then propagate.

    Dry-run prepares an isolated typed library without destination/manifest writes.
    Normal execution uses one ModuleLibrary.edit, then records source/final hashes
    and converted file in the existing manifest/report checkpoint. Recorded module
    commits are not rewritten; resume verifies their hashes. Remaining pending
    entries do not undo module-step completion. No sources/data/assets are moved,
    no ledger events are invented, and no unrecorded-write recovery is provided.
    """
    if not source.is_file():
        return
    # The formal seed fixes this seam; the assigned writer supplies its body.
    del session, entry, label, mapping
    raise NotImplementedError("Complete the settled point-local module conversion")
