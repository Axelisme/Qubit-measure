"""External point edits through the session owner, without filesystem repair."""

from hashlib import sha256
from pathlib import Path

import pytest
from zcu_tools.format_version import YamlValue
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.expected_error import FailedPreconditionError, InvalidInputError
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.context_types import PointFileRevision, PointUseRequest
from zcu_tools.gui.session.events import (
    AliasesChangedPayload,
    MdChangedPayload,
    MlChangedPayload,
)
from zcu_tools.gui.session.services.context import ContextService
from zcu_tools.resources.entry import ResultEntry

type SelectedPoint = tuple[ContextService, BaseEventBus, ResultEntry]


def test_reload_installs_before_fact_and_keeps_old_capture(selected) -> None:
    context, bus, _ = selected
    assert hasattr(context, "reload_file"), "Owner must reload a selected point file"
    from zcu_tools.gui.session.events import PointFileChangedPayload

    old = context.read_context()
    file = context.get_file_revisions()[0]
    observed: list[YamlValue] = []
    facts: list[PointFileChangedPayload] = []
    bus.subscribe(
        MdChangedPayload,
        lambda _: observed.append(context.read_context().value("Q1.rate")),
    )
    bus.subscribe(PointFileChangedPayload, facts.append)
    file.path.write_text(file.path.read_text().replace("4300.0", "5100.0"))
    revision = context.reload_file(file.path, file.revision)
    assert revision.revision == sha256(file.path.read_bytes()).hexdigest()
    assert context.read_context().value("Q1.rate") == 5100.0
    assert old.value("Q1.rate") == 4300.0
    assert observed == [5100.0]
    assert facts[-1].status.state == "reloaded"
    assert facts[-1].status.file == revision


@pytest.mark.parametrize("document", ["parameters", "library", "aliases"])
def test_invalid_candidate_retains_memory_revision_and_bytes(
    selected, document: str
) -> None:
    context, bus, _ = selected
    assert hasattr(context, "reload_file"), (
        "Invalid external candidates need strict owner admission"
    )
    file = next(
        item for item in context.get_file_revisions() if item.document == document
    )
    old = context.read_context()
    facts: list[object] = []
    for payload in (MdChangedPayload, MlChangedPayload, AliasesChangedPayload):
        bus.subscribe(payload, facts.append)
    body = {
        "parameters": 'format: zcu.parameter-container\nformat_version: "2.0"\ncomponents: {Q1: {kind: nonexistent}}\n',
        "library": 'format: zcu.module-library\nformat_version: "2.0"\ncomponents: {Q1: {pulses: {bad: {type: nonexistent}}}}\n',
        "aliases": 'format: zcu.aliases\nformat_version: "1.0"\naliases: {driver: "$later.rate", later: Q1}\n',
    }[document]
    file.path.write_text(body)
    siblings = set(file.path.parent.iterdir())
    with pytest.raises(InvalidInputError, match=str(file.path)):
        context.reload_file(file.path, file.revision)
    assert (
        next(item for item in context.get_file_revisions() if item.document == document)
        == file
    )
    assert context.read_context().value("Q1.rate") == old.value("Q1.rate")
    assert context.get_alias_rows() == ()
    assert file.path.read_text() == body
    assert not (set(file.path.parent.iterdir()) - siblings) - {
        Path(f"{file.path}.lock")
    }
    assert facts == []
    status = context.get_file_sync_status()[0]
    assert status.state == "error"
    assert status.error is not None and str(file.path) in status.error


def test_revision_checks_compare_disk_while_reload_compares_installed(selected) -> None:
    context, _, _ = selected
    assert hasattr(context, "check_file_revision"), (
        "Draft checks must reject uninstalled disk edits"
    )
    file = context.get_file_revisions()[2]
    context.check_file_revision(file)
    file.path.write_text(
        'format: zcu.aliases\nformat_version: "1.0"\naliases: {driver: Q1}\n'
    )
    with pytest.raises(FailedPreconditionError, match="revision"):
        context.check_file_revision(file)
    with pytest.raises(FailedPreconditionError, match="revision"):
        context.reload_file(file.path, "stale")
    installed = context.reload_file(file.path, None)
    context.check_file_revision(installed)
    assert context.get_alias_rows() == (AliasRow("driver", "Q1"),)
    file.path.unlink()
    with pytest.raises(FailedPreconditionError, match="revision"):
        context.check_file_revision(installed)
    assert context.reload_file(file.path, installed.revision).revision is None
    assert context.get_alias_rows() == ()


@pytest.mark.parametrize("relative", [False, True])
def test_reload_rejects_paths_outside_current_point(
    selected, tmp_path: Path, relative: bool
) -> None:
    context, _, _ = selected
    assert hasattr(context, "reload_file"), (
        "Reload must limit paths to selected point owners"
    )
    path = Path("point.yaml") if relative else tmp_path / "unowned.yaml"
    with pytest.raises(InvalidInputError):
        context.reload_file(path, None)
    with pytest.raises(InvalidInputError):
        context.check_file_revision(PointFileRevision("parameters", path, None))


def test_sync_status_projects_current_installed_version_after_owner_write(
    selected,
) -> None:
    context, _, _ = selected
    assert hasattr(context, "reload_file"), (
        "Sync status must project installed versions"
    )
    file = context.get_file_revisions()[0]
    file.path.write_text(file.path.read_text().replace("4300.0", "5100.0"))
    context.reload_file(file.path, file.revision)
    context.set_parameter(context.read_context().target, "Q1.rate", 6100.0)
    assert context.get_file_sync_status()[0].file == context.get_file_revisions()[0]


def test_switch_clears_sync_status(selected) -> None:
    context, _, entry = selected
    assert hasattr(context, "reload_file"), "Reload status must follow point lifetime"
    file = context.get_file_revisions()[2]
    file.path.write_text("invalid: true")
    with pytest.raises(InvalidInputError):
        context.reload_file(file.path, file.revision)
    assert context.get_file_sync_status()
    entry.new_point("second")
    context.use_point(PointUseRequest("second"))
    assert context.get_file_sync_status() == ()
    context.clear_context()
    assert context.get_file_sync_status() == ()
