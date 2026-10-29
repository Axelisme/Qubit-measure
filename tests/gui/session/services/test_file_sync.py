"""External point edits through the session owner, without filesystem repair."""

from hashlib import sha256
from pathlib import Path

import pytest
from zcu_tools.format_version import YamlValue
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.expected_error import FailedPreconditionError, InvalidInputError
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.context_types import FileRevision, PointUseRequest
from zcu_tools.gui.session.events import (
    AliasesChangedPayload,
    MdChangedPayload,
    MlChangedPayload,
)
from zcu_tools.gui.session.services.context import ContextService
from zcu_tools.resources.entry import ResultEntry

type SelectedPoint = tuple[ContextService, BaseEventBus, ResultEntry]


@pytest.mark.parametrize("watcher", [False, True])
@pytest.mark.parametrize("global_active", [False, True])
def test_absent_library_installs_external_creation_strictly(
    selected: SelectedPoint, watcher: bool, global_active: bool
) -> None:
    context, bus, entry = selected
    path = entry.result_path / (
        "library.yaml" if global_active else "points/first/library.yaml"
    )
    path.unlink()
    if global_active:
        context.use_global()
    else:
        context.use_point(PointUseRequest("first"))
    file = next(
        item for item in context.get_file_revisions() if item.document == "library"
    )
    assert file.revision is None
    captured = context.read_context()
    binding = context.get_source_binding()
    facts: list[MlChangedPayload] = []
    bus.subscribe(MlChangedPayload, facts.append)
    context.check_file_revision(file)

    invalid = 'format: zcu.module-library\nformat_version: "2.0"\ncomponents: {Q1: {waveforms: {bad: {type: nonexistent}}}}\n'
    path.write_text(invalid)
    with pytest.raises(FailedPreconditionError, match="revision"):
        context.check_file_revision(file)
    if watcher:
        context.synchronize_files(admit_reload=True)
    else:
        with pytest.raises(InvalidInputError):
            context.reload_file(path, None)
    assert context.read_context().library.waveform_keys() == ()
    assert (
        next(
            item for item in context.get_file_revisions() if item.document == "library"
        )
        == file
    )
    assert path.read_text() == invalid
    assert facts == []

    body = 'format: zcu.module-library\nformat_version: "2.0"\ncomponents: {Q1: {waveforms: {base: {style: const, length: 0.2}}}}\n'
    path.write_text(body)
    if watcher:
        context.synchronize_files(admit_reload=True)
    else:
        context.reload_file(path, None)
    assert context.read_context().library.waveform_keys() == ("Q1.waveforms.base",)
    assert captured.library.waveform_keys() == ()
    assert context.get_source_binding() == binding
    assert len(facts) == 1
    assert context.get_file_sync_status()[0].state == "reloaded"
    assert path.read_text() == body
    with pytest.raises(FailedPreconditionError, match="revision"):
        context.reload_file(path, None)


def entry_candidate(file: FileRevision) -> str:
    """Build valid authored bytes for the three entry document owners."""
    if file.document == "parameters":
        return file.path.read_text().replace("4300.0", "5100.0")
    if file.document == "library":
        return (
            'format: zcu.module-library\nformat_version: "2.0"\n'
            "components: {Q1: {waveforms: {global: {style: const, length: 0.2}}}}\n"
        )
    return 'format: zcu.aliases\nformat_version: "1.0"\naliases: {driver: Q1}\n'


@pytest.mark.parametrize("global_active", [False, True])
@pytest.mark.parametrize("document", ["parameters", "library", "aliases"])
def test_entry_candidates_obey_pending_validation_and_capture_isolation(
    selected: SelectedPoint, global_active: bool, document: str
) -> None:
    context, bus, entry = selected
    if global_active:
        context.use_global()
    active_files = context.get_file_revisions()
    installed_files = context.get_installed_file_revisions()
    assert len(installed_files) == (3 if global_active else 6)
    assert installed_files[:3] == active_files
    file = next(
        item
        for item in installed_files
        if item.path.parent == entry.result_path and item.document == document
    )
    captured = context.read_context(setup=True)
    active = context.read_context()
    content_facts: list[object] = []
    for payload in (MdChangedPayload, MlChangedPayload, AliasesChangedPayload):
        bus.subscribe(payload, content_facts.append)
    candidate = entry_candidate(file)
    file.path.write_text(candidate)
    context.synchronize_files(admit_reload=False)
    assert context.get_installed_file_revisions() == installed_files
    assert context.read_context(setup=True).value("Q1.rate") == 4300.0
    assert context.read_context(setup=True).library.waveform_keys() == ()
    assert context.get_alias_rows(setup=True) == ()
    assert context.get_file_sync_status()[0].state == "pending"
    assert content_facts == []
    with pytest.raises(FailedPreconditionError, match="revision"):
        context.check_file_revision(file)
    with pytest.raises(FailedPreconditionError, match="revision"):
        context.reload_file(file.path, "stale")
    context.synchronize_files(admit_reload=True)
    installed = next(
        item
        for item in context.get_installed_file_revisions()
        if item.path == file.path
    )
    assert installed.revision == sha256(candidate.encode()).hexdigest()
    assert context.get_file_sync_status()[0].state == "reloaded"
    current = context.read_context(setup=True)
    if document == "parameters":
        assert current.value("Q1.rate") == 5100.0
    elif document == "library":
        assert current.library.waveform_keys() == ("Q1.waveforms.global",)
    else:
        assert context.get_alias_rows(setup=True) == (AliasRow("driver", "Q1"),)
        assert current.lookup("$driver").value == "Q1"
    assert captured.value("Q1.rate") == 4300.0
    assert captured.library.waveform_keys() == ()
    assert captured.resolved_aliases() == ()
    assert active.value("Q1.rate") == 4300.0
    if not global_active:
        assert context.get_file_revisions() == active_files
        assert context.read_context().value("Q1.rate") == 4300.0
        assert context.read_context().library.waveform_keys() == ()
        assert context.get_alias_rows() == ()
        assert content_facts == []
    else:
        assert len(content_facts) == 1


def test_reload_installs_before_fact_and_keeps_old_capture(selected) -> None:
    context, bus, _ = selected
    assert hasattr(context, "reload_file"), "Owner must reload a selected point file"
    from zcu_tools.gui.session.events import FileChangedPayload

    old = context.read_context()
    file = context.get_file_revisions()[0]
    observed: list[YamlValue] = []
    facts: list[FileChangedPayload] = []
    bus.subscribe(
        MdChangedPayload,
        lambda _: observed.append(context.read_context().value("Q1.rate")),
    )
    bus.subscribe(FileChangedPayload, facts.append)
    file.path.write_text(file.path.read_text().replace("4300.0", "5100.0"))
    revision = context.reload_file(file.path, file.revision)
    assert revision.revision == sha256(file.path.read_bytes()).hexdigest()
    assert context.read_context().value("Q1.rate") == 5100.0
    assert old.value("Q1.rate") == 4300.0
    assert observed == [5100.0]
    assert facts[-1].status.state == "reloaded"
    assert facts[-1].status.file == revision


@pytest.mark.parametrize("scope", ["point", "global", "entry"])
@pytest.mark.parametrize("document", ["parameters", "library", "aliases"])
def test_invalid_candidate_retains_memory_revision_and_bytes(
    selected, document: str, scope: str
) -> None:
    context, bus, _ = selected
    assert hasattr(context, "reload_file"), (
        "Invalid external candidates need strict owner admission"
    )
    if scope == "global":
        context.use_global()
    setup = scope == "entry"
    files = context.get_installed_file_revisions()
    scoped_files = files[3:] if setup else files[:3]
    file = next(item for item in scoped_files if item.document == document)
    old = context.read_context(setup=setup)
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
        next(
            item
            for item in context.get_installed_file_revisions()
            if item.path == file.path
        )
        == file
    )
    assert context.read_context(setup=setup).value("Q1.rate") == old.value("Q1.rate")
    assert context.get_alias_rows(setup=setup) == ()
    assert file.path.read_text() == body
    assert set(file.path.parent.iterdir()) == siblings
    assert facts == []
    status = context.get_file_sync_status()[0]
    assert status.state == "error"
    assert status.error is not None and str(file.path) in status.error


@pytest.mark.parametrize("watcher", [False, True])
@pytest.mark.parametrize("invalid_field", ["created_at", "rate"])
def test_reload_validates_complete_parameter_candidate(
    selected, watcher: bool, invalid_field: str
) -> None:
    context, bus, _ = selected
    old = context.read_context()
    binding = context.get_source_binding()
    file = context.get_file_revisions()[0]
    facts: list[MdChangedPayload] = []
    bus.subscribe(MdChangedPayload, facts.append)
    original = file.path.read_text()
    body = (
        original.replace("created_at:", "created_at: not-a-time #")
        if invalid_field == "created_at"
        else original.replace("4300.0", "bad")
    )
    file.path.write_text(body)
    if watcher:
        context.synchronize_files(admit_reload=True)
    else:
        with pytest.raises(InvalidInputError, match=str(file.path)):
            context.reload_file(file.path, file.revision)
    assert context.get_file_revisions()[0] == file
    assert context.get_source_binding() == binding
    assert context.read_context().value("Q1.rate") == old.value("Q1.rate")
    assert facts == []
    assert context.get_file_sync_status()[0].state == "error"
    assert file.path.read_text() == body

    file.path.write_text(original.replace("4300.0", "5100.0"))
    if watcher:
        context.synchronize_files(admit_reload=True)
    else:
        context.reload_file(file.path, file.revision)
    assert context.read_context().value("Q1.rate") == 5100.0
    assert len(facts) == 1
    assert context.get_file_sync_status()[0].state == "reloaded"


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
    path = Path("parameters.yaml") if relative else tmp_path / "unowned.yaml"
    with pytest.raises(InvalidInputError):
        context.reload_file(path, None)
    with pytest.raises(InvalidInputError):
        context.check_file_revision(FileRevision("parameters", path, None))


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
