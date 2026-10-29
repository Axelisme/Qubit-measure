"""Real-file watcher admission, deterministic debounce draining and teardown."""

from zcu_tools.gui.session import adapters
from zcu_tools.gui.session.context_types import PointUseRequest
from zcu_tools.gui.session.events import MdChangedPayload

from tests.gui._qt_waits import qt_until


def test_watcher_follows_point_to_global_directory_and_keeps_old_captures(
    qapp, selected
) -> None:
    from zcu_tools.gui.session.adapters import FileWatcher

    context, bus, entry = selected
    old = context.read_context()
    point_path = context.get_file_revisions()[0].path
    watcher = FileWatcher(context, bus, admit_reload=lambda: True)
    try:
        context.use_global()
        global_capture = context.read_context()
        path = entry.result_path / "parameters.yaml"
        path.write_text(path.read_text().replace("4300.0", "5500.0"))
        watcher.flush()
        assert context.read_context().value("Q1.rate") == 5500.0
        assert global_capture.value("Q1.rate") == 4300.0
        assert old.value("Q1.rate") == 4300.0
        point_path.write_text("invalid: true")
        replacement = path.with_suffix(".tmp")
        replacement.write_text(path.read_text().replace("5500.0", "5600.0"))
        replacement.replace(path)
        qt_until(
            lambda: context.read_context().value("Q1.rate") == 5600.0,
            label="global directory reload after scope switch",
        )
        assert all(
            status.file.path.parent == entry.result_path
            for status in context.get_file_sync_status()
        )
    finally:
        watcher.close()


def test_watcher_observes_entry_directory_while_point_stays_active(
    qapp, selected
) -> None:
    from zcu_tools.gui.session.adapters import FileWatcher

    context, bus, entry = selected
    captured = context.read_context()
    path = entry.result_path / "parameters.yaml"
    watcher = FileWatcher(context, bus, admit_reload=lambda: True)
    try:
        replacement = path.with_suffix(".tmp")
        replacement.write_text(path.read_text().replace("4300.0", "5500.0"))
        replacement.replace(path)
        qt_until(
            lambda: any(
                status.file.path == path and status.state == "reloaded"
                for status in context.get_file_sync_status()
            ),
            label="inactive entry directory reload",
            watchdog_ms=2000,
        )
        assert context.current_point_label() == "first"
        assert context.read_context().value("Q1.rate") == 4300.0
        assert captured.value("Q1.rate") == 4300.0
        assert all(
            file.path.parent != entry.result_path
            for file in context.get_file_revisions()
        )
    finally:
        watcher.close()


def test_flush_uses_current_admission_and_ignores_own_writes(qapp, selected) -> None:
    assert hasattr(adapters, "FileWatcher"), (
        "Point files need a lifecycle-owned watcher"
    )
    from zcu_tools.gui.session.adapters import FileWatcher
    from zcu_tools.gui.session.events import FileChangedPayload

    context, bus, _ = selected
    admitted = False
    watcher = FileWatcher(context, bus, admit_reload=lambda: admitted)
    facts: list[FileChangedPayload] = []
    changes: list[MdChangedPayload] = []
    bus.subscribe(FileChangedPayload, facts.append)
    bus.subscribe(MdChangedPayload, changes.append)
    file = context.get_file_revisions()[0]
    try:
        file.path.write_text(file.path.read_text().replace("4300.0", "5100.0"))
        watcher.flush()
        assert context.read_context().value("Q1.rate") == 4300.0
        assert facts[-1].status.state == "pending"
        assert changes == []
        watcher.flush()
        assert len(facts) == 1
        admitted = True
        watcher.flush()
        assert context.read_context().value("Q1.rate") == 5100.0
        assert facts[-1].status.state == "reloaded"
        assert len(changes) == 1
        watcher.flush()
        assert len(facts) == 2
        target = context.read_context().target
        context.set_parameter(target, "Q1.rate", 6100.0)
        watcher.flush()
        assert len(facts) == 2
    finally:
        watcher.close()
    file.path.write_text(file.path.read_text().replace("6100.0", "7100.0"))
    watcher.flush()
    assert context.read_context().value("Q1.rate") == 6100.0


def test_directory_event_installs_new_library_after_absent_selection(
    qapp, selected
) -> None:
    from zcu_tools.gui.session.adapters import FileWatcher

    context, bus, entry = selected
    path = entry.result_path / "points/first/library.yaml"
    path.unlink()
    context.use_point(PointUseRequest("first"))
    captured = context.read_context()
    watcher = FileWatcher(context, bus, admit_reload=lambda: True)
    try:
        path.write_text(
            'format: zcu.module-library\nformat_version: "2.0"\n'
            "components: {Q1: {waveforms: {base: {style: const, length: 0.2}}}}\n"
        )
        qt_until(
            lambda: (
                context.read_context().library.waveform_keys() == ("Q1.waveforms.base",)
            ),
            label="new library file reload",
        )
        assert captured.library.waveform_keys() == ()
        assert context.get_file_revisions()[1].revision is not None
    finally:
        watcher.close()


def test_directory_events_follow_atomic_replace_point_switch_and_missing_aliases(
    qapp, selected
) -> None:
    assert hasattr(adapters, "FileWatcher"), (
        "Watch point directories, including renamed and missing files"
    )
    from zcu_tools.gui.session.adapters import FileWatcher

    context, bus, entry = selected
    watcher = FileWatcher(context, bus, admit_reload=lambda: True)
    try:
        file = context.get_file_revisions()[0]
        replacement = file.path.with_suffix(".tmp")
        replacement.write_text(file.path.read_text().replace("4300.0", "5100.0"))
        replacement.replace(file.path)
        qt_until(
            lambda: context.read_context().value("Q1.rate") == 5100.0,
            label="point file reload",
        )
        entry.new_point("second")
        context.use_point(PointUseRequest("second"))
        alias = context.get_file_revisions()[2]
        alias.path.write_text(
            'format: zcu.aliases\nformat_version: "1.0"\naliases: {driver: Q1}\n'
        )
        qt_until(
            lambda: bool(context.get_alias_rows()), label="new aliases file reload"
        )
        file.path.write_text("invalid: true")
        watcher.flush()
        assert context.get_file_sync_status()[0].file.path == alias.path
        alias.path.write_text("invalid: true")
        watcher.flush()
        assert context.get_file_sync_status()[0].state == "error"
        context.clear_context()
        watcher.flush()
        assert context.get_file_sync_status() == ()
    finally:
        watcher.close()
