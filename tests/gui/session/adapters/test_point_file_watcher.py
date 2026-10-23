"""Real-file watcher admission, deterministic debounce draining and teardown."""

from zcu_tools.gui.session import adapters
from zcu_tools.gui.session.context_types import PointUseRequest
from zcu_tools.gui.session.events import MdChangedPayload

from tests.gui._qt_waits import qt_until


def test_flush_uses_current_admission_and_ignores_own_writes(qapp, selected) -> None:
    assert hasattr(adapters, "PointFileWatcher"), (
        "Point files need a lifecycle-owned watcher"
    )
    from zcu_tools.gui.session.adapters import PointFileWatcher
    from zcu_tools.gui.session.events import PointFileChangedPayload

    context, bus, _ = selected
    admitted = False
    watcher = PointFileWatcher(context, bus, admit_reload=lambda: admitted)
    facts: list[PointFileChangedPayload] = []
    changes: list[MdChangedPayload] = []
    bus.subscribe(PointFileChangedPayload, facts.append)
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


def test_directory_events_follow_atomic_replace_point_switch_and_missing_aliases(
    qapp, selected
) -> None:
    assert hasattr(adapters, "PointFileWatcher"), (
        "Watch point directories, including renamed and missing files"
    )
    from zcu_tools.gui.session.adapters import PointFileWatcher

    context, bus, entry = selected
    watcher = PointFileWatcher(context, bus, admit_reload=lambda: True)
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
