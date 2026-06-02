"""PersistenceCaretaker — single-file Memento round-trip + degradation."""

from __future__ import annotations

from pathlib import Path

from zcu_tools.gui.services.caretaker import PersistenceCaretaker
from zcu_tools.gui.services.persistence_types import (
    APP_STATE_VERSION,
    AppPersistedState,
    PersistedSession,
    PersistedStartup,
    PersistedTab,
)


class _FakeOriginator:
    """A PersistOriginatorPort that just holds a memento in memory."""

    def __init__(self, state: AppPersistedState) -> None:
        self._captured = state
        self.restored: AppPersistedState | None = None

    def capture_persisted_state(self) -> AppPersistedState:
        return self._captured

    def restore_persisted_state(self, state: AppPersistedState):
        self.restored = state
        from zcu_tools.gui.services.ports import RestoreReport

        return RestoreReport(restored_tabs=len(state.session.tabs), rejected_tabs=())


def _sample_state() -> AppPersistedState:
    return AppPersistedState(
        startup=PersistedStartup(chip_name="chip", ip="host", port=1234),
        session=PersistedSession(
            tabs=(PersistedTab(adapter_name="fake", cfg_raw={"x": 1}),),
            active_tab_index=0,
        ),
    )


def test_flush_then_restore_roundtrip(tmp_path: Path):
    state = _sample_state()
    flusher = PersistenceCaretaker(_FakeOriginator(state), cache_dir=tmp_path)
    flusher.flush()

    receiver = _FakeOriginator(AppPersistedState())
    caretaker = PersistenceCaretaker(receiver, cache_dir=tmp_path)
    outcome = caretaker.restore_all()

    assert outcome.load_error is None
    assert receiver.restored == state


def test_restore_with_load_false_ignores_file_uses_defaults(tmp_path: Path):
    # A "clean" start: even though a valid file exists on disk, load=False
    # restores a default snapshot and never reads it. The file stays untouched.
    state = _sample_state()
    PersistenceCaretaker(_FakeOriginator(state), cache_dir=tmp_path).flush()

    receiver = _FakeOriginator(AppPersistedState())
    caretaker = PersistenceCaretaker(receiver, cache_dir=tmp_path)
    on_disk_before = caretaker.state_path.read_text(encoding="utf-8")

    outcome = caretaker.restore_all(load=False)

    assert outcome.load_error is None
    assert receiver.restored == AppPersistedState()  # default, not the file
    # restore must not touch the file (a later flush at close still overwrites).
    assert caretaker.state_path.read_text(encoding="utf-8") == on_disk_before


def test_restore_missing_file_uses_defaults(tmp_path: Path):
    receiver = _FakeOriginator(AppPersistedState())
    caretaker = PersistenceCaretaker(receiver, cache_dir=tmp_path)

    outcome = caretaker.restore_all()

    assert outcome.load_error is None
    assert receiver.restored == AppPersistedState()


def test_restore_corrupt_file_falls_back_with_error(tmp_path: Path):
    receiver = _FakeOriginator(AppPersistedState())
    caretaker = PersistenceCaretaker(receiver, cache_dir=tmp_path)
    caretaker._path.parent.mkdir(parents=True, exist_ok=True)
    caretaker._path.write_text("{ not json", encoding="utf-8")

    outcome = caretaker.restore_all()

    assert outcome.load_error is not None
    assert receiver.restored == AppPersistedState()


def test_restore_wrong_version_falls_back_with_error(tmp_path: Path):
    receiver = _FakeOriginator(AppPersistedState())
    caretaker = PersistenceCaretaker(receiver, cache_dir=tmp_path)
    caretaker._path.parent.mkdir(parents=True, exist_ok=True)
    caretaker._path.write_text(
        f'{{"version": {APP_STATE_VERSION + 99}, "startup": {{}}, "session": {{}}}}',
        encoding="utf-8",
    )

    outcome = caretaker.restore_all()

    assert outcome.load_error is not None
    assert receiver.restored == AppPersistedState()
