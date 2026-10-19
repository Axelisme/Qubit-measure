from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import pytest
import zcu_tools.gui.session.persistence as persistence_module
from zcu_tools.gui.session.persistence import PersistenceError, SingleFileCaretaker


@dataclass(frozen=True)
class _State:
    value: int = 0


class _Codec:
    def default(self) -> _State:
        return _State()

    def decode(self, raw: object) -> _State:
        if not isinstance(raw, dict) or not isinstance(raw.get("value"), int):
            raise ValueError("invalid snapshot")
        return _State(raw["value"])

    def encode(self, state: _State) -> object:
        return {"value": state.value}


_DEFAULT_STATE = _State()


class _Originator:
    def __init__(self, state: _State = _DEFAULT_STATE) -> None:
        self.state = state
        self.restored: _State | None = None

    def capture_persisted_state(self) -> _State:
        return self.state

    def restore_persisted_state(self, state: _State) -> str:
        self.restored = state
        return f"restored:{state.value}"


def _caretaker(
    originator: _Originator, tmp_path: Path
) -> SingleFileCaretaker[_State, str]:
    return SingleFileCaretaker(
        originator, codec=_Codec(), filename="state.json", cache_dir=tmp_path
    )


def test_round_trip_is_typed_and_recaptures_each_flush(tmp_path: Path) -> None:
    originator = _Originator(_State(1))
    caretaker = _caretaker(originator, tmp_path)
    caretaker.flush()
    originator.state = _State(2)
    caretaker.flush()

    receiver = _Originator()
    outcome = _caretaker(receiver, tmp_path).restore_all()

    assert outcome.report == "restored:2"
    assert outcome.load_error is None
    assert receiver.restored == _State(2)


@pytest.mark.parametrize("payload", ["{", '{"value": "bad"}'])
def test_read_or_decode_failure_restores_default_with_error(
    tmp_path: Path, payload: str
) -> None:
    caretaker = _caretaker(_Originator(), tmp_path)
    caretaker.state_path.write_text(payload, encoding="utf-8")

    outcome = caretaker.restore_all()

    assert outcome.report == "restored:0"
    assert isinstance(outcome.load_error, PersistenceError)


@pytest.mark.parametrize("payload", [b"{", b'{"value": "bad"}', b"\\xff"])
def test_bad_snapshot_opens_defaults_and_leaves_file_until_normal_save(
    tmp_path: Path, payload: bytes
) -> None:
    originator = _Originator(_State(7))
    caretaker = _caretaker(originator, tmp_path)
    caretaker.state_path.write_bytes(payload)

    outcome = caretaker.restore_all()

    assert outcome.report == "restored:0"
    assert originator.restored == _State()
    assert "using defaults" in str(outcome.load_error)
    assert caretaker.state_path.is_file()
    assert caretaker.state_path.read_bytes() == payload
    caretaker.flush()
    assert json.loads(caretaker.state_path.read_text()) == {"value": 7}


@pytest.mark.parametrize("stage", ["decode", "restore"])
def test_unexpected_provider_failures_propagate_without_backup_or_defaults(
    tmp_path: Path, stage: str
) -> None:
    expected = RuntimeError("provider unavailable")

    class ProviderCodec(_Codec):
        def decode(self, raw: object) -> _State:
            if stage == "decode":
                raise expected
            return super().decode(raw)

    class ProviderOriginator(_Originator):
        def restore_persisted_state(self, state: _State) -> str:
            raise expected

    originator = ProviderOriginator()
    caretaker = SingleFileCaretaker(
        originator, codec=ProviderCodec(), filename="state.json", cache_dir=tmp_path
    )
    payload = b'{"value": 9}'
    caretaker.state_path.write_bytes(payload)

    with pytest.raises(RuntimeError) as raised:
        caretaker.restore_all()

    assert raised.value is expected
    assert originator.restored is None
    assert caretaker.state_path.read_bytes() == payload


def test_missing_file_restores_default_without_error(tmp_path: Path) -> None:
    originator = _Originator(_State(9))

    outcome = _caretaker(originator, tmp_path).restore_all()

    assert outcome.report == "restored:0"
    assert outcome.load_error is None
    assert originator.restored == _State()


def test_metadata_lookup_failure_restores_default_with_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    originator = _Originator(_State(9))
    caretaker = _caretaker(originator, tmp_path)
    expected = OSError("metadata unavailable")

    def fail_exists(path: Path) -> bool:
        if path == caretaker.state_path:
            raise expected
        return True

    monkeypatch.setattr(Path, "exists", fail_exists)

    outcome = caretaker.restore_all()

    assert outcome.report == "restored:0"
    assert isinstance(outcome.load_error, PersistenceError)
    assert "metadata unavailable" in str(outcome.load_error)
    assert originator.restored == _State()


def test_default_cache_uses_shared_root_for_save_and_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ZCU_TOOLS_CACHE_DIR", str(tmp_path))
    caretaker = SingleFileCaretaker(
        _Originator(_State(7)), codec=_Codec(), filename="state.json"
    )
    assert caretaker.state_path == tmp_path / "gui" / "state.json"
    caretaker.flush()

    receiver = _Originator()
    outcome = SingleFileCaretaker(
        receiver, codec=_Codec(), filename="state.json"
    ).restore_all()
    assert outcome.load_error is None
    assert receiver.restored == _State(7)


@pytest.mark.parametrize("override", ["", "relative/cache"])
def test_invalid_cache_override_fails_at_caretaker_entry(
    override: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ZCU_TOOLS_CACHE_DIR", override)
    with pytest.raises(ValueError, match="ZCU_TOOLS_CACHE_DIR"):
        SingleFileCaretaker(_Originator(), codec=_Codec(), filename="state.json")


@pytest.mark.parametrize("override", ["", "relative/cache"])
def test_explicit_cache_dir_bypasses_invalid_cache_override(
    tmp_path: Path, override: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ZCU_TOOLS_CACHE_DIR", override)
    caretaker = _caretaker(_Originator(_State(8)), tmp_path)
    assert caretaker.state_path == tmp_path / "state.json"
    caretaker.flush()
    receiver = _Originator()
    _caretaker(receiver, tmp_path).restore_all()
    assert receiver.restored == _State(8)


def test_load_false_does_not_read_or_replace_existing_file(tmp_path: Path) -> None:
    caretaker = _caretaker(_Originator(), tmp_path)
    caretaker.state_path.write_text("not-json", encoding="utf-8")

    outcome = caretaker.restore_all(load=False)

    assert outcome.load_error is None
    assert caretaker.state_path.read_text(encoding="utf-8") == "not-json"


@pytest.mark.parametrize("failure", ["capture", "encode"])
def test_capture_and_encode_failures_are_typed(tmp_path: Path, failure: str) -> None:
    originator = _Originator()
    codec = _Codec()
    if failure == "capture":
        originator.capture_persisted_state = lambda: (_ for _ in ()).throw(  # type: ignore[method-assign]
            RuntimeError("capture failed")
        )
    else:
        codec.encode = lambda state: (_ for _ in ()).throw(  # type: ignore[method-assign]
            TypeError("encode failed")
        )
    caretaker = SingleFileCaretaker(
        originator, codec=codec, filename="state.json", cache_dir=tmp_path
    )

    with pytest.raises(PersistenceError) as raised:
        caretaker.flush()

    assert isinstance(raised.value.__cause__, (RuntimeError, TypeError))


def test_restore_exception_propagates_unchanged(tmp_path: Path) -> None:
    originator = _Originator()
    expected = RuntimeError("restore failed")
    originator.restore_persisted_state = lambda state: (_ for _ in ()).throw(  # type: ignore[method-assign]
        expected
    )

    with pytest.raises(RuntimeError) as raised:
        _caretaker(originator, tmp_path).restore_all()

    assert raised.value is expected


def test_mkdir_failure_is_typed_and_preserves_existing_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    caretaker = _caretaker(_Originator(_State(2)), tmp_path)
    caretaker.state_path.write_text("old", encoding="utf-8")
    expected = OSError("mkdir failed")
    original_mkdir = Path.mkdir

    def fail_parent_mkdir(
        path: Path,
        mode: int = 0o777,
        parents: bool = False,
        exist_ok: bool = False,
    ) -> None:
        if path == tmp_path:
            raise expected
        original_mkdir(path, mode=mode, parents=parents, exist_ok=exist_ok)

    monkeypatch.setattr(Path, "mkdir", fail_parent_mkdir)

    with pytest.raises(PersistenceError) as raised:
        caretaker.flush()

    assert raised.value.__cause__ is expected
    assert caretaker.state_path.read_text(encoding="utf-8") == "old"


@pytest.mark.parametrize("stage", ["open", "write", "close"])
def test_temporary_file_failures_are_typed_and_cleanup_created_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    caretaker = _caretaker(_Originator(_State(2)), tmp_path)
    caretaker.state_path.write_text("old", encoding="utf-8")
    temp_path = tmp_path / "temporary"
    expected = OSError(f"temp {stage} failed")

    class FailingTemporaryFile:
        name = str(temp_path)

        def __enter__(self) -> FailingTemporaryFile:
            temp_path.touch()
            return self

        def write(self, value: str) -> int:
            if stage == "write":
                raise expected
            temp_path.write_text(value, encoding="utf-8")
            return len(value)

        def __exit__(self, *args: object) -> None:
            if stage == "close" and args[0] is None:
                raise expected

    def fake_named_temporary_file(
        *args: object, **kwargs: object
    ) -> FailingTemporaryFile:
        if stage == "open":
            raise expected
        return FailingTemporaryFile()

    # The imported temporary-file boundary injects open/write/close I/O failures.
    monkeypatch.setattr(
        persistence_module, "NamedTemporaryFile", fake_named_temporary_file
    )

    with pytest.raises(PersistenceError) as raised:
        caretaker.flush()

    assert raised.value.__cause__ is expected
    assert caretaker.state_path.read_text(encoding="utf-8") == "old"
    assert not temp_path.exists()


@pytest.mark.parametrize("persistent", [False, True])
def test_windows_lock_retry_preserves_snapshot_and_owner_error_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, persistent: bool
) -> None:
    caretaker = _caretaker(_Originator(_State(2)), tmp_path)
    caretaker.state_path.write_text('{"value": 1}', encoding="utf-8")

    class WindowsLockError(OSError):
        winerror = 32

    expected = WindowsLockError("snapshot briefly locked")
    original_replace = Path.replace
    attempts = 0
    elapsed = 0.0

    def advance(seconds: float) -> None:
        nonlocal elapsed
        elapsed += seconds

    def replace_snapshot(source: Path, target: Path) -> Path:
        nonlocal attempts
        attempts += 1
        if persistent or attempts == 1:
            assert json.loads(target.read_text(encoding="utf-8")) == {"value": 1}
            raise expected
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", replace_snapshot)
    monkeypatch.setattr(time, "monotonic", lambda: elapsed)
    monkeypatch.setattr(time, "sleep", advance)
    failure: PersistenceError | None = None
    try:
        caretaker.flush()
    except PersistenceError as error:
        failure = error

    if persistent:
        assert failure is not None
        assert failure.__cause__ is expected
        assert json.loads(caretaker.state_path.read_text()) == {"value": 1}
        assert 0.45 <= elapsed <= 0.55
    else:
        assert failure is None, "a transient sharing violation must recover"
        assert json.loads(caretaker.state_path.read_text()) == {"value": 2}
    assert attempts > 1
    assert list(tmp_path.iterdir()) == [caretaker.state_path]


def test_replace_failure_preserves_target_and_cleans_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    caretaker = _caretaker(_Originator(_State(2)), tmp_path)
    caretaker.state_path.write_text(json.dumps({"value": 1}), encoding="utf-8")
    original_replace = Path.replace

    def fail_temp_replace(path: Path, target: Path) -> Path:
        if target == caretaker.state_path:
            raise OSError("replace failed")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_temp_replace)

    with pytest.raises(PersistenceError) as raised:
        caretaker.flush()

    assert isinstance(raised.value.__cause__, OSError)
    assert json.loads(caretaker.state_path.read_text(encoding="utf-8")) == {"value": 1}
    assert list(tmp_path.iterdir()) == [caretaker.state_path]
