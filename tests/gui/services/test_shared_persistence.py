from __future__ import annotations

import json
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


class _Originator:
    def __init__(self, state: _State = _State()) -> None:
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


def test_safe_cache_root_falls_back_when_platform_lookup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        persistence_module,
        "user_cache_dir",
        lambda *args: (_ for _ in ()).throw(RuntimeError("platform failure")),
    )
    monkeypatch.setattr(persistence_module, "gettempdir", lambda: "/fallback")

    caretaker = SingleFileCaretaker(
        _Originator(), codec=_Codec(), filename="state.json"
    )

    assert caretaker.state_path == Path("/fallback/.zcu_tools_gui/state.json")


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

    monkeypatch.setattr(
        persistence_module, "NamedTemporaryFile", fake_named_temporary_file
    )

    with pytest.raises(PersistenceError) as raised:
        caretaker.flush()

    assert raised.value.__cause__ is expected
    assert caretaker.state_path.read_text(encoding="utf-8") == "old"
    assert not temp_path.exists()


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
