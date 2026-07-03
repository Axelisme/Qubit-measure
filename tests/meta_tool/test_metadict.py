from __future__ import annotations

import json
from typing import Any, cast

import pytest
from zcu_tools.meta_tool import ExperimentManager, MetaDict


def test_metadict_complex_roundtrip_uses_tagged_encoding(tmp_path) -> None:
    path = tmp_path / "meta.json"
    md = MetaDict(path)

    md.center = complex(1.25, -2.5)
    md.note = "123"
    md.complex_shaped_text = "(1+2j)"

    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["center"] == {"__complex__": [1.25, -2.5]}
    assert raw["note"] == "123"
    assert raw["complex_shaped_text"] == {"__metadict_string__": "(1+2j)"}

    reloaded = MetaDict(path)
    assert reloaded.center == complex(1.25, -2.5)
    assert reloaded.note == "123"
    assert reloaded.complex_shaped_text == "(1+2j)"


def test_metadict_loads_legacy_complex_string_with_warning(tmp_path) -> None:
    path = tmp_path / "meta.json"
    path.write_text(json.dumps({"center": "(1+2j)"}), encoding="utf-8")

    with pytest.warns(DeprecationWarning, match="legacy MetaDict complex strings"):
        md = MetaDict(path)
        assert md.center == complex(1.0, 2.0)


def test_metadict_does_not_restore_int_like_string(tmp_path) -> None:
    path = tmp_path / "meta.json"
    path.write_text(json.dumps({"note": "123"}), encoding="utf-8")

    md = MetaDict(path)

    assert md.note == "123"


def test_metadict_rejects_invalid_complex_tag(tmp_path) -> None:
    path = tmp_path / "meta.json"
    path.write_text(
        json.dumps({"center": {"__complex__": [True, 2.0]}}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="Invalid MetaDict complex tag"):
        MetaDict(path)


def test_metadict_rejects_protected_attribute_data_keys(tmp_path) -> None:
    md = MetaDict()

    with pytest.raises(AttributeError, match="protected MetaDict attribute"):
        setattr(md, "has_persistence", False)
    with pytest.raises(AttributeError, match="protected MetaDict attribute"):
        setattr(md, "dump", "not a method")
    with pytest.raises(AttributeError, match="protected MetaDict attribute"):
        md.update({"has_persistence": "shadow"})
    with pytest.raises(TypeError, match="MetaDict keys must be str"):
        md.update(cast(Any, {1: "not a string key"}))

    assert "has_persistence" not in dict(md.items())

    path = tmp_path / "meta.json"
    path.write_text(json.dumps({"has_persistence": "shadow"}), encoding="utf-8")
    with pytest.raises(ValueError, match="protected or invalid data key"):
        MetaDict(path)


def test_metadict_update_batches_persisted_write(tmp_path) -> None:
    path = tmp_path / "meta.json"
    md = MetaDict(path)
    md.dump()
    dump_calls = 0
    original_dump = md.dump

    def counting_dump() -> None:
        nonlocal dump_calls
        dump_calls += 1
        original_dump()

    object.__setattr__(md, "dump", counting_dump)

    md.update({"alpha": 1}, beta=2)

    assert dump_calls == 1
    reloaded = MetaDict(path)
    assert reloaded.alpha == 1
    assert reloaded.beta == 2


def test_experiment_manager_str_without_active_context(tmp_path) -> None:
    em = ExperimentManager(tmp_path)

    assert "active=None" in str(em)
