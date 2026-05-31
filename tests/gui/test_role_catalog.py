"""Unit tests for the gui-side RoleCatalog (role template registry)."""

from __future__ import annotations

import pytest
from zcu_tools.gui.role_catalog import RoleCatalog, RoleEntry


def _entry(role_id: str, kind: str) -> RoleEntry:
    return RoleEntry(role_id, role_id.title(), kind, lambda ctx: ctx)  # type: ignore[arg-type]


def test_register_and_get():
    cat = RoleCatalog()
    e = _entry("res_probe", "module")
    cat.register(e)
    assert cat.has("res_probe")
    assert cat.get("res_probe") is e


def test_duplicate_role_id_raises():
    cat = RoleCatalog()
    cat.register(_entry("res_probe", "module"))
    with pytest.raises(ValueError, match="already registered"):
        cat.register(_entry("res_probe", "module"))


def test_get_unknown_raises():
    with pytest.raises(KeyError, match="not found"):
        RoleCatalog().get("nope")


def test_entries_for_filters_by_kind_and_preserves_order():
    cat = RoleCatalog()
    cat.register(_entry("a", "module"))
    cat.register(_entry("w1", "waveform"))
    cat.register(_entry("b", "module"))
    cat.register(_entry("w2", "waveform"))

    assert [e.role_id for e in cat.entries_for("module")] == ["a", "b"]
    assert [e.role_id for e in cat.entries_for("waveform")] == ["w1", "w2"]


def test_list_meta_shape():
    cat = RoleCatalog()
    cat.register(_entry("res_probe", "module"))
    meta = cat.list_meta()
    assert meta == [
        {"role_id": "res_probe", "label": "Res_Probe", "item_kind": "module"}
    ]
