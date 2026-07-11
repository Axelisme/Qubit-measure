"""Unit tests for the gui-side RoleCatalog (role template registry)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
from zcu_tools.gui.app.main.role_catalog import RoleCatalog, RoleEntry
from zcu_tools.gui.app.main.specs import MAIN_PROGRAM_SPEC_POLICY
from zcu_tools.gui.cfg import CfgSectionSpec, LiteralSpec
from zcu_tools.gui.measure_cfg import PROGRAM_SHAPES


def _entry(role_id: str, kind: str) -> RoleEntry:
    shape = PROGRAM_SHAPES.get(kind, "pulse" if kind == "module" else "const")  # type: ignore[arg-type]
    return RoleEntry(
        role_id,
        role_id.title(),
        kind,  # type: ignore[arg-type]
        lambda: shape.make_spec(MAIN_PROGRAM_SPEC_POLICY),
        lambda ctx: ctx,  # type: ignore[arg-type]
    )


def test_register_and_get():
    cat = RoleCatalog()
    e = _entry("res_probe", "module")
    cat.register(e)
    assert cat.has("res_probe")
    assert cat.get("res_probe") is e


def test_role_entry_is_immutable_after_validation() -> None:
    entry = _entry("res_probe", "module")
    RoleCatalog().register(entry)

    with pytest.raises(FrozenInstanceError):
        entry.shape = lambda: PROGRAM_SHAPES.module("pulse").make_spec(  # type: ignore[misc]
            MAIN_PROGRAM_SPEC_POLICY
        )


def test_duplicate_role_id_raises():
    cat = RoleCatalog()
    cat.register(_entry("res_probe", "module"))
    shape_calls = 0

    def shape():
        nonlocal shape_calls
        shape_calls += 1
        return PROGRAM_SHAPES.module("pulse").make_spec(MAIN_PROGRAM_SPEC_POLICY)

    with pytest.raises(ValueError, match="already registered"):
        cat.register(
            RoleEntry(
                "res_probe",
                "Duplicate",
                "module",
                shape,
                lambda ctx: ctx,  # type: ignore[arg-type]
            )
        )
    assert shape_calls == 0


def test_register_validates_shape_once_without_materializing_value() -> None:
    shape_calls = 0
    value_calls = 0

    def shape():
        nonlocal shape_calls
        shape_calls += 1
        return PROGRAM_SHAPES.module("pulse").make_spec(MAIN_PROGRAM_SPEC_POLICY)

    def value(ctx):
        nonlocal value_calls
        value_calls += 1
        return ctx

    entry = RoleEntry("pulse", "Pulse", "module", shape, value)  # type: ignore[arg-type]
    cat = RoleCatalog()
    cat.register(entry)

    assert shape_calls == 1
    assert value_calls == 0
    assert cat.get("pulse") is entry


def test_register_rejects_shape_kind_mismatch_without_inserting() -> None:
    entry = RoleEntry(
        "wrong",
        "Wrong",
        "module",
        lambda: PROGRAM_SHAPES.waveform("const").make_spec(MAIN_PROGRAM_SPEC_POLICY),
        lambda ctx: ctx,  # type: ignore[arg-type]
    )
    cat = RoleCatalog()

    with pytest.raises(
        TypeError,
        match=r"Role 'wrong' declares kind 'module'.*root kind is 'waveform'",
    ):
        cat.register(entry)

    assert not cat.has("wrong")


def test_register_rejects_non_section_shape_without_inserting() -> None:
    entry = RoleEntry(
        "wrong",
        "Wrong",
        "module",
        lambda: object(),  # type: ignore[arg-type,return-value]
        lambda ctx: ctx,  # type: ignore[arg-type]
    )
    cat = RoleCatalog()

    with pytest.raises(TypeError, match="must return CfgSectionSpec"):
        cat.register(entry)

    assert not cat.has("wrong")


def test_register_rejects_shape_with_two_root_discriminators() -> None:
    entry = RoleEntry(
        "ambiguous",
        "Ambiguous",
        "module",
        lambda: CfgSectionSpec(
            fields={"type": LiteralSpec("pulse"), "style": LiteralSpec("const")}
        ),
        lambda ctx: ctx,  # type: ignore[arg-type]
    )
    cat = RoleCatalog()

    with pytest.raises(ValueError, match="exactly one root discriminator"):
        cat.register(entry)

    assert not cat.has("ambiguous")


def test_register_shape_factory_failure_does_not_insert() -> None:
    shape_calls = 0

    def shape():
        nonlocal shape_calls
        shape_calls += 1
        raise RuntimeError("shape failed")

    entry = RoleEntry(
        "broken",
        "Broken",
        "module",
        shape,
        lambda ctx: ctx,  # type: ignore[arg-type]
    )
    cat = RoleCatalog()

    with pytest.raises(RuntimeError, match="shape failed"):
        cat.register(entry)

    assert shape_calls == 1
    assert not cat.has("broken")


@pytest.mark.parametrize(
    ("fields", "error"),
    [
        ({}, "no string literal discriminator 'type'"),
        ({"type": CfgSectionSpec()}, "no string literal discriminator 'type'"),
        ({"type": LiteralSpec(7)}, "no string literal discriminator 'type'"),
        ({"type": LiteralSpec("unknown")}, "unknown module shape 'unknown'"),
    ],
)
def test_register_rejects_malformed_or_unknown_discriminator(
    fields: dict[str, object],
    error: str,
) -> None:
    entry = RoleEntry(
        "broken",
        "Broken",
        "module",
        lambda: CfgSectionSpec(fields=fields),  # type: ignore[arg-type]
        lambda ctx: ctx,  # type: ignore[arg-type]
    )
    cat = RoleCatalog()

    with pytest.raises(ValueError, match=error):
        cat.register(entry)

    assert not cat.has("broken")


def test_catalog_access_never_rebuilds_shape_or_value() -> None:
    shape_calls = 0
    value_calls = 0

    def shape():
        nonlocal shape_calls
        shape_calls += 1
        return PROGRAM_SHAPES.module("pulse").make_spec(MAIN_PROGRAM_SPEC_POLICY)

    def value(ctx):
        nonlocal value_calls
        value_calls += 1
        return ctx

    entry = RoleEntry("pulse", "Pulse", "module", shape, value)  # type: ignore[arg-type]
    cat = RoleCatalog()
    cat.register(entry)

    assert cat.get("pulse") is entry
    assert cat.has("pulse")
    assert cat.entries_for("module") == [entry]
    assert cat.list_meta() == [
        {
            "role_id": "pulse",
            "label": "Pulse",
            "item_kind": "module",
            "default_name": "",
        }
    ]
    assert shape_calls == 1
    assert value_calls == 0


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
        {
            "role_id": "res_probe",
            "label": "Res_Probe",
            "item_kind": "module",
            "default_name": "",
        }
    ]
