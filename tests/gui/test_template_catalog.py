"""Unit tests for the gui-side TemplateCatalog (role template registry)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock

import pytest
from zcu_tools.experiment.cfg_editing import PROGRAM_SHAPES
from zcu_tools.gui.app.measure.specs import MAIN_PROGRAM_SPEC_POLICY
from zcu_tools.gui.app.measure.template_catalog import (
    TemplateCatalog,
    TemplateEntry,
    TemplateItemKind,
)
from zcu_tools.gui.cfg import CfgNodeSpec, CfgSectionSpec, LiteralSpec


def _entry(template_id: str, kind: TemplateItemKind) -> TemplateEntry:
    shape = PROGRAM_SHAPES.get(kind, "pulse" if kind == "module" else "const")
    return TemplateEntry(
        template_id,
        template_id.title(),
        kind,
        lambda: shape.make_spec(MAIN_PROGRAM_SPEC_POLICY),
        MagicMock(),
    )


def test_register_and_get():
    cat = TemplateCatalog()
    e = _entry("res_probe", "module")
    cat.register(e)
    assert cat.has("res_probe")
    assert cat.get("res_probe") is e


def test_template_entry_is_immutable_after_validation() -> None:
    entry = _entry("res_probe", "module")
    TemplateCatalog().register(entry)

    with pytest.raises(FrozenInstanceError):
        entry.__setattr__(
            "shape",
            lambda: PROGRAM_SHAPES.module("pulse").make_spec(MAIN_PROGRAM_SPEC_POLICY),
        )


def test_duplicate_template_id_raises():
    cat = TemplateCatalog()
    cat.register(_entry("res_probe", "module"))
    shape_calls = 0

    def shape():
        nonlocal shape_calls
        shape_calls += 1
        return PROGRAM_SHAPES.module("pulse").make_spec(MAIN_PROGRAM_SPEC_POLICY)

    with pytest.raises(ValueError, match="already registered"):
        cat.register(
            TemplateEntry(
                "res_probe",
                "Duplicate",
                "module",
                shape,
                MagicMock(),
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

    entry = TemplateEntry(
        "pulse", "Pulse", "module", shape, MagicMock(side_effect=value)
    )
    cat = TemplateCatalog()
    cat.register(entry)

    assert shape_calls == 1
    assert value_calls == 0
    assert cat.get("pulse") is entry


def test_register_rejects_shape_kind_mismatch_without_inserting() -> None:
    entry = TemplateEntry(
        "wrong",
        "Wrong",
        "module",
        lambda: PROGRAM_SHAPES.waveform("const").make_spec(MAIN_PROGRAM_SPEC_POLICY),
        MagicMock(),
    )
    cat = TemplateCatalog()

    with pytest.raises(
        TypeError,
        match=r"Template 'wrong' declares kind 'module'.*root kind is 'waveform'",
    ):
        cat.register(entry)

    assert not cat.has("wrong")


def test_register_rejects_non_section_shape_without_inserting() -> None:
    entry = TemplateEntry(
        "wrong",
        "Wrong",
        "module",
        MagicMock(return_value=object()),
        MagicMock(),
    )
    cat = TemplateCatalog()

    with pytest.raises(TypeError, match="must return CfgSectionSpec"):
        cat.register(entry)

    assert not cat.has("wrong")


def test_register_rejects_shape_with_two_root_discriminators() -> None:
    entry = TemplateEntry(
        "ambiguous",
        "Ambiguous",
        "module",
        lambda: CfgSectionSpec(
            fields={"type": LiteralSpec("pulse"), "style": LiteralSpec("const")}
        ),
        MagicMock(),
    )
    cat = TemplateCatalog()

    with pytest.raises(ValueError, match="exactly one root discriminator"):
        cat.register(entry)

    assert not cat.has("ambiguous")


def test_register_shape_factory_failure_does_not_insert() -> None:
    shape_calls = 0

    def shape():
        nonlocal shape_calls
        shape_calls += 1
        raise RuntimeError("shape failed")

    entry = TemplateEntry(
        "broken",
        "Broken",
        "module",
        shape,
        MagicMock(),
    )
    cat = TemplateCatalog()

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
    fields: dict[str, CfgNodeSpec],
    error: str,
) -> None:
    entry = TemplateEntry(
        "broken",
        "Broken",
        "module",
        lambda: CfgSectionSpec(fields=fields),
        MagicMock(),
    )
    cat = TemplateCatalog()

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

    entry = TemplateEntry(
        "pulse", "Pulse", "module", shape, MagicMock(side_effect=value)
    )
    cat = TemplateCatalog()
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
        TemplateCatalog().get("nope")


def test_entries_for_filters_by_kind_and_preserves_order():
    cat = TemplateCatalog()
    cat.register(_entry("a", "module"))
    cat.register(_entry("w1", "waveform"))
    cat.register(_entry("b", "module"))
    cat.register(_entry("w2", "waveform"))

    assert [e.template_id for e in cat.entries_for("module")] == ["a", "b"]
    assert [e.template_id for e in cat.entries_for("waveform")] == ["w1", "w2"]


def test_list_meta_shape():
    cat = TemplateCatalog()
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
