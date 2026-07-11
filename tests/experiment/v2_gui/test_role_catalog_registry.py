"""The experiment-side role catalog populates the gui RoleCatalog correctly,
and every entry's value resolves to a real spec (the spec round-trip used by
create_from_role)."""

from __future__ import annotations

import hashlib
import json

import pytest
from zcu_tools.experiment.v2_gui.registry import (
    ALL_ROLE_ENTRIES,
    register_all_roles,
)
from zcu_tools.gui.app.main.adapter import ExpContext
from zcu_tools.gui.app.main.role_catalog import RoleCatalog, RoleEntry
from zcu_tools.gui.cfg import DirectValue, LiteralSpec, make_custom_reference_key
from zcu_tools.gui.measure_cfg import PROGRAM_SHAPES
from zcu_tools.meta_tool import MetaDict, ModuleLibrary

from .adapters._support.test_role_default_characterization import (
    _GOLDEN_PATH,
    _serialize,
)


def _empty_ctx() -> ExpContext:
    return ExpContext(md=MetaDict(), ml=ModuleLibrary(), soc=None, soccfg=None)


def test_register_all_roles_populates_catalog():
    cat = RoleCatalog()
    register_all_roles(cat)
    ids = {m["role_id"] for m in cat.list_meta()}
    # md-aware roles + every :blank shape (incl waveform-only blanks)
    assert {"res_probe", "bath_reset", "pi_pulse", "res_waveform"} <= ids
    assert {"pulse:blank", "reset/bath:blank", "drag:blank", "arb:blank"} <= ids
    assert len(cat.list_meta()) == len(ALL_ROLE_ENTRIES)


def test_registering_all_roles_validates_shape_26_times_and_value_zero() -> None:
    shape_calls = 0
    value_calls = 0
    catalog = RoleCatalog()

    for source in ALL_ROLE_ENTRIES:

        def shape(source=source):
            nonlocal shape_calls
            shape_calls += 1
            return source.shape()

        def value(ctx, source=source):
            nonlocal value_calls
            value_calls += 1
            return source.make_value(ctx)

        catalog.register(
            RoleEntry(
                source.role_id,
                source.label,
                source.item_kind,
                shape,
                value,
                source.default_name,
            )
        )

    assert shape_calls == 26
    assert value_calls == 0
    assert len(catalog.list_meta()) == 26


def test_blank_roles_cover_every_discriminator():
    cat = RoleCatalog()
    register_all_roles(cat)
    blanks = {m["role_id"] for m in cat.list_meta() if m["role_id"].endswith(":blank")}
    # 7 module discriminators + 6 waveform styles
    assert len(blanks) == 13
    for disc in (shape.discriminator for shape in PROGRAM_SHAPES.modules()):
        assert f"{disc}:blank" in blanks


def test_waveform_blank_roles_preserve_legacy_order_and_values() -> None:
    waveform_blanks = [
        entry
        for entry in ALL_ROLE_ENTRIES
        if entry.item_kind == "waveform" and entry.role_id.endswith(":blank")
    ]
    discriminators = ["const", "cosine", "gauss", "drag", "flat_top", "arb"]

    assert [entry.role_id for entry in waveform_blanks] == [
        f"{discriminator}:blank" for discriminator in discriminators
    ]
    for entry, discriminator in zip(waveform_blanks, discriminators, strict=True):
        reference = entry.make_value(_empty_ctx())
        assert reference.chosen_key == make_custom_reference_key(discriminator)
        assert reference.value.fields["style"] == DirectValue(discriminator)


def test_all_role_metadata_preserves_exact_legacy_order() -> None:
    expected = [
        ("res_probe", "Resonator probe", "module", "readout_rf"),
        ("readout", "Pulse readout", "module", "readout_rf"),
        ("readout_dpm", "Optimized readout (DPM)", "module", "readout_dpm"),
        ("direct_readout", "Direct readout", "module", "readout_direct"),
        ("qub_probe", "Qubit probe pulse", "module", "qub_pulse"),
        ("pi_pulse", "Pi pulse", "module", "pi_amp"),
        ("pi2_pulse", "Pi/2 pulse", "module", "pi2_amp"),
        ("none_reset", "No reset", "module", "reset_none"),
        ("reset", "Pulse reset", "module", "reset_10"),
        ("two_pulse_reset", "Two-pulse reset", "module", "reset_120"),
        ("bath_reset", "Bath reset", "module", "reset_bath"),
        ("qub_waveform", "Qubit drive waveform", "waveform", "qub_flat"),
        ("res_waveform", "Res-probe waveform", "waveform", "ro_waveform"),
    ]
    expected.extend(
        (
            f"{shape.discriminator}:blank",
            f"Blank: {shape.discriminator}",
            shape.kind,
            "",
        )
        for shape in (*PROGRAM_SHAPES.modules(), *PROGRAM_SHAPES.waveforms())
    )

    assert [
        (entry.role_id, entry.label, entry.item_kind, entry.default_name)
        for entry in ALL_ROLE_ENTRIES
    ] == expected
    assert len(ALL_ROLE_ENTRIES) == 26


@pytest.mark.parametrize("entry", ALL_ROLE_ENTRIES, ids=lambda e: e.role_id)
def test_entry_carries_fresh_canonical_shape_matching_legacy_value(entry) -> None:
    first = entry.shape()
    second = entry.shape()
    key = "type" if entry.item_kind == "module" else "style"
    literal = first.fields[key]
    assert isinstance(literal, LiteralSpec)
    assert isinstance(literal.value, str)
    assert first is not second

    ref = entry.make_value(_empty_ctx())
    assert ref.value.fields[key] == DirectValue(literal.value)
    assert PROGRAM_SHAPES.get(entry.item_kind, literal.value).kind == entry.item_kind


def _role_value_payload(entries) -> dict[str, object]:
    return {
        entry.role_id: _serialize(entry.make_value(_empty_ctx())) for entry in entries
    }


def _ordered_json_oracle(payload: object) -> tuple[int, str]:
    encoded = json.dumps(
        payload,
        sort_keys=False,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return len(encoded), hashlib.sha256(encoded).hexdigest()


def test_named_role_values_match_tracked_golden_and_exact_hash() -> None:
    named_entries = [
        entry for entry in ALL_ROLE_ENTRIES if not entry.role_id.endswith(":blank")
    ]
    tracked = json.loads(_GOLDEN_PATH.read_text())
    expected = {
        entry.role_id: tracked[entry.role_id]["blank/empty"] for entry in named_entries
    }
    actual = _role_value_payload(named_entries)

    assert actual == expected
    assert _ordered_json_oracle(actual) == (
        4886,
        "ea9e0c1f883dda4f74421e44774b10cc596a7f5cb124673fc755e8ae727e6ac3",
    )


def test_structural_blank_values_preserve_exact_ordered_hash() -> None:
    blank_entries = [
        entry for entry in ALL_ROLE_ENTRIES if entry.role_id.endswith(":blank")
    ]
    payload = _role_value_payload(blank_entries)

    assert len(blank_entries) == 13
    assert _ordered_json_oracle(payload) == (
        3982,
        "d0ae77e39560a0f980c31e6cb55ab17abe3f92c25299117fb58c2ba5982ef954",
    )


def test_all_26_role_pairs_preserve_exact_ordered_hash() -> None:
    payload = [
        (entry.role_id, _serialize(entry.make_value(_empty_ctx())))
        for entry in ALL_ROLE_ENTRIES
    ]

    assert len(payload) == 26
    assert _ordered_json_oracle(payload) == (
        8919,
        "6b913a3f73cf35ae081086b53f3f3a6ecfde60e81cb2140b8dd5585b71bf19ae",
    )
