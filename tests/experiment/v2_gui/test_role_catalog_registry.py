"""The experiment-side role catalog populates the gui RoleCatalog correctly,
and every entry's value resolves to a real spec (the spec round-trip used by
create_from_role)."""

from __future__ import annotations

import pytest
from zcu_tools.experiment.v2_gui.registry import (
    ALL_ROLE_ENTRIES,
    register_all_roles,
)
from zcu_tools.gui.app.main.adapter import ExpContext
from zcu_tools.gui.app.main.cfg_schemas import _MODULE_SPEC_FACTORIES
from zcu_tools.gui.app.main.role_catalog import RoleCatalog
from zcu_tools.gui.app.main.specs import make_waveform_spec_by_style
from zcu_tools.meta_tool import MetaDict, ModuleLibrary


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


def test_blank_roles_cover_every_discriminator():
    cat = RoleCatalog()
    register_all_roles(cat)
    blanks = {m["role_id"] for m in cat.list_meta() if m["role_id"].endswith(":blank")}
    # 7 module discriminators + 6 waveform styles
    assert len(blanks) == 13
    for disc in _MODULE_SPEC_FACTORIES:
        assert f"{disc}:blank" in blanks


@pytest.mark.parametrize("entry", ALL_ROLE_ENTRIES, ids=lambda e: e.role_id)
def test_entry_value_resolves_to_spec(entry):
    """Each factory's value (md-aware OR blank) carries a discriminator that maps
    to a real spec — what create_from_role relies on (option-b spec derivation)."""
    ref = entry.make_value(_empty_ctx())
    value = ref.value
    if entry.item_kind == "module":
        disc = value.fields["type"].value
        assert disc in _MODULE_SPEC_FACTORIES
        _MODULE_SPEC_FACTORIES[disc]()  # must construct
    else:
        disc = value.fields["style"].value
        make_waveform_spec_by_style(disc)  # must construct
