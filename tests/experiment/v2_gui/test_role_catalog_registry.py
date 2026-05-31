"""The experiment-side role catalog populates the gui RoleCatalog correctly,
and every entry's value resolves to a real spec (the spec round-trip used by
create_from_role)."""

from __future__ import annotations

import pytest

from zcu_tools.experiment.v2_gui.role_catalog_registry import (
    ROLE_ENTRIES,
    register_all_roles,
)
from zcu_tools.gui.adapter import ExpContext
from zcu_tools.gui.cfg_schemas import _MODULE_SPEC_FACTORIES
from zcu_tools.gui.role_catalog import RoleCatalog
from zcu_tools.gui.specs import make_waveform_spec_by_style
from zcu_tools.meta_tool import MetaDict, ModuleLibrary


def _empty_ctx() -> ExpContext:
    return ExpContext(md=MetaDict(), ml=ModuleLibrary(), soc=None, soccfg=None)


def test_register_all_roles_populates_catalog():
    cat = RoleCatalog()
    register_all_roles(cat)
    ids = {m["role_id"] for m in cat.list_meta()}
    assert {"res_probe", "bath_reset", "pi_pulse", "res_waveform"} <= ids
    assert len(cat.list_meta()) == len(ROLE_ENTRIES)


@pytest.mark.parametrize("entry", ROLE_ENTRIES, ids=lambda e: e.role_id)
def test_entry_value_resolves_to_spec(entry):
    """Each factory's value carries a discriminator that maps to a real spec —
    this is what create_from_role relies on (option-b spec derivation)."""
    ref = entry.make_value(_empty_ctx())
    value = ref.value
    if entry.item_kind == "module":
        disc = value.fields["type"].value
        assert disc in _MODULE_SPEC_FACTORIES
        _MODULE_SPEC_FACTORIES[disc]()  # must construct
    else:
        disc = value.fields["style"].value
        make_waveform_spec_by_style(disc)  # must construct
