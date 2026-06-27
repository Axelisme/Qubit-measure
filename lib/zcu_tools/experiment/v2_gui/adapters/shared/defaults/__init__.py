"""Role defaults — the declarative ``ROLE_TABLE`` + its two generic builders, plus
the shared value-tree primitives.

Each role's default is one ``RoleDef`` literal in ``role_table.py``; the
``ROLE_FACTORIES`` table consumed by ``CfgBuilder`` and the GUI ``RoleCatalog`` is
generated from it. See ADR-0009 / ADR-0012.
"""

from .helpers import (
    make_default_value,
    make_trig_offset,
    patch_pulse_fields,
    patch_ro_cfg_fields,
    select_named_module_value,
)
from .module_defaults import NamedModuleValue
from .role_factories import ROLE_FACTORIES, RoleFactorySpec
from .role_table import ROLE_TABLE, Md, RoleDef, Source, role_blank, role_ref

__all__ = [
    # the role vocabulary as data + its generated factory table
    "ROLE_TABLE",
    "ROLE_FACTORIES",
    "RoleFactorySpec",
    "RoleDef",
    "Md",
    "Source",
    "role_blank",
    "role_ref",
    # shared value-tree primitives
    "make_default_value",
    "make_trig_offset",
    "patch_pulse_fields",
    "patch_ro_cfg_fields",
    "select_named_module_value",
    "NamedModuleValue",
]
