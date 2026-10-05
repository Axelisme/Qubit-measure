"""Role defaults — the declarative ``ROLE_TABLE`` + its two generic builders, plus
the shared value-tree primitives.

Each role's default is one ``RoleDef`` literal in ``role_table.py``; the
``ROLE_FACTORIES`` table consumed by fresh cfg materialization and ``RoleCatalog`` is
generated from it. See ADR-0009 / ADR-0012.
"""

from zcu_lab.v2._support.measure.defaults.helpers import make_trig_offset
from zcu_lab.v2._support.measure.defaults.helpers import patch_pulse_fields
from zcu_lab.v2._support.measure.defaults.helpers import patch_ro_cfg_fields
from zcu_lab.v2._support.measure.defaults.module_defaults import select_named_module_value
from zcu_lab.v2._support.measure.defaults.module_defaults import NamedModuleValue
from zcu_lab.v2._support.measure.defaults.role_factories import ROLE_FACTORIES
from zcu_lab.v2._support.measure.defaults.role_factories import RoleFactorySpec
from zcu_lab.v2._support.measure.defaults.role_table import ROLE_TABLE
from zcu_lab.v2._support.measure.defaults.role_table import Md
from zcu_lab.v2._support.measure.defaults.role_table import RoleDef
from zcu_lab.v2._support.measure.defaults.role_table import Source
from zcu_lab.v2._support.measure.defaults.role_table import role_blank
from zcu_lab.v2._support.measure.defaults.role_table import role_ref

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
    "make_trig_offset",
    "patch_pulse_fields",
    "patch_ro_cfg_fields",
    "select_named_module_value",
    "NamedModuleValue",
]
