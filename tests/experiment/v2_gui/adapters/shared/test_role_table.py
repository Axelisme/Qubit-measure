"""The data-driven ROLE_TABLE + builders reproduce the role-factory golden.

Proves ``role_blank`` / ``role_ref`` over ``ROLE_TABLE`` produce byte-identical
value trees to the legacy per-role factories (the golden), so wiring the table
into ``ROLE_FACTORIES`` is a behaviour-preserving change. ``readout_dpm`` is not
in ``ROLE_TABLE`` yet (it is migrated together with its live-eval normalization),
so it is covered by the legacy-factory characterization, not here.
"""

from __future__ import annotations

from typing import Any

import pytest
from zcu_tools.experiment.v2_gui.adapters.shared.defaults.role_table import (
    ROLE_TABLE,
    role_blank,
    role_ref,
)

from .test_role_default_characterization import (
    _POPULATED_MD,
    _load_golden,
    _mk_ctx,
    _serialize,
)


def _compute_via_table(role_id: str) -> dict[str, Any]:
    role = ROLE_TABLE[role_id]
    entry: dict[str, Any] = {}
    for fx_name, md in (("empty", {}), ("pop", _POPULATED_MD)):
        entry[f"blank/{fx_name}"] = _serialize(role_blank(role, _mk_ctx(md)))
        if role.lib is not None:
            entry[f"ref/{fx_name}/opt=False"] = _serialize(role_ref(role, _mk_ctx(md)))
            entry[f"ref/{fx_name}/opt=True"] = _serialize(
                role_ref(role, _mk_ctx(md), optional=True)
            )
    return entry


@pytest.mark.parametrize("role_id", sorted(ROLE_TABLE))
def test_role_table_reproduces_golden_payload(role_id: str) -> None:
    golden = _load_golden()
    assert _compute_via_table(role_id) == golden[role_id], (
        f"ROLE_TABLE[{role_id!r}] payload differs from the role-factory golden — "
        "the data-driven builder is not behaviour-identical to the legacy factory"
    )
