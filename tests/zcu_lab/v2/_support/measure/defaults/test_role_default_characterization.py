"""Characterize canonical role factories against manually reviewed expectations.

The shared golden preserves deferred EvalValue links, inline numeric literals,
reference tags, nested shapes and disabled optional references. Expected edits
must be reviewed field by field against the frozen migration/canonical contract,
never generated from factory output. Waveform roles have blank variants only.
Retire this migration pin when the role-factory facade is retired; the role-table
owner retains the default-tree contract.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.zcu_lab.v2._support.measure._role_characterization import (
    POPULATED_QUBIT,
    POPULATED_RESONATOR,
)
from tests.zcu_lab.v2._support.measure._role_characterization import (
    load_golden as _load_golden,
)
from tests.zcu_lab.v2._support.measure._role_characterization import (
    make_context as _mk_ctx,
)
from tests.zcu_lab.v2._support.measure._role_characterization import (
    serialize as _serialize,
)
from zcu_lab.v2._support.measure.defaults.role_factories import ROLE_FACTORIES


def _compute_role(role_id: str) -> dict[str, Any]:
    """Recompute the serialized blank + ref payloads for one role."""
    spec = ROLE_FACTORIES[role_id]
    entry: dict[str, Any] = {}
    for fx_name, ctx in (
        ("empty", _mk_ctx()),
        ("pop", _mk_ctx(qubit=POPULATED_QUBIT, resonator=POPULATED_RESONATOR)),
    ):
        entry[f"blank/{fx_name}"] = _serialize(spec.blank(ctx))
        if spec.ref is not None:
            entry[f"ref/{fx_name}/opt=False"] = _serialize(spec.ref(ctx))
            entry[f"ref/{fx_name}/opt=True"] = _serialize(spec.ref(ctx, optional=True))
    return entry


@pytest.mark.parametrize("role_id", sorted(ROLE_FACTORIES))
def test_role_default_payload_matches_golden(role_id: str) -> None:
    golden = _load_golden()
    assert _compute_role(role_id) == golden[role_id], (
        f"role {role_id!r} default payload differs from the manually reviewed "
        "canonical golden"
    )
