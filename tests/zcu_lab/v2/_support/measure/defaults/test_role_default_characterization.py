"""Characterization snapshot of every role factory's default value tree.

This pins the EXACT value tree each ``ROLE_FACTORIES`` entry produces — for the
blank factory and (where present) the ref factory under optional True/False — so
the role-engine data-ization refactor cannot silently change a default payload.
The golden lives in ``_role_default_golden.json`` next to this file; regenerate it
ONLY for a deliberate, reviewed behavior change (e.g. the readout_dpm live-eval
normalization) by running ``_serialize`` over the current factories.

The serialization preserves the DirectValue-vs-EvalValue distinction (the GUI's
live-vs-snapshot behavior), the chosen ref tag, nested ref shapes, and the
None disabled-optional-ref state (ADR-0010) — i.e. everything the refactor must
keep identical.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.zcu_lab.v2._support.measure._role_characterization import (
    POPULATED_MD as _POPULATED_MD,
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
    for fx_name, md in (("empty", {}), ("pop", _POPULATED_MD)):
        entry[f"blank/{fx_name}"] = _serialize(spec.blank(_mk_ctx(md)))
        if spec.ref is not None:
            entry[f"ref/{fx_name}/opt=False"] = _serialize(spec.ref(_mk_ctx(md)))
            entry[f"ref/{fx_name}/opt=True"] = _serialize(
                spec.ref(_mk_ctx(md), optional=True)
            )
    return entry


@pytest.mark.parametrize("role_id", sorted(ROLE_FACTORIES))
def test_role_default_payload_matches_golden(role_id: str) -> None:
    golden = _load_golden()
    assert _compute_role(role_id) == golden[role_id], (
        f"role {role_id!r} default payload changed vs golden — if this is a "
        "deliberate, reviewed behavior change, regenerate the golden"
    )
