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
from zcu_tools.experiment.v2_gui.adapters._support.defaults.role_factories import (
    ROLE_FACTORIES,
)

from ._role_characterization import (
    POPULATED_MD as _POPULATED_MD,
)
from ._role_characterization import (
    load_golden as _load_golden,
)
from ._role_characterization import (
    make_context as _mk_ctx,
)
from ._role_characterization import (
    serialize as _serialize,
)


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


def test_golden_covers_exactly_the_registered_roles() -> None:
    golden = _load_golden()
    assert set(golden) == set(ROLE_FACTORIES), (
        "golden role set drifted from ROLE_FACTORIES; regenerate the golden for a "
        "reviewed role add/remove"
    )


def test_readout_dpm_golden_keeps_optimized_readout_live_links() -> None:
    """The golden pins readout_dpm to live ro_optimize outputs, not snapshots."""

    payload = _load_golden()["readout_dpm"]["blank/pop"]["v"]
    pulse_cfg = payload["pulse_cfg"]
    ro_cfg = payload["ro_cfg"]

    assert pulse_cfg["freq"] == {"E": "best_ro_freq", "r": None}
    assert pulse_cfg["gain"] == {"E": "best_ro_gain", "r": None}
    assert pulse_cfg["waveform"]["v"]["length"] == {
        "E": "best_ro_length + 0.1",
        "r": None,
    }
    assert ro_cfg["ro_freq"] == {"E": "best_ro_freq", "r": None}
    assert ro_cfg["ro_length"] == {"E": "best_ro_length", "r": None}


@pytest.mark.parametrize("role_id", sorted(ROLE_FACTORIES))
def test_role_default_payload_matches_golden(role_id: str) -> None:
    golden = _load_golden()
    assert _compute_role(role_id) == golden[role_id], (
        f"role {role_id!r} default payload changed vs golden — if this is a "
        "deliberate, reviewed behavior change, regenerate the golden"
    )
