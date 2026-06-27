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

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from zcu_tools.experiment.v2_gui.adapters.shared.defaults.role_factories import (
    ROLE_FACTORIES,
)
from zcu_tools.gui.app.main.adapter import (
    CfgSectionValue,
    DirectValue,
    EvalValue,
    ModuleRefValue,
    SweepValue,
    WaveformRefValue,
)

_GOLDEN_PATH = Path(__file__).with_name("_role_default_golden.json")

# A representative populated MetaDict: every md key the role factories read, so
# present keys lower to EvalValue and absent keys to their DirectValue fallback.
_POPULATED_MD = {
    "q_f": 4200.0,
    "qub_ch": 4,
    "r_f": 6500.0,
    "res_ch": 3,
    "ro_ch": 1,
    "timeFly": 0.7,
    "best_ro_freq": 6300.0,
    "best_ro_gain": 0.22,
    "best_ro_length": 2.0,
}


def _mk_ctx(md: dict[str, Any]) -> MagicMock:
    """A ctx whose md answers from ``md`` and whose ml is empty (no adoption —
    the library-lookup selector is exercised by test_role_factories / kept
    verbatim; this snapshot pins the blank-seed payload)."""
    ctx = MagicMock()
    ctx.md.get.side_effect = lambda k, d=None: md.get(k, d)
    ctx.md.__contains__ = lambda _self, k: k in md
    ml = MagicMock()
    ml.modules = {}
    ml.waveforms = {}
    ctx.ml = ml
    return ctx


def _serialize(node: object) -> Any:
    """Normalize a value-tree node to a JSON-comparable structure that keeps the
    DirectValue/EvalValue distinction and ref/section nesting."""
    if node is None:
        return None
    if isinstance(node, DirectValue):
        return {"D": node.value}
    if isinstance(node, EvalValue):
        return {"E": node.expr, "r": node.resolved}
    if isinstance(node, SweepValue):
        return {"sweep": [node.start, node.stop, node.expts, round(node.step, 9)]}
    if isinstance(node, (ModuleRefValue, WaveformRefValue)):
        return {
            "ref": node.chosen_key,
            "ov": node.is_overridden,
            "v": _serialize(node.value),
        }
    if isinstance(node, CfgSectionValue):
        return {k: _serialize(v) for k, v in node.fields.items()}
    return repr(node)


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


def _load_golden() -> dict[str, Any]:
    return json.loads(_GOLDEN_PATH.read_text())


def test_golden_covers_exactly_the_registered_roles() -> None:
    golden = _load_golden()
    assert set(golden) == set(ROLE_FACTORIES), (
        "golden role set drifted from ROLE_FACTORIES; regenerate the golden for a "
        "reviewed role add/remove"
    )


@pytest.mark.parametrize("role_id", sorted(ROLE_FACTORIES))
def test_role_default_payload_matches_golden(role_id: str) -> None:
    golden = _load_golden()
    assert _compute_role(role_id) == golden[role_id], (
        f"role {role_id!r} default payload changed vs golden — if this is a "
        "deliberate, reviewed behavior change, regenerate the golden"
    )
