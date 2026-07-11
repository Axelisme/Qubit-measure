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
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from zcu_tools.experiment.v2_gui.adapters._support.defaults.role_factories import (
    ROLE_FACTORIES,
)
from zcu_tools.gui.cfg import (
    CfgSectionValue,
    DirectValue,
    EvalValue,
    ReferenceValue,
    SweepValue,
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
        direct = cast(DirectValue, node)
        return {"D": direct.value}
    if isinstance(node, EvalValue):
        eval_value = cast(EvalValue, node)
        return {"E": eval_value.expr, "r": eval_value.resolved}
    if isinstance(node, SweepValue):
        sweep = cast(SweepValue, node)
        return {"sweep": [sweep.start, sweep.stop, sweep.expts, round(sweep.step, 9)]}
    if isinstance(node, ReferenceValue):
        ref = cast(ReferenceValue, node)
        return {
            "ref": ref.chosen_key,
            "ov": ref.is_overridden,
            "v": _serialize(ref.value),
        }
    if isinstance(node, CfgSectionValue):
        section = cast(CfgSectionValue, node)
        return {k: _serialize(v) for k, v in section.fields.items()}
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
