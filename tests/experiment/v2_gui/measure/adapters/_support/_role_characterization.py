"""Shared role characterization context, serializer and golden location."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from zcu_tools.gui.cfg import (
    CfgSectionValue,
    DirectValue,
    EvalValue,
    ReferenceValue,
    SweepValue,
)

GOLDEN_PATH = Path(__file__).with_name("_role_default_golden.json")

# A representative populated MetaDict: every md key the role factories read, so
# present keys lower to EvalValue and absent keys to their DirectValue fallback.
POPULATED_MD = {
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


def make_context(md: dict[str, Any]) -> MagicMock:
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


def serialize(node: object) -> Any:
    """Normalize a value-tree node to a JSON-comparable structure that keeps the
    DirectValue/EvalValue distinction and ref/section nesting."""
    if isinstance(node, DirectValue):
        return {"D": node.value}
    if isinstance(node, EvalValue):
        return {"E": node.expr, "r": node.resolved}
    if isinstance(node, SweepValue):
        return {"sweep": [node.start, node.stop, node.expts, round(node.step, 9)]}
    if isinstance(node, ReferenceValue):
        return {
            "ref": node.chosen_key,
            "ov": node.is_overridden,
            "v": serialize(node.value),
        }
    if isinstance(node, CfgSectionValue):
        return {k: serialize(v) for k, v in node.fields.items()}
    return None if node is None else repr(node)


def load_golden() -> dict[str, Any]:
    return json.loads(GOLDEN_PATH.read_text())
