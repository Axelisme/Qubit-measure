"""Tests for the transitions form parsing/preset logic (headless)."""

from __future__ import annotations

import pytest
from zcu_tools.fluxdep_gui.ui.transitions_form import (
    PRESETS,
    TransitionsForm,
    format_pairs,
    parse_pairs,
)
from zcu_tools.notebook.persistance import TransitionDict

# --- pure parse/format -----------------------------------------------------


def test_parse_pairs_basic():
    assert parse_pairs("(0,1),(0,2)") == [(0, 1), (0, 2)]


def test_parse_pairs_with_spaces():
    assert parse_pairs(" (0, 1) , (1, 3) ") == [(0, 1), (1, 3)]


def test_parse_pairs_empty():
    assert parse_pairs("") == []
    assert parse_pairs("   ") == []


def test_parse_pairs_malformed_raises():
    with pytest.raises(ValueError):
        parse_pairs("(0,1,2)")
    with pytest.raises(ValueError):
        parse_pairs("(a,b)")


def test_format_pairs_roundtrip():
    pairs = [(0, 1), (1, 3), (2, 4)]
    assert parse_pairs(format_pairs(pairs)) == pairs


# --- widget ----------------------------------------------------------------


def test_form_default_preset_basic(qapp):
    form = TransitionsForm()
    t = form.get_transitions()
    assert t["transitions"] == PRESETS["basic"]["transitions"]
    assert t["mirror"] == PRESETS["basic"]["mirror"]
    form.deleteLater()


def test_form_get_transitions_drops_empty(qapp):
    form = TransitionsForm()
    # set only one category, clear the rest
    form.set_transitions(TransitionDict({"transitions": [(0, 1)]}))
    t = form.get_transitions()
    assert t == {"transitions": [(0, 1)]}
    form.deleteLater()


def test_form_set_then_get_roundtrip(qapp):
    form = TransitionsForm()
    src = TransitionDict({"transitions": [(0, 1), (1, 2)], "mirror": [(0, 2)]})
    form.set_transitions(src)
    out = form.get_transitions()
    assert out["transitions"] == [(0, 1), (1, 2)]
    assert out["mirror"] == [(0, 2)]
    form.deleteLater()
