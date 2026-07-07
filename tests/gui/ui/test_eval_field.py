"""Unit tests for EvalNumericField (session/ui/eval_field.py).

Test patterns borrowed from tests/gui/ui/test_cfg_form.py (ScalarWidget ghost /
context-menu / eval↔direct switching). MetaDict is constructed inline following
the pattern in tests/gui/test_expression.py.
"""

from __future__ import annotations

import pytest
from zcu_tools.gui.session.expression import EvalRef
from zcu_tools.gui.session.ui.eval_field import EvalNumericField
from zcu_tools.meta_tool import MetaDict


def _field(md: MetaDict, decimals: int = 6) -> EvalNumericField:
    return EvalNumericField(
        minimum=-1e9,
        maximum=1e9,
        decimals=decimals,
        md_provider=lambda: md,
        type_=float,
    )


def _field_no_md() -> EvalNumericField:
    """Field whose md_provider always raises (simulates no active context)."""

    def provider() -> MetaDict:
        raise RuntimeError("no active context")

    return EvalNumericField(
        minimum=-1e9, maximum=1e9, decimals=6, md_provider=provider, type_=float
    )


# ---------------------------------------------------------------------------
# 1. direct mode default
# ---------------------------------------------------------------------------


def test_direct_default_read_raw_returns_float(qapp):
    md = MetaDict()
    f = _field(md)
    f.load_direct(1.5)
    result = f.read_raw()
    assert isinstance(result, float)
    assert result == pytest.approx(1.5)


def test_direct_default_mode_is_direct(qapp):
    md = MetaDict()
    f = _field(md)
    assert f._mode == "direct"


# ---------------------------------------------------------------------------
# 2. switch to eval → ghost resolves correctly
# ---------------------------------------------------------------------------


def test_eval_mode_read_raw_returns_eval_ref(qapp):
    md = MetaDict()
    md.flx_int = 0.5
    f = _field(md)
    f._switch_to_eval()
    f._line_edit.setText("flx_int")
    result = f.read_raw()
    assert isinstance(result, EvalRef)
    assert result.expr == "flx_int"
    assert result.type_ is float


def test_eval_mode_ghost_shows_resolved_value(qapp):
    md = MetaDict()
    md.flx_int = 0.5
    f = _field(md)
    f._switch_to_eval()
    f._line_edit.setText("flx_int")
    # Ghost is updated via textChanged → _sync_ghost
    assert f._ghost.text() == "= 0.5"
    assert "red" not in f._ghost.styleSheet()
    assert "gray" in f._ghost.styleSheet()


# ---------------------------------------------------------------------------
# 3. undefined name → ghost shows red '= ?'
# ---------------------------------------------------------------------------


def test_undefined_name_shows_red_ghost(qapp):
    md = MetaDict()
    f = _field(md)
    f._switch_to_eval()
    f._line_edit.setText("missing_var")
    assert f._ghost.text() == "= ?"
    assert "red" in f._ghost.styleSheet()
    assert f._ghost.toolTip() != ""


# ---------------------------------------------------------------------------
# 4. syntax error → ghost shows red '= ?'
# ---------------------------------------------------------------------------


def test_syntax_error_shows_red_ghost(qapp):
    md = MetaDict()
    f = _field(md)
    f._switch_to_eval()
    f._line_edit.setText("r_f[")
    assert f._ghost.text() == "= ?"
    assert "red" in f._ghost.styleSheet()


# ---------------------------------------------------------------------------
# 5. eval→direct carries resolved value back
# ---------------------------------------------------------------------------


def test_eval_to_direct_carries_resolved_value(qapp):
    md = MetaDict()
    md.flx_int = 0.5
    f = _field(md)
    f._switch_to_eval()
    f._line_edit.setText("flx_int")
    # Ghost has resolved successfully; switch back to direct
    f._switch_to_direct()
    assert f._mode == "direct"
    result = f.read_raw()
    assert isinstance(result, float)
    assert result == pytest.approx(0.5)


def test_eval_to_direct_falls_back_to_stored_value_on_unresolved(qapp):
    md = MetaDict()
    f = _field(md)
    f.load_direct(7.0)
    f._switch_to_eval()
    f._line_edit.setText("missing_var")  # cannot resolve
    f._switch_to_direct()
    assert f._mode == "direct"
    # Falls back to stored direct value
    assert f.read_raw() == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# 6. load_direct in eval mode must NOT clobber the expression (R3 regression)
# ---------------------------------------------------------------------------


def test_load_direct_in_eval_mode_preserves_expression(qapp):
    """Simulates a 1-second poll repaint: load_direct is called while the user
    has an expression in the eval field. The expression must be preserved."""
    md = MetaDict()
    md.flx_int = 0.5
    f = _field(md)
    f._switch_to_eval()
    f._line_edit.setText("flx_int")

    # Poll repaint fires with a new device value
    f.load_direct(9.0)

    # Still in eval mode, expression unchanged
    assert f._mode == "eval"
    assert f._line_edit.text() == "flx_int"
    # Backing direct value updated (used if user later switches back to direct)
    assert f._direct_value == pytest.approx(9.0)
    # read_raw still returns EvalRef
    result = f.read_raw()
    assert isinstance(result, EvalRef)
    assert result.expr == "flx_int"


# ---------------------------------------------------------------------------
# 7. reset_to_direct: unconditionally switches back to direct mode
# ---------------------------------------------------------------------------


def test_reset_to_direct_from_eval_mode(qapp):
    md = MetaDict()
    md.flx_int = 0.5
    f = _field(md)
    f.load_direct(3.0)
    f._switch_to_eval()
    f._line_edit.setText("flx_int")

    f.reset_to_direct()

    assert f._mode == "direct"
    # Spinbox shows the stored backing value (3.0, not the expression result)
    assert f.read_raw() == pytest.approx(3.0)
    # isHidden() checks the widget's own hidden flag; isVisible() also requires
    # all ancestors to be shown, so it is false in headless tests.
    assert not f._spin.isHidden()
    assert f._line_edit.isHidden()
    assert f._ghost.isHidden()


def test_load_expression_switches_to_eval_mode(qapp):
    md = MetaDict()
    md.q_f = 4567.0
    f = _field(md)

    f.load_expression("q_f", direct_fallback=1.25)

    assert f._mode == "eval"
    assert f._line_edit.text() == "q_f"
    assert not f._line_edit.isHidden()
    assert f._spin.isHidden()
    result = f.read_raw()
    assert isinstance(result, EvalRef)
    assert result.expr == "q_f"


# ---------------------------------------------------------------------------
# 8. no active context → ghost shows red
# ---------------------------------------------------------------------------


def test_no_context_md_provider_shows_red_ghost(qapp):
    """When md_provider raises, the ghost must show red '= ?'."""
    f = _field_no_md()
    f._switch_to_eval()
    f._line_edit.setText("flx_int")
    assert f._ghost.text() == "= ?"
    assert "red" in f._ghost.styleSheet()


# ---------------------------------------------------------------------------
# 9. ghost formatting: trailing-zero stripping
# ---------------------------------------------------------------------------


def test_ghost_strips_trailing_zeros(qapp):
    md = MetaDict()
    md.r_f = 6000.0
    f = _field(md, decimals=3)
    f._switch_to_eval()
    f._line_edit.setText("r_f")
    # 6000.0 with 3 decimals → "6000.000" → stripped to "6000.0"
    assert f._ghost.text() == "= 6000.0"


# ---------------------------------------------------------------------------
# 10. Context menu wiring: action triggers mode switch
# ---------------------------------------------------------------------------


def test_context_menu_action_direct_to_eval(qapp):
    """Right-click 'Use expression' action is wired to _switch_to_eval.

    Avoids modal exec_() by calling _build_context_menu directly and
    triggering the returned action, which fires the connected slot.
    """
    md = MetaDict()
    md.flx_int = 0.5
    f = _field(md)
    assert f._mode == "direct"

    # Build the menu from the spinbox's embedded QLineEdit
    spin_line_edit = f._spin.lineEdit()
    assert spin_line_edit is not None
    menu = f._build_context_menu(spin_line_edit)
    assert menu is not None

    # The last action added is the toggle action ("Use expression" in direct mode)
    actions = menu.actions()
    toggle_action = actions[-1]
    assert toggle_action.text() == "Use expression"

    # Trigger it — fires the connected _switch_to_eval slot without exec_()
    toggle_action.trigger()

    assert f._mode == "eval"


def _field_bounded(
    md: MetaDict, minimum: float, maximum: float, decimals: int = 3
) -> EvalNumericField:
    """Field with a specific [minimum, maximum] range for bounds-check tests."""
    return EvalNumericField(
        minimum=minimum,
        maximum=maximum,
        decimals=decimals,
        md_provider=lambda: md,
        type_=float,
    )


# ---------------------------------------------------------------------------
# 11. read_raw() in eval mode carries field min/max into EvalRef
# ---------------------------------------------------------------------------


def test_read_raw_eval_carries_min_max(qapp):
    """EvalRef returned by read_raw() must embed the field's minimum and maximum."""
    md = MetaDict()
    f = EvalNumericField(minimum=1e-9, maximum=1e9, decimals=9, md_provider=lambda: md)
    f._switch_to_eval()
    f._line_edit.setText("1.0")
    result = f.read_raw()
    assert isinstance(result, EvalRef)
    assert result.minimum == 1e-9
    assert result.maximum == 1e9


# ---------------------------------------------------------------------------
# 12. ghost range check: out-of-range → red; in-range → gray
# ---------------------------------------------------------------------------


def test_ghost_above_max_shows_red(qapp):
    """Resolved value exceeds maximum → ghost is red with 'out of range' tooltip."""
    md = MetaDict()
    md.x = 200.0  # above maximum=100
    f = _field_bounded(md, minimum=0.0, maximum=100.0)
    f._switch_to_eval()
    f._line_edit.setText("x")
    assert "red" in f._ghost.styleSheet()
    assert "out of range" in f._ghost.toolTip()


def test_ghost_below_min_shows_red(qapp):
    """Resolved value below minimum → ghost is red with 'out of range' tooltip."""
    md = MetaDict()
    md.x = 0.0  # below minimum=1e-9
    f = _field_bounded(md, minimum=1e-9, maximum=1e9)
    f._switch_to_eval()
    f._line_edit.setText("x")
    assert "red" in f._ghost.styleSheet()
    assert "out of range" in f._ghost.toolTip()


def test_ghost_in_range_shows_gray(qapp):
    """Resolved value within bounds → ghost is gray (normal success state)."""
    md = MetaDict()
    md.x = 50.0  # within [0, 100]
    f = _field_bounded(md, minimum=0.0, maximum=100.0)
    f._switch_to_eval()
    f._line_edit.setText("x")
    assert "gray" in f._ghost.styleSheet()
    assert "red" not in f._ghost.styleSheet()
    assert f._ghost.toolTip() == ""


def test_context_menu_action_eval_to_direct(qapp):
    """Right-click 'Use direct value' action is wired to _switch_to_direct."""
    md = MetaDict()
    md.flx_int = 0.5
    f = _field(md)
    f._switch_to_eval()
    f._line_edit.setText("flx_int")
    assert f._mode == "eval"

    # Build menu from the eval-mode QLineEdit
    menu = f._build_context_menu(f._line_edit)
    assert menu is not None

    actions = menu.actions()
    toggle_action = actions[-1]
    assert toggle_action.text() == "Use direct value"

    toggle_action.trigger()

    assert f._mode == "direct"
    # Resolved ghost value (0.5) should be carried back into the spinbox
    assert f.read_raw() == pytest.approx(0.5)
