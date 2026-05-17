"""Tests for macro/meta.py: MetaMacro."""

from __future__ import annotations

from unittest.mock import MagicMock

from zcu_tools.program.v2.macro.meta import MetaMacro


def _make_prog(get_reg_map=None):
    prog = MagicMock()
    if get_reg_map:
        prog._get_reg.side_effect = lambda name: get_reg_map.get(name, name)
    else:
        prog._get_reg.side_effect = lambda name: name
    return prog


def test_meta_macro_translate_calls_add_meta(mock_prog):
    macro = MetaMacro(type="LOOP_START", name="my_loop")
    macro.translate(mock_prog)
    mock_prog._add_meta.assert_called_once_with(
        type="LOOP_START", name="my_loop", info={}
    )


def test_meta_macro_translate_passes_info(mock_prog):
    info = {"counter_reg": "r0", "n": 10}
    macro = MetaMacro(type="LOOP_START", name="lp", info=info)
    macro.translate(mock_prog)
    _, kwargs = mock_prog._add_meta.call_args
    assert kwargs["info"]["counter_reg"] == "r0"
    assert kwargs["info"]["n"] == 10


def test_meta_macro_translate_resolves_regs():
    prog = _make_prog(get_reg_map={"my_reg": "r5"})
    macro = MetaMacro(
        type="LOOP_BODY_START",
        name="lp",
        info={"n": 10},
        regs={"counter_reg": "my_reg"},
    )
    macro.translate(prog)
    _, kwargs = prog._add_meta.call_args
    assert kwargs["info"]["counter_reg"] == "r5"
    assert kwargs["info"]["n"] == 10


def test_meta_macro_default_info_and_regs(mock_prog):
    macro = MetaMacro(type="BRANCH_END", name="br")
    assert macro.info == {}
    assert macro.regs == {}
    macro.translate(mock_prog)
    mock_prog._add_meta.assert_called_once()


def test_meta_macro_various_types(mock_prog):
    for t in ("LOOP_START", "LOOP_BODY_START", "LOOP_BODY_END", "LOOP_END",
              "BRANCH_START", "BRANCH_CASE_START", "BRANCH_CASE_END", "BRANCH_END"):
        macro = MetaMacro(type=t, name="x")
        macro.translate(mock_prog)
    assert mock_prog._add_meta.call_count == 8
