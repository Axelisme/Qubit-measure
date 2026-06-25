"""Tests for macro/meta.py: MetaMacro."""

from __future__ import annotations

from zcu_tools.program.v2.macro.meta import MetaMacro

from tests.program.v2.support import ProgramTrace


def _make_prog(get_reg_map=None):
    prog = ProgramTrace()
    if get_reg_map:
        prog.set_reg_map(get_reg_map)
    return prog


def test_meta_macro_translate_calls_add_meta(mock_prog):
    macro = MetaMacro(type="LOOP_START", name="my_loop")
    macro.translate(mock_prog)
    assert mock_prog.only("meta").kwargs == {
        "type": "LOOP_START",
        "name": "my_loop",
        "info": {},
    }


def test_meta_macro_translate_passes_info(mock_prog):
    info = {"counter_reg": "r0", "n": 10}
    macro = MetaMacro(type="LOOP_START", name="lp", info=info)
    macro.translate(mock_prog)
    kwargs = mock_prog.only("meta").kwargs
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
    macro.translate(prog)  # type: ignore[arg-type]
    kwargs = prog.only("meta").kwargs
    assert kwargs["info"]["counter_reg"] == "r5"
    assert kwargs["info"]["n"] == 10


def test_meta_macro_default_info_and_regs(mock_prog):
    macro = MetaMacro(type="BRANCH_END", name="br")
    assert macro.info == {}
    assert macro.regs == {}
    macro.translate(mock_prog)
    assert mock_prog.count("meta") == 1


def test_meta_macro_various_types(mock_prog):
    for t in (
        "LOOP_START",
        "LOOP_BODY_START",
        "LOOP_BODY_END",
        "LOOP_END",
        "BRANCH_START",
        "BRANCH_CASE_START",
        "BRANCH_CASE_END",
        "BRANCH_END",
    ):
        macro = MetaMacro(type=t, name="x")
        macro.translate(mock_prog)
    assert mock_prog.count("meta") == 8
