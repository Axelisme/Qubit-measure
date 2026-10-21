"""Tests for metadata emitted through MetaMacro.translate on the real compiler."""

from __future__ import annotations

import pytest
from qick.asm_v2 import AsmInst
from zcu_tools.program.v2.ir.base import IRCompileMixin
from zcu_tools.program.v2.macro.meta import MetaMacro
from zcu_tools.program.v2.mocksoc import make_mock_soccfg


@pytest.fixture
def prog() -> IRCompileMixin:
    return IRCompileMixin(make_mock_soccfg())


def test_meta_macro_translate_records_metadata(prog: IRCompileMixin):
    macro = MetaMacro(type="LOOP_START", name="my_loop")
    macro.translate(prog)
    assert prog.meta_infos == [
        {
            "kind": "meta",
            "type": "LOOP_START",
            "name": "my_loop",
            "info": {},
            "p_addr": 1,
        }
    ]


def test_meta_macro_translate_passes_info(prog: IRCompileMixin):
    info = {"counter_reg": "r0", "n": 10}
    macro = MetaMacro(type="LOOP_START", name="lp", info=info)
    macro.translate(prog)
    assert prog.meta_infos == [
        {
            "kind": "meta",
            "type": "LOOP_START",
            "name": "lp",
            "info": {"counter_reg": "r0", "n": 10},
            "p_addr": 1,
        }
    ]


def test_meta_macro_translate_resolves_regs(prog: IRCompileMixin):
    macro = MetaMacro(
        type="LOOP_BODY_START",
        name="lp",
        info={"n": 10},
        regs={"counter_reg": "my_reg"},
    )
    # Registration after macro construction still resolves at translate time.
    prog.add_reg(name="my_reg", addr=5)
    macro.translate(prog)
    assert prog.meta_infos == [
        {
            "kind": "meta",
            "type": "LOOP_BODY_START",
            "name": "lp",
            "info": {"counter_reg": "r5", "n": 10},
            "p_addr": 1,
        }
    ]


def test_meta_macro_default_info_and_regs(prog: IRCompileMixin):
    macro = MetaMacro(type="BRANCH_END", name="br")
    assert macro.info == {}
    assert macro.regs == {}
    macro.translate(prog)
    assert len(prog.meta_infos) == 1
    assert prog.meta_infos == [
        {
            "kind": "meta",
            "type": "BRANCH_END",
            "name": "br",
            "info": {},
            "p_addr": 1,
        }
    ]


def test_meta_macro_various_types(prog: IRCompileMixin):
    types = (
        "LOOP_START",
        "LOOP_BODY_START",
        "LOOP_BODY_END",
        "LOOP_END",
        "BRANCH_START",
        "BRANCH_CASE_START",
        "BRANCH_CASE_END",
        "BRANCH_END",
    )
    for meta_type in types:
        macro = MetaMacro(type=meta_type, name="x")
        macro.translate(prog)
    assert len(prog.meta_infos) == 8
    assert prog.meta_infos == [
        {"kind": "meta", "type": meta_type, "name": "x", "info": {}, "p_addr": 1}
        for meta_type in types
    ]


def test_meta_macro_copies_info_and_overwrites_register_keys(prog: IRCompileMixin):
    prog.add_reg(name="my_reg", addr=5)
    info = {"counter_reg": "old_value", "n": 10}
    regs = {"counter_reg": "my_reg"}
    macro = MetaMacro(type="LOOP_START", name="lp", info=info, regs=regs)
    macro.translate(prog)

    assert info == {"counter_reg": "old_value", "n": 10}
    assert regs == {"counter_reg": "my_reg"}
    info["n"] = 20
    assert prog.meta_infos == [
        {
            "kind": "meta",
            "type": "LOOP_START",
            "name": "lp",
            "info": {"counter_reg": "r5", "n": 10},
            "p_addr": 1,
        }
    ]


@pytest.mark.parametrize(("register", "address"), [("r7", "r7"), ("s_zero", "s0")])
def test_meta_macro_resolves_asm_addresses_and_aliases(
    prog: IRCompileMixin, register: str, address: str
):
    macro = MetaMacro(type="BRANCH_START", name="br", regs={"compare_reg": register})
    macro.translate(prog)
    assert prog.meta_infos == [
        {
            "kind": "meta",
            "type": "BRANCH_START",
            "name": "br",
            "info": {"compare_reg": address},
            "p_addr": 1,
        }
    ]


def test_meta_macro_unknown_register_preserves_metadata_and_inputs(
    prog: IRCompileMixin,
):
    MetaMacro(type="BRANCH_END", name="previous").translate(prog)
    prog.add_reg(name="my_reg", addr=5)
    info = {"n": 10}
    regs = {"counter_reg": "my_reg", "compare_reg": "missing_reg"}
    macro = MetaMacro(type="LOOP_START", name="lp", info=info, regs=regs)

    with pytest.raises(KeyError, match="^'missing_reg'$") as exc_info:
        macro.translate(prog)

    assert exc_info.value.args == ("missing_reg",)
    assert info == {"n": 10}
    assert regs == {"counter_reg": "my_reg", "compare_reg": "missing_reg"}
    assert prog.meta_infos == [
        {
            "kind": "meta",
            "type": "BRANCH_END",
            "name": "previous",
            "info": {},
            "p_addr": 1,
        }
    ]


def test_meta_macro_records_current_address_without_emitting_asm(prog: IRCompileMixin):
    macro = MetaMacro(type="BRANCH_END", name="br")
    initial_asm = list(prog.prog_list)
    macro.translate(prog)
    assert prog.p_addr == 1
    assert prog.prog_list == initial_asm

    AsmInst(inst={"CMD": "NOP"}, addr_inc=1).translate(prog)
    advanced_asm = list(prog.prog_list)
    macro.translate(prog)
    assert prog.p_addr == 2
    assert prog.prog_list == advanced_asm
    assert prog.meta_infos == [
        {"kind": "meta", "type": "BRANCH_END", "name": "br", "info": {}, "p_addr": 1},
        {"kind": "meta", "type": "BRANCH_END", "name": "br", "info": {}, "p_addr": 2},
    ]
