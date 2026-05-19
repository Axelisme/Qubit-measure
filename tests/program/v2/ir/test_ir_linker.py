import pytest
from zcu_tools.program.v2.ir.instructions import (
    LabelInst,
    MetaInst,
    NopInst,
    RegWriteInst,
)
from zcu_tools.program.v2.ir.labels import Label
from zcu_tools.program.v2.ir.linker import IRLinker
from zcu_tools.program.v2.ir.operands import Immediate, Register, SrcKeyword


def test_linker_link_tracks_meta_and_label_without_advancing_address():
    linker = IRLinker()
    insts = [
        MetaInst(type="LOOP_START", name="loop", info={"n": 2}),
        LabelInst(name=Label("entry"), can_remove=True),
        RegWriteInst(dst=Register("r0"), src=SrcKeyword.IMM, lit=Immediate(1)),
        NopInst(),
    ]

    prog_list, labels, meta_infos, cursor = linker.link(insts)

    assert prog_list == [
        {
            "CMD": "REG_WR",
            "DST": "r0",
            "SRC": "imm",
            "LIT": "#1",
            "P_ADDR": 0,
            "LINE": 2,
        },
        {"CMD": "NOP", "P_ADDR": 1, "LINE": 3},
    ]
    assert labels == {"entry": "&0"}
    assert meta_infos == [
        {
            "kind": "meta",
            "type": "LOOP_START",
            "name": "loop",
            "info": {"n": 2},
            "p_addr": 0,
        },
        {"kind": "label", "name": "entry", "can_remove": True, "p_addr": 0},
    ]
    assert cursor.final_p_addr == 2
    assert cursor.final_line == 3


@pytest.mark.parametrize(
    ("labels", "meta_infos", "expected"),
    [
        (
            {},
            [{"kind": "label", "name": "missing", "p_addr": 0}],
            r"Missing in labels dict: \['missing'\]",
        ),
        (
            {"extra": "&0"},
            [],
            r"Not found in meta_infos: \['extra'\]",
        ),
    ],
)
def test_unlink_rejects_label_mismatch(labels, meta_infos, expected):
    linker = IRLinker()

    with pytest.raises(ValueError, match=expected):
        linker.unlink([], labels, meta_infos)


def test_unlink_reconstructs_trailing_markers_in_original_order():
    linker = IRLinker()
    prog_list = [{"CMD": "NOP", "P_ADDR": 0}]
    labels = {"tail": "&1"}
    meta_infos = [
        {"kind": "meta", "type": "LOOP_END", "name": "loop", "info": {}, "p_addr": 1},
        {"kind": "label", "name": "tail", "p_addr": 1},
        {"kind": "meta", "type": "BRANCH_END", "name": "loop", "info": {}, "p_addr": 1},
    ]

    logical = linker.unlink(prog_list, labels, meta_infos)

    assert [type(inst).__name__ for inst in logical] == [
        "NopInst",
        "MetaInst",
        "LabelInst",
        "MetaInst",
    ]
    assert logical[1].to_dict()["type"] == "LOOP_END"
    assert logical[2].to_dict()["name"] == "tail"
    assert logical[3].to_dict()["type"] == "BRANCH_END"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (3, 3),
        ("&7", 7),
        ("11", 11),
    ],
)
def test_parse_label_addr_accepts_int_and_string_forms(value, expected):
    assert IRLinker._parse_label_addr("target", value) == expected


def test_parse_label_addr_rejects_invalid_values():
    with pytest.raises(ValueError, match="Invalid label address for 'bad'"):
        IRLinker._parse_label_addr("bad", "&oops")

    with pytest.raises(ValueError, match="Invalid label address type for 'bad'"):
        IRLinker._parse_label_addr("bad", 1.5)
