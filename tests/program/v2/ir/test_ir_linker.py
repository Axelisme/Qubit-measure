from collections.abc import Callable

import pytest
from zcu_tools.program.v2.ir.instructions import (
    Instruction,
    LabelInst,
    MetaInst,
    NopInst,
    RegWriteInst,
    WaitInst,
)
from zcu_tools.program.v2.ir.labels import Label
from zcu_tools.program.v2.ir.linker import IRLinker
from zcu_tools.program.v2.ir.operands import Immediate, Register, SrcKeyword


def test_position_preserves_order_and_counts_wait_labels_and_metadata():
    insts: list[Instruction] = [
        MetaInst(type="LOOP_START", name="loop"),
        LabelInst(name=Label("entry")),
        WaitInst(c_op="time"),
        MetaInst(type="LOOP_END", name="loop"),
        LabelInst(name=Label("tail")),
        NopInst(),
        LabelInst(name=Label("end")),
        MetaInst(type="TIMING_BOUNDARY", name="body_end"),
    ]
    positioned, cursor = IRLinker().position(insts)

    assert [record.instruction for record in positioned] == insts
    assert all(
        record.instruction is inst
        for record, inst in zip(positioned, insts, strict=True)
    )
    assert [record.p_addr for record in positioned] == [0, 0, 0, 2, 2, 2, 3, 3]
    assert [record.line for record in positioned] == [0, 1, 2, 2, 3, 4, 5, 5]
    assert cursor.final_p_addr == 3
    assert cursor.final_line == 5


def test_linker_link_tracks_meta_and_label_without_advancing_address():
    linker = IRLinker()
    insts: list[Instruction] = [
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


@pytest.mark.parametrize("operation", [IRLinker.link, IRLinker.position])
def test_linker_rejects_duplicate_label_names(
    operation: Callable[[IRLinker, list[Instruction]], object],
):
    linker = IRLinker()
    insts: list[Instruction] = [
        LabelInst(name=Label("dup")),
        NopInst(),
        LabelInst(name=Label("dup")),
    ]

    with pytest.raises(ValueError, match="duplicate label 'dup'"):
        operation(linker, insts)


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
