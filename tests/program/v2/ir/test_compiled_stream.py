"""Public compilation publishes a detached, pre-optimization IR artifact."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from qick.asm_v2 import Pulse as NativePulse
from qick.asm_v2 import QickParam, QickProgramV2, QickSweep1D
from qick.qick_asm import AbsQickProgram
from zcu_tools.program.v2 import ir
from zcu_tools.program.v2.base import MyProgramV2
from zcu_tools.program.v2.ir.instructions import (
    BaseInst,
    LabelInst,
    MetaInst,
    RegWriteInst,
    WaitInst,
)
from zcu_tools.program.v2.ir.io_instructions import PortWriteInst
from zcu_tools.program.v2.mocksoc import make_mock_soccfg
from zcu_tools.program.v2.modular import ModularProgramV2
from zcu_tools.program.v2.modules import Branch, Repeat, SoftDelay
from zcu_tools.program.v2.modules.pulse import Pulse, PulseCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg
from zcu_tools.program.v2.timeline import TimelineProgramV2


def _program(path: str) -> tuple[MyProgramV2, Callable[[], None]]:
    if path == "legacy":
        repeat = Repeat("inner", 3, range_hint=(1, 4)).add_content(
            SoftDelay("pause", delay=0.1)
        )
        prog = ModularProgramV2(make_mock_soccfg(), modules=[repeat])

        def update() -> None:
            repeat.n = 4

        return prog, update

    class NativeProgram(TimelineProgramV2):
        loop_count = 3

        def _initialize(self, cfg: None) -> None:
            self.add_reg("counter")

        def _body(self, cfg: None) -> None:
            self.open_inner_loop("inner", "counter", self.loop_count, range_hint=(1, 4))
            self.delay(t=0.1)
            self.close_inner_loop("inner", "counter", self.loop_count)

    native = NativeProgram(make_mock_soccfg())

    def update_native() -> None:
        native.loop_count = 4

    return native, update_native


def _invalidate_binary(prog: AbsQickProgram) -> None:
    # The QICK base owns the nullable public compile cache.
    prog.binprog = None


def _markers(snapshot: ir.CompiledStream) -> list[MetaInst]:
    return [
        inst
        for record in snapshot.records
        if isinstance(inst := record.instruction, MetaInst)
    ]


@pytest.mark.parametrize("path", ["legacy", "native"])
def test_compile_snapshot_preserves_typed_loop_and_detaches_mutable_payload(path: str):
    prog, _ = _program(path)
    assert hasattr(prog, "compiled_stream"), "compile must publish compiled_stream"
    snapshot = prog.compiled_stream
    assert isinstance(snapshot, ir.CompiledStream)
    assert isinstance(snapshot.records, tuple)
    markers = _markers(snapshot)
    loop_start = next(
        inst for inst in markers if inst.type == "LOOP_START" and inst.name == "inner"
    )
    assert loop_start.name == "inner"
    assert loop_start.info["n"] == 3
    assert loop_start.info["range_hint"] == (1, 4)
    assert [
        inst.type
        for inst in markers
        if inst.name == "inner" and inst.type.startswith("LOOP_")
    ] == [
        "LOOP_START",
        "LOOP_BODY_START",
        "LOOP_BODY_END",
        "LOOP_END",
    ]
    record = next(
        record
        for record in snapshot.records
        if isinstance(record.instruction, RegWriteInst)
    )
    write = record.instruction
    assert isinstance(write, RegWriteInst)
    original_dst = write.dst.name
    write.dst.name = "r999"
    fresh_write = record.instruction
    assert isinstance(fresh_write, RegWriteInst)
    assert fresh_write.dst.name == original_dst

    loop_start.info["n"] = 999
    assert (
        next(
            inst
            for inst in _markers(snapshot)
            if inst.type == "LOOP_START" and inst.name == "inner"
        ).info["n"]
        == 3
    )
    with pytest.raises(FrozenInstanceError):
        record.__setattr__("p_addr", 999)
    with pytest.raises(FrozenInstanceError):
        snapshot.__setattr__("records", ())
    assert any(isinstance(record.instruction, BaseInst) for record in snapshot.records)


@pytest.mark.parametrize("path", ["legacy", "native"])
def test_recompile_replaces_records_without_mutating_saved_snapshot_or_outputs(
    path: str,
):
    prog, update = _program(path)
    previous = prog.compiled_stream
    original_program = deepcopy(prog.prog_list)
    original_binary = deepcopy(prog.binprog)
    original_records = [record.instruction for record in previous.records]

    # Reading and editing a detached instruction does not edit public outputs.
    write = next(inst for inst in original_records if isinstance(inst, RegWriteInst))
    write.dst.name = "r999"
    assert prog.prog_list == original_program
    np.testing.assert_equal(prog.binprog, original_binary)
    prog.compile()
    assert prog.compiled_stream is previous  # QICK cached compile is a no-op.
    _invalidate_binary(prog)
    prog.compile()
    assert prog.compiled_stream is not previous
    assert prog.prog_list == original_program
    np.testing.assert_equal(prog.binprog, original_binary)

    update()
    _invalidate_binary(prog)
    prog.compile()
    current = prog.compiled_stream
    assert current is not previous
    old_loop = next(
        inst
        for inst in _markers(previous)
        if inst.type == "LOOP_START" and inst.name == "inner"
    )
    new_loop = next(
        inst
        for inst in _markers(current)
        if inst.type == "LOOP_START" and inst.name == "inner"
    )
    assert old_loop.info["n"] == 3
    assert new_loop.info["n"] == 4
    assert len(_markers(current)) == len(_markers(previous))
    # The optimizer has consumed structure, but the saved logical records retain it.
    assert all(meta.get("type") != "LOOP_START" for meta in prog.meta_infos)


@pytest.mark.parametrize("path", ["legacy", "native"])
def test_invalidated_binary_cannot_expose_a_stale_snapshot(path: str):
    prog, _ = _program(path)
    saved = prog.compiled_stream
    _invalidate_binary(prog)
    with pytest.raises(RuntimeError, match="no completed compilation"):
        _ = prog.compiled_stream
    assert (
        next(
            inst
            for inst in _markers(saved)
            if inst.type == "LOOP_START" and inst.name == "inner"
        ).info["n"]
        == 3
    )
    prog.compile()
    assert prog.compiled_stream is not saved


def test_snapshot_retains_branch_case_order_and_compare_register_before_optimization():
    branch = Branch(
        "select", SoftDelay("first", delay=0.1), SoftDelay("second", delay=0.2)
    )
    prog = ModularProgramV2(make_mock_soccfg(), modules=[branch], sweep=[("select", 2)])
    markers = _markers(prog.compiled_stream)
    branch_start = next(inst for inst in markers if inst.type == "BRANCH_START")
    assert branch_start.name == "select"
    assert branch_start.info["compare_reg"].startswith("r")
    assert [
        (inst.type, inst.name) for inst in markers if inst.type.startswith("BRANCH_")
    ] == [
        ("BRANCH_START", "select"),
        ("BRANCH_CASE_START", "0"),
        ("BRANCH_CASE_END", "0"),
        ("BRANCH_CASE_START", "1"),
        ("BRANCH_CASE_END", "1"),
        ("BRANCH_END", "select"),
    ]
    assert all(meta.get("type") != "BRANCH_START" for meta in prog.meta_infos)


def test_native_loop_markers_preserve_qick_sweep_updates_and_inner_structure():
    class UnmarkedLoops(ModularProgramV2):
        # Trusted QICK producers retain the original hardware sweep shell.
        def open_loop(self, n, name=None):
            QickProgramV2.open_loop(self, n, name=name)

        def close_loop(self):
            QickProgramV2.close_loop(self)

    def build(program_type: type[ModularProgramV2]) -> ModularProgramV2:
        pulse = Pulse(
            "swept",
            PulseCfg(
                ch=0,
                nqz=1,
                freq=1000.0,
                gain=QickSweep1D("scan", 0.1, 0.5),
                pre_delay=QickSweep1D("scan", 0.0, 0.2),
                waveform=ConstWaveformCfg(length=0.1),
            ),
        )
        inner = Repeat("inner", 2).add_content(pulse)
        return program_type(
            make_mock_soccfg(), modules=[inner], sweep=[("scan", 3)], reps=2
        )

    prog = build(ModularProgramV2)
    starts = [
        meta for meta in _markers(prog.compiled_stream) if meta.type == "LOOP_START"
    ]
    assert [(meta.name, meta.info["n"]) for meta in starts] == [
        ("reps", 2),
        ("scan", 3),
        ("inner", 2),
    ]
    for meta in starts[:2]:
        assert meta.info["native"] is True
        assert meta.info["counter_reg"].startswith("r")
        assert meta.info["range_hint"] is None
    assert [
        (meta.type, meta.name)
        for meta in _markers(prog.compiled_stream)
        if meta.type.startswith("LOOP_")
    ] == [
        ("LOOP_START", "reps"),
        ("LOOP_BODY_START", "reps"),
        ("LOOP_START", "scan"),
        ("LOOP_BODY_START", "scan"),
        ("LOOP_START", "inner"),
        ("LOOP_BODY_START", "inner"),
        ("LOOP_BODY_END", "inner"),
        ("LOOP_END", "inner"),
        ("LOOP_BODY_END", "scan"),
        ("LOOP_END", "scan"),
        ("LOOP_BODY_END", "reps"),
        ("LOOP_END", "reps"),
    ]
    original = build(UnmarkedLoops)
    assert prog.prog_list == original.prog_list
    np.testing.assert_equal(prog.binprog, original.binprog)


def test_timeline_native_loop_annotations_keep_waveform_and_register_sweeps():
    class NativeSweep(TimelineProgramV2):
        def _initialize(self, cfg: None) -> None:
            self.add_loop("scan", 3)
            self.add_reg("counter")
            self.declare_gen(0, nqz=1)
            self.add_pulse(
                0,
                "shared",
                style="const",
                freq=1000.0,
                phase=0,
                gain=QickSweep1D("scan", 0.1, 0.5),
                length=0.1,
            )

        def _body(self, cfg: None) -> None:
            self.open_inner_loop("inner", "counter", 2)
            self.pulse(0, "shared", t=QickSweep1D("scan", 0.0, 0.2))
            self.delay_auto()
            self.close_inner_loop("inner", "counter", 2)

    class UnmarkedNativeSweep(NativeSweep):
        def open_loop(self, n, name=None):
            QickProgramV2.open_loop(self, n, name=name)

        def close_loop(self):
            QickProgramV2.close_loop(self)

    prog = NativeSweep(make_mock_soccfg(), reps=2)
    assert [
        (meta.name, meta.info["n"])
        for meta in _markers(prog.compiled_stream)
        if meta.type == "LOOP_START"
    ] == [("reps", 2), ("scan", 3), ("inner", 2)]
    original = UnmarkedNativeSweep(make_mock_soccfg(), reps=2)
    assert prog.prog_list == original.prog_list
    np.testing.assert_equal(prog.binprog, original.binprog)


def test_legacy_shared_pulse_sources_follow_nested_module_emissions():
    cfg = PulseCfg(
        ch=0,
        nqz=1,
        freq=1000.0,
        gain=0.5,
        waveform=ConstWaveformCfg(length=0.1),
    )
    first, second = Pulse("first", cfg), Pulse("second", cfg)

    class ParentPulse(Pulse):
        def init(self, prog: ModularProgramV2 | TimelineProgramV2) -> None:
            super().init(prog)
            second.init(prog)

        def run(
            self, prog: ModularProgramV2, t: float | QickParam = 0.0
        ) -> float | QickParam:
            return super().run(prog, second.emit(prog, t))

    nested = Repeat("inner", 2).add_content(first)
    parent = ParentPulse("parent", cfg)
    branch = Branch("choice", nested, parent)
    prog = ModularProgramV2(make_mock_soccfg(), modules=[branch], sweep=[("choice", 2)])
    assert prog.pulse_registry.count == 1
    assert first.pulse_id == second.pulse_id
    scopes: list[str] = []
    outputs: list[tuple[str, ...]] = []
    for record in prog.compiled_stream.records:
        inst = record.instruction
        if isinstance(inst, MetaInst) and inst.type == "SOURCE_START":
            scopes.append(inst.name)
        elif isinstance(inst, MetaInst) and inst.type == "SOURCE_END":
            assert scopes.pop() == inst.name
        elif isinstance(inst, PortWriteInst):
            outputs.append(tuple(scopes))
    assert outputs == [
        ("choice", "inner", "first"),
        ("choice", "parent", "second"),
        ("choice", "parent"),
    ]
    assert not scopes
    assert not any(
        name.startswith("native:pulse:") for names in outputs for name in names
    )


def test_legacy_body_boundaries_exclude_qick_final_wait_and_delay():
    class UserWait(SoftDelay):
        def run(
            self, prog: ModularProgramV2, t: float | QickParam = 0.0
        ) -> float | QickParam:
            prog.wait(t=t + self.delay)
            return t

    prog = ModularProgramV2(
        make_mock_soccfg(),
        modules=[UserWait("user", delay=0.2)],
        initial_delay=0.1,
        relax_delay=0.3,
    )
    sections: list[str] = []
    user_waits = []
    shell_waits = []
    for record in prog.compiled_stream.records:
        inst = record.instruction
        if isinstance(inst, MetaInst) and inst.type == "TIMING_BOUNDARY":
            sections.append(inst.name)
        elif isinstance(inst, WaitInst):
            (user_waits if sections == ["body_start"] else shell_waits).append(inst)
    assert sections == ["body_start", "body_end"]
    assert len(user_waits) == 1
    assert len(shell_waits) == 1


def test_native_pulse_sources_use_call_ordinals_and_rebuild_stably():
    class NativePulses(MyProgramV2):
        def _initialize(self, cfg: None) -> None:
            self.declare_gen(0, nqz=1)
            self.add_pulse(
                0, "shared", style="const", freq=1000.0, phase=0, gain=0.5, length=0.1
            )

        def _body(self, cfg: None) -> None:
            self.pulse(0, "shared", t=0.0)
            with self.source_scope("module"):
                self.pulse(0, "shared", t=0.1)
            self.pulse(0, "shared", t=0.2, tag="tagged")
            self.pulse(0, "shared", t=0.3)

    prog = NativePulses(make_mock_soccfg())
    saved = prog.compiled_stream
    assert [
        (meta.type, meta.name)
        for meta in _markers(saved)
        if meta.type.startswith("SOURCE_")
    ] == [
        ("SOURCE_START", "native:pulse:1"),
        ("SOURCE_END", "native:pulse:1"),
        ("SOURCE_START", "module"),
        ("SOURCE_END", "module"),
        ("SOURCE_START", "tagged"),
        ("SOURCE_END", "tagged"),
        ("SOURCE_START", "native:pulse:3"),
        ("SOURCE_END", "native:pulse:3"),
    ]

    class UnmarkedPulses(NativePulses):
        def pulse(self, ch, name, t=0, tag=None):
            self.append_macro(NativePulse(ch=ch, name=name, t=t, tag=tag))

    # Source annotations preserve the native pulse machine instructions.
    np.testing.assert_equal(prog.binprog, UnmarkedPulses(make_mock_soccfg()).binprog)
    original = deepcopy(prog.binprog)
    _invalidate_binary(prog)
    prog.compile()
    assert _markers(prog.compiled_stream) == _markers(saved)
    np.testing.assert_equal(prog.binprog, original)


def test_snapshot_positions_keep_labels_boundaries_and_two_word_wait():
    class Program(TimelineProgramV2):
        def _body(self, cfg: None) -> None:
            self.label("before_wait")
            self.wait(t=0.2)
            self.label("after_wait")
            self.nop()

    prog = Program(make_mock_soccfg(), initial_delay=0, relax_delay=0)
    records = prog.compiled_stream.records
    labels = {
        inst.name.name: record.p_addr
        for record in records
        if isinstance(inst := record.instruction, LabelInst)
    }
    boundaries = {
        inst.name: record.p_addr
        for record in records
        if isinstance(inst := record.instruction, MetaInst)
        and inst.type == "TIMING_BOUNDARY"
    }
    user_wait = next(
        record
        for record in records
        if isinstance(record.instruction, WaitInst)
        and boundaries["body_start"] <= record.p_addr < boundaries["body_end"]
    )
    assert labels["before_wait"] == boundaries["body_start"]
    # Big-PMEM WAIT has a label-address setup word before its two-word directive.
    assert user_wait.p_addr == labels["before_wait"] + 1
    assert labels["after_wait"] == user_wait.p_addr + 2
    # Native body closure adds a TIME inc_ref after the explicit NOP.
    assert boundaries["body_end"] == labels["after_wait"] + 2
    assert list(boundaries) == ["body_start", "body_end"]
    assert [record.p_addr for record in records] == sorted(
        record.p_addr for record in records
    )
