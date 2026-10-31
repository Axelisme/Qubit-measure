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
from zcu_tools.program.v2.ir.io_instructions import DmemReadInst, PortWriteInst
from zcu_tools.program.v2.mocksoc import make_mock_soccfg
from zcu_tools.program.v2.modular import ModularProgramV2
from zcu_tools.program.v2.modules import (
    Branch,
    Join,
    LoadWord,
    Repeat,
    ScanWith,
    SoftDelay,
)
from zcu_tools.program.v2.modules.base import Module
from zcu_tools.program.v2.modules.pulse import Pulse, PulseCfg, TableLengthPulse
from zcu_tools.program.v2.modules.readout import (
    PulseReadout,
    PulseReadoutCfg,
    TablePulseReadout,
)
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg
from zcu_tools.program.v2.timeline import TimelineProgramV2
from zcu_tools.program.v2.utils import readout_freq_words


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


def _assert_table_snapshot(
    prog: ModularProgramV2 | TimelineProgramV2, gen: list[int], adc: list[int]
) -> None:
    loop = next(
        meta
        for meta in _markers(prog.compiled_stream)
        if meta.type == "LOOP_START" and meta.name == "scan"
    )
    tables = loop.info["tables"]
    dmem = prog.compile_datamem()
    if dmem is None:
        pytest.fail("table hooks require compiled DMEM")
    assert len(tables) == 2
    pairs = []
    for address, fields in tables.items():
        assert set(fields) == {0}
        offset = fields[0]
        pairs.append(
            (
                int(prog.waves[address].freq) % 2**32,
                [int(word) % 2**32 for word in dmem[offset : offset + 2]],
            )
        )
    assert sorted(pairs) == sorted(
        [(gen[0], [gen[1], gen[0]]), (adc[0], [adc[1], adc[0]])]
    )


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
    assert loop_start.info["sweeps"] == {"registers": {}, "waves": {}}
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
    sweep = starts[1].info["sweeps"]
    # The pulse start and DelayAuto end both sweep, in tProc cycles.
    assert list(sweep["registers"].values()) == [
        {"start": 0, "step": 43},
        {"start": 44, "step": 43},
    ]
    assert list(sweep["waves"].values()) == [{3: {"start": 3276, "step": 6553}}]
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


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize(
    "form", ["fixed", "runtime", "table", "repeat", "branch", "join"]
)
def test_readout_snapshot_labels_semantic_owner_and_author_children(native, form):
    soccfg = make_mock_soccfg()
    cfg = PulseReadoutCfg(
        gen_ch=0,
        ro_ch=0,
        nqz=1,
        freq=100,
        gain=0.1,
        waveform=ConstWaveformCfg(length=0.1),
        ro_length=0.1,
    )
    readout = cfg.build("readout")
    modules: list[Module] = [readout]
    gen, adc = readout_freq_words(
        soccfg, [100, 200], gen_ch=0, ro_ch=0, mixer_freq=None, nqz=1
    )
    if form == "runtime":
        modules = [
            LoadWord("frequency", gen, "scan", "freq_word"),
            LoadWord("adc", adc, "scan", "adc_word"),
            LoadWord("gain", [3276, 6552], "scan", "gain_word"),
            PulseReadout(
                "readout",
                cfg,
                freq_val="freq_word",
                ro_freq_val="adc_word",
                gain_val="gain_word",
            ),
        ]
    elif form == "table":
        modules = [
            TablePulseReadout(
                "readout", cfg, idx_reg="scan", freq_words=gen, ro_freq_words=adc
            )
        ]
    elif form == "repeat" and not native:
        modules = [Repeat("control", 2).add_content(readout)]
    elif form == "branch":
        modules = [
            Branch("control", readout, SoftDelay("pause", 0.2), compare_by="scan")
        ]
    elif form == "join":
        child = Pulse(
            "author_child",
            PulseCfg(
                ch=1, nqz=1, freq=80, gain=0.2, waveform=ConstWaveformCfg(length=0.2)
            ),
        )
        modules = [Join(readout, child)]

    class Program(TimelineProgramV2):
        def _initialize(self, cfg):
            self.add_loop("scan", 2)
            for module in modules:
                self.declare(module)

        def _body(self, cfg):
            if form == "repeat":
                self.sync()
                with self.repeat('control', 2):
                    self.emit(readout)
                    self.sync()
            else:
                for module in modules:
                    self.emit(module)

    prog = (
        Program(soccfg)
        if native
        else ModularProgramV2(soccfg, modules=modules, sweep=[("scan", 2)])
    )
    if form == "table":
        _assert_table_snapshot(prog, gen, adc)
        assert [
            meta.name
            for meta in _markers(prog.compiled_stream)
            if meta.type == "SOURCE_START"
        ] == ["readout"]
    assert _port_sources(prog.compiled_stream) == (
        ["readout", "readout", "author_child"]
        if form == "join"
        else ["readout", "readout"]
    )


def test_legacy_scan_loop_and_loader_keep_scan_owner_and_child_restores_it():
    scan = ScanWith("scan_owner", [2, 0, 1], "selected").add_content(
        SoftDelay("authored", 0.01)
    )
    parent = Repeat("outer", 2).add_content([scan, SoftDelay("after", 0.02)])
    prog = ModularProgramV2(make_mock_soccfg(), modules=[parent])
    scopes: list[str] = []
    starts: list[str] = []
    scan_owners: list[tuple[str, ...]] = []
    loader_owners: list[tuple[str, ...]] = []
    for record in prog.compiled_stream.records:
        inst = record.instruction
        if isinstance(inst, MetaInst) and inst.type == "SOURCE_START":
            scopes.append(inst.name)
            starts.append(inst.name)
        elif isinstance(inst, MetaInst) and inst.type == "SOURCE_END":
            assert scopes.pop() == inst.name
        elif isinstance(inst, MetaInst) and inst.name == "scan_owner_count":
            scan_owners.append(tuple(scopes))
        elif isinstance(inst, DmemReadInst):
            loader_owners.append(tuple(scopes))
    assert starts == ["outer", "scan_owner", "authored", "after"]
    assert scan_owners
    assert all(owner == ("outer", "scan_owner") for owner in scan_owners)
    assert loader_owners == [("outer", "scan_owner")]
    assert not scopes


def test_repeat_rejects_internal_module_that_is_not_its_child():
    child = SoftDelay("same_name", 0.01)
    stranger = SoftDelay("same_name", 0.01)

    class InvalidOwner(Repeat):
        def run(
            self,
            prog: ModularProgramV2,
            t: float | QickParam = 0.0,
            *,
            internal_modules: tuple[Module, ...] = (),
        ) -> float | QickParam:
            return super().run(prog, t, internal_modules=(stranger,))

    owner = InvalidOwner("owner", 2).add_content(child)
    with pytest.raises(ValueError, match="internal_modules.*children"):
        ModularProgramV2(make_mock_soccfg(), modules=[owner])


@pytest.mark.parametrize("joined", [False, True])
def test_legacy_table_scopes_keep_authored_children_and_anonymous_scheduling(joined):
    def table(name, ch):
        cfg = PulseCfg(
            ch=ch,
            nqz=1,
            freq=80,
            gain=0.2,
            waveform=ConstWaveformCfg(length=0.1),
        )
        return TableLengthPulse(name, cfg, lengths=[0.1, 0.2], idx_reg="scan")

    left = table("left", 0)
    content = Join(left, table("right", 1)) if joined else left
    parent = Repeat("outer", 2).add_content([content, SoftDelay("after", 0.01)])
    prog = ModularProgramV2(make_mock_soccfg(), modules=[parent], sweep=[("scan", 2)])
    starts = [
        meta.name
        for meta in _markers(prog.compiled_stream)
        if meta.type == "SOURCE_START"
    ]
    assert starts == (
        ["outer", "", "left", "right", "after"]
        if joined
        else ["outer", "left", "after"]
    )
    assert _port_sources(prog.compiled_stream) == (
        ["left", "right"] if joined else ["left"]
    )
    scopes: list[str] = []
    schedule: list[tuple[str, ...]] = []
    for record in prog.compiled_stream.records:
        inst = record.instruction
        if isinstance(inst, MetaInst) and inst.type == "SOURCE_START":
            scopes.append(inst.name)
        elif isinstance(inst, MetaInst) and inst.type == "SOURCE_END":
            assert scopes.pop() == inst.name
        elif isinstance(inst, RegWriteInst) and scopes and scopes[-1] == "":
            schedule.append(tuple(scopes))
    if joined:
        # Join's duration loader and common wait restore the anonymous owner.
        assert schedule
        assert set(schedule) == {("outer", "")}
    assert not scopes


def _port_sources(snapshot: ir.CompiledStream) -> list[str]:
    scopes: list[str] = []
    sources: list[str] = []
    for record in snapshot.records:
        inst = record.instruction
        if isinstance(inst, MetaInst) and inst.type == "SOURCE_START":
            scopes.append(inst.name)
        elif isinstance(inst, MetaInst) and inst.type == "SOURCE_END":
            assert scopes.pop() == inst.name
        elif isinstance(inst, PortWriteInst):
            sources.append(scopes[-1])
    assert not scopes
    return sources


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
