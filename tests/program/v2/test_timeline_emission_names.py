"""Emission naming scopes through public lowering, failures and compilation."""

from collections.abc import Callable, Sequence
from typing import override

import pytest
from zcu_tools.program.v2 import ir
from zcu_tools.program.v2.modules import Module
from zcu_tools.program.v2.timeline import TimelineProgramV2

from tests.program._profiles import TPROC_MHZ, load_soccfg


class LoweringModule(Module):
    def __init__(self, name: str, action: Callable[[TimelineProgramV2], None]):
        self.name = name
        self.action = action

    @override
    def declare(self, prog):
        pass

    @override
    def lower(self, prog, *, timeline, tag):
        self.action(prog)


def compile_modules(
    body: Callable[[TimelineProgramV2], None], modules: Sequence[Module]
):
    class Program(TimelineProgramV2):
        def _initialize(self, cfg):
            self.add_reg("output", addr=10)
            for module in modules:
                self.declare(module)

        def _body(self, cfg):
            body(self)

    return Program(load_soccfg(), initial_delay=0, relax_delay=0)


def port_value(prog: TimelineProgramV2, value: int):
    prog.write_reg("output", value)
    prog.asm_inst(
        {"CMD": "DPORT_WR", "DST": "0", "SRC": "reg", "DATA": "r10", "TIME": "@0"}
    )


def test_nested_failed_lower_restores_naming_scope_and_discards_partial_emission():
    parent_names: list[str] = []
    child_names: list[str] = []

    def failing_lower(prog):
        child_names.append(prog.emission_name("scratch"))
        port_value(prog, 999)
        raise RuntimeError("synthetic lower failure")

    failing = LoweringModule("failing", failing_lower)

    def parent_lower(prog):
        parent_names.append(prog.emission_name("scratch"))
        with pytest.raises(RuntimeError, match="synthetic lower failure"):
            prog.emit(failing)
        parent_names.append(prog.emission_name("scratch"))
        port_value(prog, 17)

    parent = LoweringModule("parent", parent_lower)

    def body(prog):
        prog.emit(parent)
        prog.emit(parent)
        with pytest.raises(ValueError, match="active module lower"):
            prog.emission_name("outside")

    prog = compile_modules(body, [parent, failing])
    result = ir.analyze_timing(
        ir.CompiledTimingInput(prog.prog_list, prog.labels, prog.timing_layout)
    )
    assert [(event.cycle, event.value) for event in result.shots[0].events] == [
        (0, 17),
        (0, 17),
    ]
    assert len(set(parent_names + child_names)) == 6
    first_parent_names, first_child_names = parent_names[:], child_names[:]
    prog.compile()
    assert parent_names == first_parent_names
    assert child_names == first_child_names
    compile_modules(body, [parent, failing])
    assert parent_names == first_parent_names * 2
    assert child_names == first_child_names * 2

    # A nested failure must not consume the caller's next local name. Compare
    # the first parent's sequence with the same lowering absent a nested emit.
    standalone_names: list[str] = []

    def standalone_lower(prog):
        standalone_names.extend(
            [prog.emission_name("scratch"), prog.emission_name("scratch")]
        )

    standalone = LoweringModule("standalone", standalone_lower)
    compile_modules(lambda p: p.emit(standalone), [standalone])
    assert first_parent_names[:2] == standalone_names
    with pytest.raises(ValueError, match="active module lower"):
        prog.emission_name("outside")


@pytest.mark.parametrize("native_loop", ["repeat", "open_loop"])
def test_generated_loop_name_collision_with_explicit_native_name_is_rejected(
    native_loop,
):
    names: list[str] = []

    def lower(prog):
        name = prog.emission_name("loop")
        names.append(name)
        prog.delay_by_timeline()
        with prog.repeat(name, 2):
            prog.timeline_inc(1 / TPROC_MHZ)
            prog.delay_by_timeline()

    module = LoweringModule("loop_owner", lower)

    def body(prog):
        prog.emit(module)
        if native_loop == "repeat":
            prog.delay_by_timeline()
            with prog.repeat(names[0], 1):
                prog.timeline_inc(1 / TPROC_MHZ)
                prog.delay_by_timeline()
        else:
            prog.open_loop(1, name=names[0])
            prog.close_loop()

    with pytest.raises(
        (ValueError, NameError), match="(loop name already used|already exists)"
    ):
        compile_modules(body, [module])
