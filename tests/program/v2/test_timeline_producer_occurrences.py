"""Time-input proofs belong to each lookup's structured execution position."""

import pytest
from qick.asm_v2 import Macro, WriteReg
from zcu_tools.program.v2 import ir
from zcu_tools.program.v2.modules import LoadValue
from zcu_tools.program.v2.timeline import TimelineProgramV2

from tests.program._profiles import load_soccfg


class UnregisteredIndexWrite(Macro):
    def translate(self, prog):
        WriteReg(dst="left_index", src=0).translate(prog)


def branch_program(
    *,
    uninitialized: int | None = None,
    late_initialization: bool = False,
    unregistered: str | None = None,
    time_consumer: bool = True,
):
    loaders = [
        LoadValue("left", [4, 9], "left_index", "duration", use_existed=True),
        LoadValue("right", [6, 11], "right_index", "duration", use_existed=True),
    ]

    class Program(TimelineProgramV2):
        def _initialize(self, cfg):
            self.add_loop("axis", 2)
            self.add_reg("left_index")
            self.add_reg("right_index")
            self.add_reg("duration")
            for loader in loaders:
                self.declare(loader)

        def _body(self, cfg):
            if unregistered == "before":
                self.append_macro(UnregisteredIndexWrite())
            with self.branch(by="axis", align_timeline=True) as branch:
                for index, loader in enumerate(loaders):
                    with branch.case(index):
                        if index != uninitialized:
                            self.write_reg(loader.idx_reg, index)
                        self.emit(loader)
                        if late_initialization:
                            self.write_reg(loader.idx_reg, index)
            if time_consumer:
                self.timeline_inc("duration")
            if unregistered == "after":
                self.append_macro(UnregisteredIndexWrite())

    return Program(load_soccfg(), reps=2, initial_delay=0, relax_delay=0)


def test_branch_results_keep_both_valid_input_proofs_with_distinct_indices():
    prog = branch_program()
    result = ir.analyze_timing(
        ir.CompiledTimingInput(
            prog.prog_list, prog.labels, prog.timing_layout, prog.compile_datamem()
        )
    )
    # Axis 0 loads left[0]=4. Axis 1 loads right[1]=11. Reps restarts axis.
    assert [shot.body_end_cycle - shot.start_cycle for shot in result.shots] == [
        4,
        11,
        4,
        11,
    ]
    assert [shot.end_cycle - shot.start_cycle for shot in result.shots] == [
        4,
        11,
        4,
        11,
    ]


@pytest.mark.parametrize("uninitialized", [0, 1])
@pytest.mark.parametrize("late_initialization", [False, True])
def test_every_branch_lookup_requires_inputs_at_lookup_not_at_later_consumer(
    uninitialized, late_initialization
):
    index = "left_index" if uninitialized == 0 else "right_index"
    with pytest.raises(ValueError, match=f"{index}: unknown or uninitialized"):
        branch_program(
            uninitialized=uninitialized, late_initialization=late_initialization
        )


@pytest.mark.parametrize("position", ["before", "after"])
def test_global_audit_still_checks_early_and_late_producer_dependency_writes(position):
    with pytest.raises(
        ValueError, match="left_index: unregistered.*write at instruction"
    ):
        branch_program(unregistered=position)


def test_non_time_lookup_does_not_gain_a_new_input_initialization_restriction():
    prog = branch_program(
        uninitialized=0, late_initialization=True, time_consumer=False
    )
    prog.compile()
