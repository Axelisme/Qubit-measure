"""Native row-major table indices observed at public compiled consumers."""

from collections.abc import Mapping, Sequence

import pytest
from qick.asm_v2 import AsmInst, Macro, WriteReg
from zcu_tools.program.v2.macro.dmem import DmemLookup, sweep_table_index
from zcu_tools.program.v2.mocksoc import make_mock_soccfg
from zcu_tools.program.v2.timeline import TimelineProgramV2

from tests.program.v2.timing_support import shots


class IndexedReads(Macro):
    def __init__(self, axes: Sequence[Mapping[str, int]], offset: int):
        self.axes = axes
        self.offset = offset

    def preprocess(self, prog):
        shared = {}
        self.indices = [
            sweep_table_index(prog, axes, name=f"index_{i}", shared_indices=shared)
            for i, axes in enumerate(self.axes)
        ]

    def translate(self, prog):
        for index in self.indices:
            DmemLookup(
                dst="value", index=index, offset=self.offset, address="address"
            ).translate(prog)
            WriteReg(dst="output", src="value").translate(prog)
            AsmInst(
                inst={
                    "CMD": "DPORT_WR",
                    "DST": "0",
                    "SRC": "reg",
                    "DATA": "r10",
                    "TIME": "@0",
                },
                addr_inc=1,
            ).translate(prog)


def indexed_program(axes: Sequence[Mapping[str, int]]) -> TimelineProgramV2:
    class Program(TimelineProgramV2):
        def _initialize(self, cfg):
            self.add_loop("outer", 2)
            self.add_loop("inner", 3)
            self.add_reg("output", addr=10)
            self.add_reg("value")
            self.add_reg("address")
            self.offset = self.add_dmem([101, 102, 103, 104, 105, 106])

        def _body(self, cfg):
            self.append_macro(IndexedReads(axes, self.offset))

    return Program(make_mock_soccfg(0, 0), reps=2, initial_delay=0, relax_delay=0)


@pytest.mark.parametrize("mixed", [False, True])
def test_native_indices_share_capacity_but_preserve_axis_order_and_counts(mixed):
    axes = (
        [{"outer": 2}, {"inner": 3}, {"outer": 2, "inner": 3}, {"inner": 3, "outer": 2}]
        if mixed
        else [{"outer": 2, "inner": 3}] * 20
    )
    expected = (
        [
            [101, 101, 101, 101],
            [101, 102, 102, 103],
            [101, 103, 103, 105],
            [102, 101, 104, 102],
            [102, 102, 105, 104],
            [102, 103, 106, 106],
        ]
        if mixed
        else [[value] * 20 for value in [101, 102, 103, 104, 105, 106]]
    )
    # Twenty identical requests must fit the ordinary 16-register profile.
    # Observe values, not index names or the compile-local sharing dictionary.
    for prog in [indexed_program(axes), indexed_program(axes)]:
        for _ in range(2):
            assert [[e.value for e in s.events] for s in shots(prog)] == expected * 2
            prog.binprog = None
            prog.compile()


class InvalidCachedIndex(Macro):
    def __init__(self, axes: Mapping[str, int]):
        self.axes = axes

    def preprocess(self, prog):
        # A caller-supplied hit must not bypass validation. The token is never
        # dereferenced: invalid input must fail before returning any register.
        sweep_table_index(
            prog,
            self.axes,
            name="invalid",
            shared_indices={tuple(self.axes.items()): "unallocated"},
        )


@pytest.mark.parametrize(
    "axes, count, message",
    [
        ({"outer": 0}, 2, "invalid native sweep axis outer"),
        ({"outer": 3}, 2, "invalid native sweep axis outer"),
        ({"missing": 2}, 2, "invalid native sweep axis missing"),
        ({"outer": 65536, "inner": 65536}, 65536, "signed 32-bit"),
    ],
)
def test_invalid_native_index_hit_still_rejects_axes_counts_and_overflow(
    axes, count, message
):
    class Program(TimelineProgramV2):
        def _initialize(self, cfg):
            self.add_loop("outer", count)
            self.add_loop("inner", count)

        def _body(self, cfg):
            self.append_macro(InvalidCachedIndex(axes))

    with pytest.raises(ValueError, match=message):
        Program(make_mock_soccfg(0, 0), initial_delay=0, relax_delay=0)
