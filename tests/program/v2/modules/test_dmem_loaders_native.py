"""Native table declarations and register consumers through linked compilation."""

from collections.abc import Callable, Sequence

import pytest
from zcu_tools.program.v2 import ir
from zcu_tools.program.v2.modules import LoadValue, LoadWord
from zcu_tools.program.v2.timeline import TimelineProgramV2

from tests.program._profiles import TPROC_MHZ
from tests.program._profiles import load_soccfg


def compile_lookup(
    loader: LoadValue | LoadWord,
    *,
    prefix: Sequence[int] = (),
    indices: Sequence[int] | None = None,
    existing_destination: bool = False,
    before_body: Callable[[], None] | None = None,
):
    class Program(TimelineProgramV2):
        def _initialize(self, cfg):
            self.add_reg("index")
            self.add_reg("output", addr=10)
            self.add_dmem(prefix)
            if existing_destination:
                self.add_reg("value")
            self.declare(loader)

        def _body(self, cfg):
            if before_body is not None:
                before_body()
            self.fork("default", into="lookup")
            self.timeline_inc(13 / TPROC_MHZ, timeline="lookup")
            for index in range(len(loader.values)) if indices is None else indices:
                self.write_reg("index", index)
                self.emit(loader, timeline="lookup")
                self.delay_by_timeline()  # Lookup must not accidentally advance default either.
                # Native register consumer: no address reconstruction or decoder.
                self.write_reg("output", "value")
                self.asm_inst(
                    {
                        "CMD": "DPORT_WR",
                        "DST": "0",
                        "SRC": "reg",
                        "DATA": "r10",
                        "TIME": "@3",
                    }
                )

    return Program(load_soccfg(), initial_delay=5 / TPROC_MHZ, relax_delay=7 / TPROC_MHZ)


def compiled_words(prog: TimelineProgramV2) -> list[int]:
    words = prog.compile_datamem()
    return [] if words is None else words.tolist()


def analyze(prog: TimelineProgramV2) -> ir.TimingAnalysis:
    return ir.analyze_timing(
        ir.CompiledTimingInput(
            prog.prog_list, prog.labels, prog.timing_layout, prog.compile_datamem()
        )
    )


@pytest.mark.parametrize("loader_type", [LoadValue, LoadWord])
@pytest.mark.parametrize("values", [[13, 29], [0, 1, 2, 3] * 16])
def test_shared_lookup_keeps_each_programs_dmem_allocation(loader_type, values):
    shared = loader_type("shared", values, "index", "value")
    other_programs = []

    def build_other():
        other_programs.append(compile_lookup(shared, prefix=(91, 83, 79)))

    first = compile_lookup(shared, before_body=build_other)
    fresh = compile_lookup(loader_type("shared", values, "index", "value"))
    other_fresh = compile_lookup(
        loader_type("shared", values, "index", "value"), prefix=(91, 83, 79)
    )
    assert first.prog_list == fresh.prog_list
    assert analyze(first) == analyze(fresh)
    assert other_programs[0].prog_list == other_fresh.prog_list
    first.compile()
    assert first.prog_list == fresh.prog_list


@pytest.mark.parametrize("prefix", [(), (91, -7, 83)])
def test_scalar_lookup_reaches_consumer_without_advancing_timeline(prefix):
    loader = LoadValue("table", [0, 17, 2**31 - 1], "index", "value")
    prog = compile_lookup(loader, prefix=prefix)
    result = analyze(prog)
    assert compiled_words(prog) == [*prefix, 0, 17, 2147483647]
    assert [(event.cycle, event.value) for event in result.shots[0].events] == [
        (8, 0),
        (8, 17),
        (8, 2147483647),
    ]
    # Initial reference 5 + unchanged named end 13; shell adds 7.
    assert (result.shots[0].body_end_cycle, result.shots[0].end_cycle) == (18, 25)


@pytest.mark.parametrize("prefix", [(), (91, -7, 83)])
def test_raw_words_keep_all_bits_at_the_register_consumer(prefix):
    loader = LoadWord(
        "words", [0, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF], "index", "value"
    )
    prog = compile_lookup(loader, prefix=prefix)
    result = analyze(prog)
    assert compiled_words(prog) == [*prefix, 0, 2147483647, -2147483648, -1]
    assert [(event.cycle, event.value) for event in result.shots[0].events] == [
        (8, 0),
        (8, 2147483647),
        (8, -2147483648),
        (8, -1),
    ]
    assert (result.shots[0].body_end_cycle, result.shots[0].end_cycle) == (18, 25)


@pytest.mark.parametrize("prefix", [(), (91, -7, 83)])
@pytest.mark.parametrize(
    "pattern,word_count",
    [
        ([0, 1], 3),
        ([0, 1, 2, 3], 5),
        ([0, 1, 7, 15], 9),
        ([0, 1, 127, 255], 17),
        ([0, 1, 32767, 65535], 33),
    ],
)
def test_compressed_scalar_lookup_unpacks_every_slot(pattern, word_count, prefix):
    # 65 indices cross two word boundaries even for one-bit packing. The final
    # one-entry word checks partial packing; full words also exercise bit 31.
    values = pattern * (64 // len(pattern)) + [1]
    loader = LoadValue("packed", values, "index", "value")
    prog = compile_lookup(loader, prefix=prefix)
    result = analyze(prog)
    assert loader.is_compressed
    assert len(compiled_words(prog)) == len(prefix) + word_count
    assert [event.value for event in result.shots[0].events] == values
    assert {event.cycle for event in result.shots[0].events} == {8}
    assert (result.shots[0].body_end_cycle, result.shots[0].end_cycle) == (18, 25)


@pytest.mark.parametrize("auto_compress", [False, True])
def test_uncompressed_large_scalar_tables_preserve_values(auto_compress):
    values = [0, 65536, 2147483647] * 11
    loader = LoadValue("large", values, "index", "value", auto_compress=auto_compress)
    prog = compile_lookup(loader)
    assert not loader.is_compressed
    assert compiled_words(prog) == values
    assert [event.value for event in analyze(prog).shots[0].events] == values


def test_small_scalar_table_can_disable_compression():
    values = [0, 3, 1, 2] * 16
    loader = LoadValue("plain", values, "index", "value", auto_compress=False)
    prog = compile_lookup(loader)
    assert not loader.is_compressed
    assert compiled_words(prog) == values
    assert [event.value for event in analyze(prog).shots[0].events] == values


@pytest.mark.parametrize("loader_type", [LoadValue, LoadWord])
def test_existing_destination_is_reused_by_nonempty_lookup(loader_type):
    loader = loader_type("reuse", [13, 29], "index", "value", use_existed=True)
    prog = compile_lookup(loader, existing_destination=True)
    assert [event.value for event in analyze(prog).shots[0].events] == [13, 29]


@pytest.mark.parametrize("loader_type", [LoadValue, LoadWord])
def test_missing_existing_destination_is_rejected(loader_type):
    loader = loader_type("missing", [13], "index", "value", use_existed=True)
    with pytest.raises(KeyError, match="value"):
        compile_lookup(loader)


@pytest.mark.parametrize("loader_type", [LoadValue, LoadWord])
@pytest.mark.parametrize("use_existed", [False, True])
def test_empty_lookup_preserves_destination_and_allocates_no_dmem(
    loader_type, use_existed
):
    loader = loader_type("empty", [], "unused_index", "value", use_existed=use_existed)

    class Program(TimelineProgramV2):
        def _initialize(self, cfg):
            self.add_reg("output", addr=10)
            if use_existed:
                self.add_reg("value")
            self.declare(loader)
            # Unlike scalar empty tables, raw-word empty tables do not allocate
            # a destination. A later owner can still claim this register name.
            if loader_type is LoadWord and not use_existed:
                self.add_reg("value")

        def _body(self, cfg):
            self.write_reg("value", 29)
            self.timeline_inc(13 / TPROC_MHZ)
            self.emit(loader)
            self.write_reg("output", "value")
            self.asm_inst(
                {
                    "CMD": "DPORT_WR",
                    "DST": "0",
                    "SRC": "reg",
                    "DATA": "r10",
                    "TIME": "@3",
                }
            )

    prog = Program(load_soccfg(), initial_delay=0, relax_delay=0)
    assert prog.compile_datamem() is None
    result = analyze(prog)
    assert [(event.cycle, event.value) for event in result.shots[0].events] == [(3, 29)]
    assert (result.shots[0].body_end_cycle, result.shots[0].end_cycle) == (13, 13)


@pytest.mark.parametrize("loader_type", [LoadValue, LoadWord])
@pytest.mark.parametrize("index", [-1, 2])
def test_out_of_buffer_lookup_is_rejected_by_compiled_consumer(loader_type, index):
    prog = compile_lookup(
        loader_type("table", [13, 29], "index", "value"), indices=[index]
    )
    with pytest.raises(ValueError, match="Dmem word address out of range"):
        analyze(prog)


@pytest.mark.parametrize("index", [-1, 96])
def test_compressed_out_of_buffer_address_is_rejected(index):
    loader = LoadValue("packed", [0, 1] * 32 + [1], "index", "value")
    prog = compile_lookup(loader, indices=[index])
    with pytest.raises(ValueError, match="Dmem word address out of range"):
        analyze(prog)


@pytest.mark.parametrize("loader_type", [LoadValue, LoadWord])
def test_unknown_index_register_fails_compilation(loader_type):
    with pytest.raises(KeyError, match="missing_index"):
        compile_lookup(loader_type("table", [13], "missing_index", "value"))


@pytest.mark.parametrize("values", [[13], [0, 1] * 32 + [1]])
def test_lookup_reports_register_exhaustion(values):
    loader = LoadValue("table", values, "index", "value")

    class Program(TimelineProgramV2):
        def _initialize(self, cfg):
            for index in range(31):
                self.add_reg(f"reserved_{index}")
            self.declare(loader)

        def _body(self, cfg):
            self.emit(loader)

    with pytest.raises(RuntimeError, match="more data registers than are available"):
        Program(load_soccfg(), initial_delay=0, relax_delay=0)


@pytest.mark.parametrize("loader_type", [LoadValue, LoadWord])
def test_recompile_rebuilds_table_without_accumulating_offsets(loader_type):
    loader = loader_type("table", [13, 29], "index", "value")
    prog = compile_lookup(loader, prefix=[91, -7, 83])
    prog.compile()
    assert compiled_words(prog) == [91, -7, 83, 13, 29]
    result = analyze(prog)
    assert [event.value for event in result.shots[0].events] == [13, 29]
    assert (result.shots[0].body_end_cycle, result.shots[0].end_cycle) == (18, 25)


@pytest.mark.parametrize("loader_type", [LoadValue, LoadWord])
def test_destination_name_collision_is_rejected(loader_type):
    with pytest.raises(NameError, match="value.*already exists"):
        compile_lookup(
            loader_type("table", [13], "index", "value"), existing_destination=True
        )
