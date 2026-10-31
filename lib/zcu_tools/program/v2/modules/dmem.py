from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Self, override

from qick.asm_v2 import QickParam

if TYPE_CHECKING:
    from zcu_tools.program.v2.modular import ModularProgramV2
    from zcu_tools.program.v2.timeline import TimelineProgramV2


from .._timeline_bounds import CycleBounds
from ..macro.dmem import DmemLookup
from ..macro.table import DeclaredTable, TableSelection
from .base import Module, ModuleGroup
from .control import Repeat

logger = logging.getLogger(__name__)

# Packing pays off only when the dmem savings outweigh the extra shift/mask
# instructions emitted per access. Empirically, ~30 values is the break-even.
_COMPRESS_MIN_VALUES = 30

# Individual (uncompressed) dmem values must be non-negative so that a direct
# read_dmem into a register is always a valid non-negative integer regardless of
# the instruction's sign-extension behaviour.  Packed words produced by the
# compression path may legitimately set bit 31; the SR + AND extraction in run()
# recovers the correct value in either arithmetic or logical shift mode.
_INT32_MAX = (1 << 31) - 1
_UINT32_MAX = (1 << 32) - 1


def _encode_uint32_as_int32(value: int) -> int:
    if value < 0 or value > _UINT32_MAX:
        raise ValueError(f"raw uint32 word must be in [0, {_UINT32_MAX}]")
    if value > _INT32_MAX:
        return value - (1 << 32)
    return value


class LoadValue(Module):
    """Load a scalar table value selected by an index register.

    Empty tables emit no runtime lookup and are only valid on paths that do not
    execute the lookup. The destination is still declared at compile time unless
    use_existed=True, so consumers on those paths can resolve its register name.
    """

    def __init__(
        self,
        name: str,
        values: Sequence[int],
        idx_reg: str,
        val_reg: str,
        *,
        use_existed: bool = False,
        auto_compress: bool = True,
    ) -> None:
        self.name = name
        self.values = [int(v) for v in values]
        if any(v < 0 or v > _INT32_MAX for v in self.values):
            raise ValueError(
                f"LoadValue values must be in [0, {_INT32_MAX}] (signed int32 positive range)"
            )
        self.use_existed = use_existed
        self.auto_compress = auto_compress
        self._is_empty = len(self.values) == 0

        self.idx_reg = idx_reg
        self.val_reg = val_reg
        self.offset = 0

        self._packed_values: list[int] = list(self.values)
        self._is_compressed = False
        self._bits_per_value = 32
        self._values_per_word = 1
        self._slot_mask = 0
        self._value_mask = 0xFFFFFFFF
        self._word_shift = 0
        self._bits_shift = 0

        self._plan_compression()

    @override
    def declare(self, prog: TimelineProgramV2) -> None:
        """Allocate the scalar destination and signed-int32 dmem table on prog.

        use_existed=True leaves destination ownership to the caller. Empty
        tables still declare the destination but allocate no memory.
        Register allocation and dmem errors propagate from prog.
        """
        if not self.use_existed:
            prog.add_reg(self.val_reg)
        self._table = DeclaredTable(tuple(self.values))
        if not self._is_empty:
            self.offset = prog.add_dmem(self._packed_values)

    @override
    def lower(
        self, prog: TimelineProgramV2, *, timeline: str | None, tag: str | None
    ) -> None:
        """Read index idx_reg into scalar val_reg without changing any timeline.

        idx_reg must hold a valid zero-based table index. Empty tables emit no
        lookup. timeline and tag belong to the emission and do not alter lookup.
        Unknown registers and allocation errors propagate from prog.
        """
        self.lower_index(prog, self.idx_reg)

    def lower_index(
        self,
        prog: TimelineProgramV2,
        index: str,
    ) -> None:
        """Load from this declared table using this emission's index register.

        The caller owns index and must initialize it to a valid zero-based
        table index. This method emits only during native body lowering and
        reuses the declared table/destination; it allocates neither. It does
        not mutate idx_reg or any emission state, and does not advance any
        timeline. Empty tables emit nothing. Unknown registers
        and invalid timing dependency bounds fail during compilation.
        """
        if self.values:
            with prog.time_producer(
                self.val_reg,
                CycleBounds(min(self.values), max(self.values)),
                dependency_ranges={index: CycleBounds(0, len(self.values) - 1)},
                selection=TableSelection(self._table, self.val_reg, index),
            ):
                self._emit_lookup(prog, index)

    def lower_selected(self, prog: TimelineProgramV2) -> TableSelection:
        """Load a selected-playback scalar and capture its index at this position.

        Declaration must precede this call. The returned immutable relation binds
        actual playback to this table and its exact writers. A loader-owned
        capture survives later author selector changes. Repeated emissions reuse
        registers, with each occurrence loading and capturing before playback.
        Empty tables fail. Ordinary lower/lower_index keep their scalar contract.
        """
        if not self.values:
            raise ValueError(f"{self.name}: selected playback needs a nonempty table")
        captured = prog.add_reg(f"{self.name}_selected_index", allow_reuse=True)
        relation = TableSelection(self._table, self.val_reg, captured)
        index_limits = CycleBounds(0, len(self.values) - 1)
        with prog.time_producer(
            captured,
            index_limits,
            dependency_ranges={self.idx_reg: index_limits},
            selection=relation,
        ):
            prog.write_reg(captured, self.idx_reg)
        with prog.time_producer(
            self.val_reg,
            CycleBounds(min(self.values), max(self.values)),
            dependency_ranges={captured: index_limits},
            selection=relation,
        ):
            self._emit_lookup(prog, captured)
        return relation

    @property
    def is_compressed(self) -> bool:
        """Whether several values share one dmem word."""
        return self._is_compressed

    def init(self, prog: ModularProgramV2) -> None:
        # QICK resolves all emitted consumers before unreachable code is removed.
        if not self.use_existed:
            prog.add_reg(self.val_reg)

        if self._is_empty:
            logger.debug(
                "LoadValue.init: short-circuit empty values name='%s' (no dmem allocated)",
                self.name,
            )
            return

        self.offset = prog.add_dmem(self._packed_values)

        logger.debug(
            "LoadValue.init: name='%s', auto_compress=%s, compressed=%s, "
            "values=%d, packed=%d, bits=%d, per_word=%d",
            self.name,
            self.auto_compress,
            self._is_compressed,
            len(self.values),
            len(self._packed_values),
            self._bits_per_value,
            self._values_per_word,
        )

    def run(
        self, prog: ModularProgramV2, t: float | QickParam = 0.0
    ) -> float | QickParam:
        self._emit_lookup(prog, self.idx_reg)
        return t

    def _emit_lookup(
        self, prog: ModularProgramV2 | TimelineProgramV2, index: str
    ) -> None:
        if self._is_empty:
            return

        # addr_reg computes dmem address / shift amount; word_reg holds the
        # fetched packed word. addr_reg is reused for shift once word is loaded.
        temp_reg_num = 2 if self._is_compressed else 1
        with prog.acquire_temp_reg(temp_reg_num) as (addr_reg, *other_regs):
            word_reg = other_regs[0] if self._is_compressed else ""  # not use

            if not self._is_compressed:
                prog.append_macro(
                    DmemLookup(
                        dst=self.val_reg,
                        index=index,
                        offset=self.offset,
                        address=addr_reg,
                    )
                )
                return

            # addr = (idx ASR #word_shift) [+ #offset]
            prog.write_reg_op(addr_reg, index, "ASR", self._word_shift)
            if self.offset != 0:
                prog.inc_reg(addr_reg, self.offset)
            prog.read_dmem(dst=word_reg, addr=addr_reg)

            # tProc v2 uses only B[3:0] as the shift count, including register
            # operands. Split the desired 0..31-bit shift into two equal
            # 0..15-bit shifts, plus one odd bit for 1-bit packed values.
            # addr_reg is safe to reuse here: read_dmem above has already
            # consumed it as the address, and word_reg now holds the fetched
            # word, so addr_reg's value is dead.
            shift_reg = addr_reg
            prog.write_reg_op(shift_reg, index, "AND", self._slot_mask)
            if self._bits_shift == 0:
                prog.write_reg_op(shift_reg, shift_reg, "ASR", 1)
            elif self._bits_shift > 1:
                prog.write_reg_op(shift_reg, shift_reg, "SL", self._bits_shift - 1)
            prog.write_reg_op(self.val_reg, word_reg, "SR", shift_reg)
            prog.write_reg_op(self.val_reg, self.val_reg, "SR", shift_reg)
            if self._bits_shift == 0:
                prog.write_reg_op(shift_reg, index, "AND", 1)
                prog.write_reg_op(self.val_reg, self.val_reg, "SR", shift_reg)

            prog.write_reg_op(self.val_reg, self.val_reg, "AND", self._value_mask)

    def allow_rerun(self) -> bool:
        return True

    @staticmethod
    def _bits_needed(value: int) -> int:
        if value < 0:
            raise ValueError("unsigned bit width requires non-negative value")
        return max(1, value.bit_length())

    def _pack_values(self, bits_per_value: int, value_mask: int) -> list[int]:
        packed: list[int] = []
        per_word = self._values_per_word
        for i in range(0, len(self.values), per_word):
            word = 0
            chunk = self.values[i : i + per_word]
            for slot, value in enumerate(chunk):
                encoded = value & value_mask
                word |= encoded << (slot * bits_per_value)
            # Packed words may use all 32 bits (including bit 31).  Convert to
            # the signed int32 two's-complement representation so that the value
            # fits in np.int32 without overflow; the hardware bit pattern is
            # identical and the SR + AND extraction in run() remains correct.
            packed.append(_encode_uint32_as_int32(word))
        return packed

    def _plan_compression(self) -> None:
        if not self.auto_compress or len(self.values) < _COMPRESS_MIN_VALUES:
            return

        max_value = max(self.values)
        bits = self._bits_needed(max_value)

        # Round bits up to a power of 2 so the shift amount is always
        # (slot << bits_shift), avoiding a dmem shift table. This never hurts
        # packing density: values_per_word = pow2_floor(32 // bits) already
        # collapses non-pow2 bits to the same value as the next pow2 above.
        bits = 1 if bits <= 1 else 1 << (bits - 1).bit_length()

        values_per_word = 32 // bits
        if values_per_word < 2:
            return

        value_mask = (1 << bits) - 1

        self._is_compressed = True
        self._bits_per_value = bits
        self._values_per_word = values_per_word
        self._slot_mask = values_per_word - 1
        self._value_mask = value_mask
        self._word_shift = int(math.log2(values_per_word))
        self._bits_shift = int(math.log2(bits))
        self._packed_values = self._pack_values(bits, value_mask)


class LoadWord(Module):
    """Load an uncompressed raw uint32 word selected by idx_reg into val_reg.

    values accepts 0..2**32-1; dmem stores signed int32 with identical bits.
    use_existed=True requires a caller-owned destination. Empty tables declare
    no registers or memory and emit no lookup. No default value is supplied.
    """

    def __init__(
        self,
        name: str,
        values: Sequence[int],
        idx_reg: str,
        val_reg: str,
        *,
        use_existed: bool = False,
    ) -> None:
        self.name = name
        self.values = [int(v) for v in values]
        self._encoded_values = [_encode_uint32_as_int32(v) for v in self.values]
        self.use_existed = use_existed
        self._is_empty = len(self.values) == 0

        self.idx_reg = idx_reg
        self.val_reg = val_reg
        self.offset = 0

    @override
    def declare(self, prog: TimelineProgramV2) -> None:
        """Allocate nonempty raw-word dmem and, unless use_existed, destination.

        Empty tables allocate neither. Allocation errors propagate from prog.
        """
        if self._is_empty:
            return
        self._table = DeclaredTable(tuple(self.values))
        self.offset = prog.add_dmem(self._encoded_values)
        if not self.use_existed:
            prog.add_reg(self.val_reg)

    @override
    def lower(
        self, prog: TimelineProgramV2, *, timeline: str | None, tag: str | None
    ) -> None:
        """Copy the indexed raw word to val_reg without moving any timeline.

        idx_reg must hold a valid zero-based table index. Empty tables emit
        nothing. timeline and tag identify the emission, not a lookup time.
        Unknown registers and allocation errors propagate from prog.
        """
        if self.values:
            with prog.time_producer(
                self.val_reg,
                CycleBounds(min(self.values), max(self.values)),
                dependency_ranges={self.idx_reg: CycleBounds(0, len(self.values) - 1)},
                selection=TableSelection(self._table, self.val_reg, self.idx_reg),
            ):
                self._emit_lookup(prog)

    def init(self, prog: ModularProgramV2) -> None:
        if self._is_empty:
            logger.debug(
                "LoadWord.init: short-circuit empty values name='%s' (no dmem allocated)",
                self.name,
            )
            return

        self.offset = prog.add_dmem(self._encoded_values)

        if not self.use_existed:
            prog.add_reg(self.val_reg)

        logger.debug(
            "LoadWord.init: name='%s', values=%d",
            self.name,
            len(self.values),
        )

    def run(
        self, prog: ModularProgramV2, t: float | QickParam = 0.0
    ) -> float | QickParam:
        self._emit_lookup(prog)
        return t

    def _emit_lookup(self, prog: ModularProgramV2 | TimelineProgramV2) -> None:
        if self._is_empty:
            return

        with prog.acquire_temp_reg(1) as (addr_reg,):
            if self.offset == 0:
                prog.write_reg(addr_reg, self.idx_reg)
            else:
                prog.write_reg_op(addr_reg, self.idx_reg, "+", self.offset)
            prog.read_dmem(dst=self.val_reg, addr=addr_reg)

    def allow_rerun(self) -> bool:
        return True


class ScanWith(Module):
    """Scan a nonempty scalar table in order, loading before each child sequence.

    Each native emission gets an independent fixed-count loop and counter.
    Each point synchronizes the selected timeline before the next point starts.
    Destination registers belong to the declaration unless an existing register
    is explicitly shared with use_existed=True. Values follow the
    nonnegative signed-int32 LoadValue contract. Empty values raise ValueError.
    """

    def __init__(
        self,
        name: str,
        values: Sequence[int],
        val_reg: str,
        *,
        use_existed: bool = False,
    ) -> None:
        self.name = name

        n = len(values)
        if n == 0:
            raise ValueError("ScanWith requires a non-empty values sequence")
        repeat_mod = Repeat(f"{name}_count", n, range_hint=(0, n - 1))
        self.value_loader = LoadValue(
            name=f"{name}_load",
            idx_reg=repeat_mod.name,
            val_reg=val_reg,
            values=values,
            use_existed=use_existed,
        )
        repeat_mod.add_content(self.value_loader)

        self.repeat_mod = repeat_mod

    @override
    def declare(self, prog: TimelineProgramV2) -> None:
        """Declare the table destination and all children on prog, in order.

        The native loop owns its index register. use_existed=True leaves the
        value register to the caller. Child declaration errors propagate.
        """
        for mod in self.repeat_mod.sub_modules:
            prog.declare(mod)

    @override
    def lower(
        self, prog: TimelineProgramV2, *, timeline: str | None, tag: str | None
    ) -> None:
        """Load each scalar then emit children on timeline (default if None).

        Synchronize the parent at entry and close each point's actual end via
        a fixed-count native Repeat. tag identifies this scan emission; children
        remain separate emissions. Invalid timeline or child errors propagate.
        """
        parent = "default" if timeline is None else timeline
        index = prog.emission_name(self.repeat_mod.name)
        with prog.repeat(index, self.repeat_mod.n, timeline=parent):
            self.value_loader.lower_index(prog, index)
            for mod in self.repeat_mod.sub_modules[1:]:
                prog.emit(mod, timeline=parent)

    def add_content(self, mod: ModuleGroup) -> Self:
        """Append a module or ordered group; skip None entries (no module)."""
        self.repeat_mod.add_content(mod)
        return self

    def init(self, prog: ModularProgramV2) -> None:
        self.repeat_mod.init(prog)

    def run(
        self, prog: ModularProgramV2, t: float | QickParam = 0.0
    ) -> float | QickParam:
        return self.repeat_mod.run(prog, t, internal_modules=(self.value_loader,))
