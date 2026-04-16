from __future__ import annotations

import logging
import math

from qick.asm_v2 import QickParam
from typing_extensions import TYPE_CHECKING, Self, Sequence, TypeAlias, Union

if TYPE_CHECKING:
    from zcu_tools.program.v2.modular import ModularProgramV2


from .base import Module
from .control import Repeat

logger = logging.getLogger(__name__)

SubModule: TypeAlias = Union[Module, list[Module]]


class LoadValue(Module):
    def __init__(
        self,
        name: str,
        values: Sequence[int],
        idx_reg: str,
        val_reg: str,
        use_existed: bool = False,
        auto_compress: bool = True,
    ) -> None:
        self.name = name
        self.values = list(values)
        if len(self.values) == 0:
            raise ValueError("LoadValue requires a non-empty values sequence")
        self.use_existed = use_existed
        self.auto_compress = auto_compress

        self.idx_reg = idx_reg
        self.val_reg = val_reg
        self.addr_reg = ""
        self.word_reg = ""
        self.word_idx_reg = ""
        self.slot_reg = ""
        self.shift_reg = ""
        self.sign_reg = ""
        self.offset = 0
        self.shift_offset = 0

        self._packed_values: list[int] = list(self.values)
        self._is_compressed = False
        self._signed_mode = False
        self._bits_per_value = 32
        self._values_per_word = 1
        self._slot_mask = 0
        self._value_mask = 0xFFFFFFFF
        self._sign_bit_mask = 0
        self._sign_bias = 0
        self._word_shift = 0
        self._shift_table: list[int] = []

        self._plan_compression()

    def init(self, prog: ModularProgramV2) -> None:
        self.offset = prog.add_dmem(self._packed_values)
        if self._is_compressed:
            self.shift_offset = prog.add_dmem(self._shift_table)

        temp_reg_num = 1
        if self._is_compressed:
            temp_reg_num = 6 if self._signed_mode else 5
        temp_regs = prog.acquire_temp_reg(temp_reg_num)
        self.addr_reg = temp_regs[0]

        if self._is_compressed:
            self.word_idx_reg = temp_regs[1]
            self.slot_reg = temp_regs[2]
            self.shift_reg = temp_regs[3]
            self.word_reg = temp_regs[4]
            if self._signed_mode:
                self.sign_reg = temp_regs[5]

        if not self.use_existed:
            prog.add_reg(self.val_reg)

        logger.debug(
            "LoadValue.init: name='%s', auto_compress=%s, compressed=%s, values=%d, packed=%d, bits=%d, per_word=%d",
            self.name,
            self.auto_compress,
            self._is_compressed,
            len(self.values),
            len(self._packed_values),
            self._bits_per_value,
            self._values_per_word,
        )

    def run(
        self, prog: ModularProgramV2, t: Union[float, QickParam] = 0.0
    ) -> Union[float, QickParam]:
        if not self._is_compressed:
            # addr = bind_sweep_index + dmem_offset
            prog.write_reg(self.addr_reg, self.idx_reg)
            if self.offset != 0:
                prog.inc_reg(self.addr_reg, self.offset)

            prog.read_dmem(dst=self.val_reg, addr=self.addr_reg)
            return t

        idx_hw = prog._get_reg(self.idx_reg)
        shift_hw = prog._get_reg(self.shift_reg)
        word_hw = prog._get_reg(self.word_reg)
        val_hw = prog._get_reg(self.val_reg)

        # word_idx = idx >> log2(values_per_word)
        self._write_op(prog, self.word_idx_reg, f"{idx_hw} ASR #{self._word_shift}")
        # slot = idx & (values_per_word - 1)
        self._write_op(prog, self.slot_reg, f"{idx_hw} AND #{self._slot_mask}")

        # addr = bind_sweep_index + dmem_offset
        prog.write_reg(self.addr_reg, self.word_idx_reg)
        if self.offset != 0:
            prog.inc_reg(self.addr_reg, self.offset)
        prog.read_dmem(dst=self.word_reg, addr=self.addr_reg)

        # shift = shift_table[slot]
        prog.write_reg(self.addr_reg, self.slot_reg)
        if self.shift_offset != 0:
            prog.inc_reg(self.addr_reg, self.shift_offset)
        prog.read_dmem(dst=self.shift_reg, addr=self.addr_reg)

        # extracted = (word >> shift) & value_mask
        self._write_op(prog, self.val_reg, f"{word_hw} ASR {shift_hw}")
        self._write_op(prog, self.val_reg, f"{val_hw} AND #{self._value_mask}")

        if self._signed_mode:
            sign_clear_label = f"{self.name}_sign_clear"
            self._write_op(prog, self.sign_reg, f"{val_hw} AND #{self._sign_bit_mask}")
            prog.cond_jump(sign_clear_label, self.sign_reg, "Z")
            self._write_op(prog, self.val_reg, f"{val_hw} - #{self._sign_bias}")
            prog.label(sign_clear_label)

        return t

    @staticmethod
    def _bits_needed_unsigned(value: int) -> int:
        if value < 0:
            raise ValueError("unsigned bit width requires non-negative value")
        return max(1, value.bit_length())

    @staticmethod
    def _bits_needed_signed(min_value: int, max_value: int) -> int:
        for bits in range(2, 33):
            lower = -(1 << (bits - 1))
            upper = (1 << (bits - 1)) - 1
            if lower <= min_value and max_value <= upper:
                return bits
        return 32

    def _pack_values(self, bits_per_value: int, value_mask: int) -> list[int]:
        packed: list[int] = []
        per_word = self._values_per_word
        for i in range(0, len(self.values), per_word):
            word = 0
            chunk = self.values[i : i + per_word]
            for slot, value in enumerate(chunk):
                encoded = value & value_mask
                word |= encoded << (slot * bits_per_value)
            packed.append(word)
        return packed

    def _plan_compression(self) -> None:
        if not self.auto_compress or len(self.values) < 30:
            return

        min_value = min(self.values)
        max_value = max(self.values)
        self._signed_mode = min_value < 0

        if self._signed_mode:
            bits = self._bits_needed_signed(min_value, max_value)
        else:
            bits = self._bits_needed_unsigned(max_value)

        max_per_word = 32 // bits if bits > 0 else 1
        if max_per_word < 2:
            return

        values_per_word = 1 << int(math.floor(math.log2(max_per_word)))
        if values_per_word < 2:
            return

        packed_len = (len(self.values) + values_per_word - 1) // values_per_word
        if packed_len >= len(self.values):
            return

        value_mask = (1 << bits) - 1

        self._is_compressed = True
        self._bits_per_value = bits
        self._values_per_word = values_per_word
        self._slot_mask = values_per_word - 1
        self._value_mask = value_mask
        self._word_shift = int(math.log2(values_per_word))
        self._shift_table = [i * bits for i in range(values_per_word)]
        self._packed_values = self._pack_values(bits, value_mask)

        if self._signed_mode:
            self._sign_bit_mask = 1 << (bits - 1)
            self._sign_bias = 1 << bits

    def _write_op(self, prog: ModularProgramV2, dst: str, op: str) -> None:
        prog.asm_inst(
            {
                "CMD": "REG_WR",
                "DST": prog._get_reg(dst),
                "SRC": "op",
                "OP": op,
            }
        )


class ScanWith(Module):
    def __init__(
        self, name: str, values: Sequence[int], val_reg: str, use_existed: bool = False
    ) -> None:
        self.name = name
        self.repeat_mod = Repeat(name=f"{name}_repeat", n=len(values))
        self.repeat_mod.add_content(
            LoadValue(
                name=f"{name}_load",
                idx_reg=self.repeat_mod.idx_reg,
                val_reg=val_reg,
                values=values,
                use_existed=use_existed,
            )
        )

    def add_content(self, mod: SubModule) -> Self:
        self.repeat_mod.add_content(mod)
        return self

    def init(self, prog: ModularProgramV2) -> None:
        self.repeat_mod.init(prog)

    def run(
        self, prog: ModularProgramV2, t: Union[float, QickParam] = 0.0
    ) -> Union[float, QickParam]:
        return self.repeat_mod.run(prog, t)
