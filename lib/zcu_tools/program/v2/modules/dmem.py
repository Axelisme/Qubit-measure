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
        self.offset = 0

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
        self._bits_shift = 0

        self._plan_compression()

    def init(self, prog: ModularProgramV2) -> None:
        self.offset = prog.add_dmem(self._packed_values)

        # addr_reg computes dmem address / shift amount; word_reg holds the
        # fetched packed word. addr_reg is reused for shift once word is loaded.
        temp_reg_num = 2 if self._is_compressed else 1
        temp_regs = prog.acquire_temp_reg(temp_reg_num)
        self.addr_reg = temp_regs[0]
        if self._is_compressed:
            self.word_reg = temp_regs[1]

        if not self.use_existed:
            prog.add_reg(self.val_reg)

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
        self, prog: ModularProgramV2, t: Union[float, QickParam] = 0.0
    ) -> Union[float, QickParam]:
        addr_reg = self.addr_reg
        if not self._is_compressed:
            # addr = idx [+ offset]
            if self.offset == 0:
                prog.write_reg(addr_reg, self.idx_reg)
            else:
                prog.write_reg_op(addr_reg, self.idx_reg, "+", self.offset)
            prog.read_dmem(dst=self.val_reg, addr=addr_reg)
            return t

        # addr = (idx ASR #word_shift) [+ #offset]
        prog.write_reg_op(addr_reg, self.idx_reg, "ASR", self._word_shift)
        if self.offset != 0:
            prog.inc_reg(addr_reg, self.offset)
        prog.read_dmem(dst=self.word_reg, addr=addr_reg)

        # shift = (idx AND #slot_mask) [SL #bits_shift]
        shift_reg = addr_reg  # reuse addr_reg
        prog.write_reg_op(shift_reg, self.idx_reg, "AND", self._slot_mask)
        if self._bits_shift > 0:
            prog.write_reg_op(shift_reg, shift_reg, "SL", self._bits_shift)
        prog.write_reg_op(self.val_reg, self.word_reg, "ASR", shift_reg)

        prog.write_reg_op(self.val_reg, self.val_reg, "AND", self._value_mask)

        if self._signed_mode:
            sign_clear_label = f"{self.name}_sign_clear"
            # skip the bias subtraction when the sign bit is clear
            prog.cond_jump(
                sign_clear_label, self.val_reg, "Z", "&", self._sign_bit_mask
            )
            prog.inc_reg(self.val_reg, -self._sign_bias)
            prog.label(sign_clear_label)

        return t

    def allow_rerun(self) -> bool:
        return not self._is_compressed or not self._signed_mode

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
        raise ValueError(
            f"values out of signed 32-bit range: min={min_value}, max={max_value}"
        )

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

        # Round bits up to a power of 2 so the shift amount is always
        # (slot << bits_shift), avoiding a dmem shift table. This never hurts
        # packing density: values_per_word = pow2_floor(32 // bits) already
        # collapses non-pow2 bits to the same value as the next pow2 above.
        bits = 1 if bits <= 1 else 1 << (bits - 1).bit_length()
        if bits > 32:
            return

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

        if self._signed_mode:
            self._sign_bit_mask = 1 << (bits - 1)
            self._sign_bias = 1 << bits


class ScanWith(Module):
    def __init__(
        self, name: str, values: Sequence[int], val_reg: str, use_existed: bool = False
    ) -> None:
        self.name = name
        repeat_mod = Repeat(f"{name}_count", len(values))
        repeat_mod.add_content(
            LoadValue(
                name=f"{name}_load",
                idx_reg=repeat_mod.name,
                val_reg=val_reg,
                values=values,
                use_existed=use_existed,
            )
        )

        self.repeat_mod = repeat_mod

    def add_content(self, mod: SubModule) -> Self:
        self.repeat_mod.add_content(mod)
        return self

    def init(self, prog: ModularProgramV2) -> None:
        self.repeat_mod.init(prog)

    def run(
        self, prog: ModularProgramV2, t: Union[float, QickParam] = 0.0
    ) -> Union[float, QickParam]:
        return self.repeat_mod.run(prog, t)
