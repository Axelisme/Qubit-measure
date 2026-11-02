"""Uncompressed scalar lookup with a caller-owned index register."""

from qick.asm_v2 import Macro, QickProgramV2, ReadDmem, WriteReg

from .write_reg import WriteRegOp


class DmemLookup(Macro):
    """Read one uncompressed dmem word into dst at index + offset.

    index and address are declared register names; address is caller-owned
    scratch, distinct from index and dst. offset is a zero-based word offset.
    The caller proves index bounds and initialization before using the result
    as time. This macro neither allocates storage nor changes reference time.
    """

    def __init__(self, *, dst: str, index: str, offset: int, address: str):
        super().__init__()
        self.dst = dst
        self.index = index
        self.offset = offset
        self.address = address

    def translate(self, prog: QickProgramV2) -> None:
        copy = (
            WriteReg(dst=self.address, src=self.index)
            if self.offset == 0
            else WriteRegOp(dst=self.address, lhs=self.index, op="+", rhs=self.offset)
        )
        copy.translate(prog)
        ReadDmem(dst=self.dst, addr=self.address).translate(prog)
