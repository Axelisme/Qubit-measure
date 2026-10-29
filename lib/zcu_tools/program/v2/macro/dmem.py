"""Uncompressed scalar lookup and native swept table-index ownership."""

from collections.abc import Mapping

from qick.asm_v2 import Macro, QickParam, QickProgramV2, ReadDmem, WriteReg

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


def sweep_table_index(
    prog: QickProgramV2, axes: Mapping[str, int], *, name: str
) -> str:
    """Allocate an exact row-major index over native QICK sweep axes.

    Call during preprocess, after native loop discovery. axes maps native loop
    names to their positive counts, outermost first. Only selected axes enter
    the index. QICK owns its initialization, increments and resets, including
    repeated entry from outer loops. Invalid axes/counts fail before allocation.
    name is a fresh register name. The returned register counts table words,
    not bytes or time. Its native sweep initializer covers 0..product(counts)-1
    at lookups. The timing caller must register the axes' loop counters as bounds
    dependencies and audit them together with this register's native writers.
    """
    stride = 1
    spans: dict[str, int] = {}
    for axis, count in reversed(tuple(axes.items())):
        if count < 1 or prog.loop_dict.get(axis) != count:
            raise ValueError(f"{name}: invalid native sweep axis {axis}")
        spans[axis] = stride * (count - 1)
        stride *= count
    if stride > 2**31 - 1:
        raise ValueError(f"{name}: table index exceeds signed 32-bit cycles")
    raw = QickParam(0, spans).to_int(scale=1, quantize=1, parname=name)
    raw.to_steps(prog.loop_dict)
    return prog.add_reg(name=name, init=raw)
