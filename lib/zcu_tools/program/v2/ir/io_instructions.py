"""QICK memory and port instructions sharing the core instruction base."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

from .hw_semantics import STATUS_REG, TIMED_BASE_REG
from .instruction_fields import (
    VALID_TRIG_SRCS,
    CondCode,
    DmemSrc,
    DportSrc,
    TrigSrc,
    parse_cond_code,
    parse_dmem_src_keyword,
    parse_dport_data_field,
    parse_dport_src_keyword,
    parse_mem_addr_field,
    parse_optional_addr,
    parse_optional_immediate,
    parse_optional_side_write,
    parse_optional_src,
    parse_optional_time,
    parse_port_dst,
    parse_wp_port,
    remap_label_ref,
    require_alu_expr,
    require_literal,
    require_register,
)
from .instructions import BaseInst
from .labels import Label, LabelRef
from .operands import (
    AddrType,
    ExprType,
    Immediate,
    ImmValue,
    MemAddr,
    Register,
    SideWrite,
    SrcKeyword,
    SrcType,
    TimeType,
    parse_label,
)


@dataclass(frozen=True)
class PortWriteInst(BaseInst):
    """WPORT_WR instruction: write to output port."""

    dst: Register | ImmValue
    src: SrcType | None = None
    addr: Register | MemAddr | None = None
    time: TimeType | None = None
    wr: SideWrite | None = None
    op: ExprType | None = None
    uf: bool = False
    if_cond: CondCode | None = None
    ww: str | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PortWriteInst:
        return cls(
            dst=parse_port_dst(str(d["DST"])),
            src=parse_optional_src(d, "SRC", "WPORT_WR.SRC"),
            addr=parse_mem_addr_field(str(d["ADDR"]), "ADDR") if "ADDR" in d else None,
            time=parse_optional_time(d, "TIME", "WPORT_WR.TIME"),
            wr=parse_optional_side_write(d, "WR", "WPORT_WR.WR"),
            op=require_alu_expr(d["OP"], "WPORT_WR.OP") if "OP" in d else None,
            uf="UF" in d,
            if_cond=parse_cond_code(d.get("IF")),
            ww=d.get("WW"),
        )

    @property
    def reg_read(self) -> frozenset[str]:
        reads: frozenset[str] = frozenset({TIMED_BASE_REG})
        if isinstance(self.src, Register):
            reads = reads | self.src.regs()
        for op in [self.dst, self.addr, self.time]:
            if isinstance(op, Register):
                reads = reads | op.regs()
        if self.op:
            reads = reads | self.op.regs()
        return reads

    @property
    def reg_write(self) -> frozenset[str]:
        return self.wr.regs() if self.wr else frozenset()

    def to_dict(self) -> dict[str, Any]:
        src_val = (
            self.src.value
            if isinstance(self.src, SrcKeyword)
            else str(self.src)
            if self.src
            else None
        )
        d = {
            "CMD": "WPORT_WR",
            "DST": str(self.dst) if self.dst else None,
            "SRC": src_val,
            "ADDR": str(self.addr) if self.addr else None,
            "TIME": str(self.time) if self.time else None,
            "WR": str(self.wr) if self.wr else None,
            "OP": str(self.op) if self.op else None,
            "UF": "1" if self.uf else None,
            "IF": self.if_cond if self.if_cond else None,
            "WW": self.ww,
        }
        return {k: v for k, v in d.items() if v is not None}


@dataclass(frozen=True)
class DmemReadInst(BaseInst):
    """DMEM read lowered to the native REG_WR form."""

    dst: Register
    src: SrcKeyword = SrcKeyword.DMEM
    addr: AddrType | None = None
    wr: SideWrite | None = None
    op: ExprType | None = None
    lit: Immediate | None = None
    uf: bool = False
    if_cond: CondCode | None = None
    label: LabelRef | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DmemReadInst:
        return cls(
            dst=require_register(d["DST"], "DST"),
            src=SrcKeyword.DMEM,
            addr=parse_optional_addr(d, "ADDR", "REG_WR.ADDR"),
            wr=parse_optional_side_write(d, "WR", "REG_WR.WR"),
            op=require_alu_expr(d["OP"], "REG_WR.OP") if "OP" in d else None,
            lit=parse_optional_immediate(d, "LIT", "REG_WR.LIT"),
            uf="UF" in d,
            if_cond=parse_cond_code(d.get("IF")),
            label=parse_label(d.get("LABEL")),
        )

    @property
    def reg_read(self) -> frozenset[str]:
        reads: frozenset[str] = frozenset()
        if isinstance(self.addr, Register):
            reads = reads | self.addr.regs()
        if self.op:
            reads = reads | self.op.regs()
        return reads

    @property
    def reg_write(self) -> frozenset[str]:
        writes = self.dst.regs() if self.dst else frozenset()
        if self.wr is not None:
            writes = writes | self.wr.regs()
        return writes

    @property
    def need_label(self) -> Label | None:
        if self.label is not None and not self.label.is_pseudo():
            return self.label.as_label()
        return None

    def to_dict(self) -> dict[str, Any]:
        d = {
            "CMD": "REG_WR",
            "DST": str(self.dst) if self.dst else None,
            "SRC": self.src.value,
            "ADDR": str(self.addr) if self.addr else None,
            "WR": str(self.wr) if self.wr else None,
            "OP": str(self.op) if self.op else None,
            "LIT": str(self.lit) if self.lit else None,
            "UF": "1" if self.uf else None,
            "IF": self.if_cond if self.if_cond else None,
            "LABEL": str(self.label) if self.label is not None else None,
        }
        return {k: v for k, v in d.items() if v is not None}

    def remap_labels(self, remap: dict[str, Label]) -> DmemReadInst:
        new_label = remap_label_ref(self.label, remap)
        if new_label is self.label:
            return self
        return dataclasses.replace(self, label=new_label)


@dataclass(frozen=True)
class DmemWriteInst(BaseInst):
    """DMEM_WR instruction: write to data memory."""

    dst: Register | MemAddr
    src: DmemSrc = "imm"
    op: ExprType | None = None
    lit: Immediate | None = None
    wr: SideWrite | None = None
    uf: bool = False
    if_cond: CondCode | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DmemWriteInst:
        dst_raw = str(d["DST"])
        if dst_raw.startswith("[") and dst_raw.endswith("]"):
            dst_raw = dst_raw[1:-1]
        return cls(
            dst=parse_mem_addr_field(dst_raw, "DST"),
            src=parse_dmem_src_keyword(str(d["SRC"])),
            wr=parse_optional_side_write(d, "WR", "DMEM_WR.WR"),
            op=require_alu_expr(d["OP"], "DMEM_WR.OP") if "OP" in d else None,
            lit=parse_optional_immediate(d, "LIT", "DMEM_WR.LIT"),
            uf="UF" in d,
            if_cond=parse_cond_code(d.get("IF")),
        )

    @property
    def reg_read(self) -> frozenset[str]:
        reads: frozenset[str] = frozenset()
        if isinstance(self.dst, Register):
            reads = reads | self.dst.regs()
        if self.op:
            reads = reads | self.op.regs()
        return reads

    @property
    def reg_write(self) -> frozenset[str]:
        return self.wr.regs() if self.wr else frozenset()

    def to_dict(self) -> dict[str, Any]:
        dst_str = str(self.dst)
        if isinstance(self.dst, MemAddr):
            dst_str = f"[{dst_str}]"
        d = {
            "CMD": "DMEM_WR",
            "DST": dst_str,
            "SRC": self.src,
            "OP": str(self.op) if self.op else None,
            "LIT": str(self.lit) if self.lit else None,
            "WR": str(self.wr) if self.wr else None,
            "UF": "1" if self.uf else None,
            "IF": self.if_cond if self.if_cond else None,
        }
        return {k: v for k, v in d.items() if v is not None}


@dataclass(frozen=True)
class WmemWriteInst(BaseInst):
    """WMEM_WR instruction: write wave registers into wave memory."""

    addr: Register | MemAddr | None = None
    time: TimeType | None = None
    wr: SideWrite | None = None
    op: ExprType | None = None
    uf: bool = False
    if_cond: CondCode | None = None
    wp: str | None = None
    port: str | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> WmemWriteInst:
        wp, port = parse_wp_port(d, "WMEM_WR")
        return cls(
            addr=parse_mem_addr_field(str(d["DST"]), "DST") if "DST" in d else None,
            time=parse_optional_time(d, "TIME", "WMEM_WR.TIME"),
            wr=parse_optional_side_write(d, "WR", "WMEM_WR.WR"),
            op=require_alu_expr(d["OP"], "WMEM_WR.OP") if "OP" in d else None,
            uf="UF" in d,
            if_cond=parse_cond_code(d.get("IF")),
            wp=wp,
            port=port,
        )

    @property
    def reg_read(self) -> frozenset[str]:
        reads = Register("r_wave").regs() | frozenset({TIMED_BASE_REG})
        if isinstance(self.addr, Register):
            reads = reads | self.addr.regs()
        if isinstance(self.time, Register):
            reads = reads | self.time.regs()
        if self.op:
            reads = reads | self.op.regs()
        return reads

    @property
    def reg_write(self) -> frozenset[str]:
        return self.wr.regs() if self.wr else frozenset()

    def to_dict(self) -> dict[str, Any]:
        d = {
            "CMD": "WMEM_WR",
            "DST": str(self.addr) if self.addr else None,
            "TIME": str(self.time) if self.time else None,
            "WR": str(self.wr) if self.wr else None,
            "OP": str(self.op) if self.op else None,
            "UF": "1" if self.uf else None,
            "IF": self.if_cond if self.if_cond else None,
            "WP": self.wp,
            "PORT": self.port,
        }
        return {k: v for k, v in d.items() if v is not None}


@dataclass(frozen=True)
class DportWriteInst(BaseInst):
    """DPORT_WR instruction: write to data port."""

    dst: Register | ImmValue
    src: DportSrc = "imm"
    data: Register | ImmValue = ImmValue(0)
    time: TimeType | None = None
    wr: SideWrite | None = None
    op: ExprType | None = None
    uf: bool = False
    if_cond: CondCode | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DportWriteInst:
        src = parse_dport_src_keyword(d.get("SRC"))
        return cls(
            dst=parse_port_dst(str(d["DST"])),
            src=src,
            data=parse_dport_data_field(d.get("DATA"), src),
            time=parse_optional_time(d, "TIME", "DPORT_WR.TIME"),
            wr=parse_optional_side_write(d, "WR", "DPORT_WR.WR"),
            op=require_alu_expr(d["OP"], "DPORT_WR.OP") if "OP" in d else None,
            uf="UF" in d,
            if_cond=parse_cond_code(d.get("IF")),
        )

    @property
    def reg_read(self) -> frozenset[str]:
        reads: frozenset[str] = frozenset({TIMED_BASE_REG})
        if isinstance(self.src, Register):
            reads = reads | self.src.regs()
        for op in [self.dst, self.data, self.time]:
            if isinstance(op, Register):
                reads = reads | op.regs()
        if self.op:
            reads = reads | self.op.regs()
        return reads

    @property
    def reg_write(self) -> frozenset[str]:
        return self.wr.regs() if self.wr else frozenset()

    def to_dict(self) -> dict[str, Any]:
        d = {
            "CMD": "DPORT_WR",
            "DST": str(self.dst) if self.dst else None,
            "SRC": self.src,
            "DATA": str(self.data),
            "TIME": str(self.time) if self.time else None,
            "WR": str(self.wr) if self.wr else None,
            "OP": str(self.op) if self.op else None,
            "UF": "1" if self.uf else None,
            "IF": self.if_cond if self.if_cond else None,
        }
        return {k: v for k, v in d.items() if v is not None}


@dataclass(frozen=True)
class DportReadInst(BaseInst):
    """DPORT_RD instruction: read I/Q result from a data port."""

    dst: Register | ImmValue

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DportReadInst:
        return cls(dst=parse_port_dst(str(d["DST"])))

    @property
    def reg_read(self) -> frozenset[str]:
        reads: frozenset[str] = frozenset({STATUS_REG})
        if isinstance(self.dst, Register):
            reads = reads | self.dst.regs()
        return reads

    @property
    def reg_write(self) -> frozenset[str]:
        return frozenset({"s8", "s9"})

    def to_dict(self) -> dict[str, Any]:
        d = {
            "CMD": "DPORT_RD",
            "DST": str(self.dst) if self.dst else None,
        }
        return {k: v for k, v in d.items() if v is not None}


@dataclass(frozen=True)
class TrigInst(BaseInst):
    """TRIG instruction: set or clear a trigger port."""

    dst: Register | ImmValue
    src: TrigSrc
    time: TimeType | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrigInst:
        return cls(
            dst=parse_port_dst(str(d["DST"])),
            src=require_literal(str(d["SRC"]), "TRIG.SRC", VALID_TRIG_SRCS),
            time=parse_optional_time(d, "TIME", "TRIG.TIME"),
        )

    @property
    def reg_read(self) -> frozenset[str]:
        reads: frozenset[str] = frozenset({TIMED_BASE_REG})
        if isinstance(self.dst, Register):
            reads = reads | self.dst.regs()
        if isinstance(self.time, Register):
            reads = reads | self.time.regs()
        return reads

    def to_dict(self) -> dict[str, Any]:
        d = {
            "CMD": "TRIG",
            "DST": str(self.dst) if self.dst else None,
            "SRC": self.src,
            "TIME": str(self.time) if self.time else None,
        }
        return {k: v for k, v in d.items() if v is not None}
