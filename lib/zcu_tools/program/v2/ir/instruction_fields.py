"""Finite QICK instruction field values, decoding and label-reference remapping."""

from __future__ import annotations

from typing import Any, Literal, TypeGuard

from .labels import Label, LabelRef
from .operands import (
    AddrType,
    AluExpr,
    Immediate,
    ImmValue,
    MemAddr,
    Register,
    SideWrite,
    SrcType,
    TimeOffset,
    TimeType,
    parse_addr,
    parse_alu_expr,
    parse_imm_value,
    parse_immediate,
    parse_mem_addr,
    parse_register,
    parse_side_write,
    parse_src,
    parse_time,
)

# QICK IF condition codes.
CondCode = Literal["Z", "S", "NZ", "NS"]
_VALID_COND_CODES: frozenset[CondCode] = frozenset({"Z", "S", "NZ", "NS"})

# TIME.C_OP timing-reference operations.
TimeCOp = Literal["inc_ref", "set_ref", "updt", "rst"]
VALID_TIME_COPS: frozenset[TimeCOp] = frozenset({"inc_ref", "set_ref", "updt", "rst"})

# FLAG.C_OP flag operations.
FlagCOp = Literal["set", "clr", "inv"]
VALID_FLAG_COPS: frozenset[FlagCOp] = frozenset({"set", "clr", "inv"})

# WAIT.C_OP completion conditions.
WaitCOp = Literal["time", "port_dt", "div_rdy", "div_dt", "qpa_rdy", "qpa_dt"]
VALID_WAIT_COPS: frozenset[WaitCOp] = frozenset(
    {"time", "port_dt", "div_rdy", "div_dt", "qpa_rdy", "qpa_dt"}
)

# CLEAR.C_OP peripheral flag groups.
ClearCOp = Literal["arith", "div", "qnet", "qcom", "qpa", "qpb", "port", "all"]
VALID_CLEAR_COPS: frozenset[ClearCOp] = frozenset(
    {"arith", "div", "qnet", "qcom", "qpa", "qpb", "port", "all"}
)

# NET.C_OP network operations.
NetCOp = Literal[
    "set_net", "sync_net", "updt_offset", "set_dt", "get_dt", "set_flag", "get_flag"
]
VALID_NET_COPS: frozenset[NetCOp] = frozenset(
    {"set_net", "sync_net", "updt_offset", "set_dt", "get_dt", "set_flag", "get_flag"}
)

# COM.C_OP communication operations.
ComCOp = Literal[
    "set_flag",
    "sync",
    "reset",
    "set_byte_1",
    "set_byte_2",
    "set_hw_1",
    "set_hw_2",
    "set_word_1",
    "set_word_2",
]
VALID_COM_COPS: frozenset[ComCOp] = frozenset(
    {
        "set_flag",
        "sync",
        "reset",
        "set_byte_1",
        "set_byte_2",
        "set_hw_1",
        "set_hw_2",
        "set_word_1",
        "set_word_2",
    }
)

# ARITH.C_OP multiply-accumulate operations.
ArithCOp = Literal["T", "TP", "TM", "PT", "PTP", "PTM", "MT", "MTP", "MTM"]
VALID_ARITH_COPS: frozenset[ArithCOp] = frozenset(
    {"T", "TP", "TM", "PT", "PTP", "PTM", "MT", "MTP", "MTM"}
)

# TRIG.SRC trigger level operations.
TrigSrc = Literal["set", "clr"]
VALID_TRIG_SRCS: frozenset[TrigSrc] = frozenset({"set", "clr"})

# CMD selectors for custom peripheral A or B.
PACOp = Literal["PA", "PB"]
VALID_PA_CMDS: frozenset[PACOp] = frozenset({"PA", "PB"})

# COM.R1 literal flag values for set_flag.
ComFlagVal = Literal["0", "1"]
VALID_COM_FLAG_VALS: frozenset[ComFlagVal] = frozenset({"0", "1"})


def is_valid_literal[T: str](val: str, valid: frozenset[T]) -> TypeGuard[T]:
    """Return whether val belongs to valid, narrowing it to the allowed literal type."""
    return val in valid


def require_literal[T: str](val: str, field: str, valid: frozenset[T]) -> T:
    """Return val from valid, or raise ValueError naming field and the allowed values."""
    if not is_valid_literal(val, valid):
        raise ValueError(
            f"{field}: {val!r} is not valid, expected one of {sorted(valid)}"
        )
    return val


def parse_cond_code(val: str | None) -> CondCode | None:
    """Return a QICK IF code, preserve None, or raise ValueError for an unknown code."""
    if val is None:
        return None
    if not is_valid_literal(val, _VALID_COND_CODES):
        raise ValueError(
            f"IF: {val!r} is not a valid condition code, expected one of {sorted(_VALID_COND_CODES)}"
        )
    return val


def require_register(val: str, field: str) -> Register:
    """Parse val as a register, or raise ValueError identifying the QICK field."""
    reg = parse_register(val)
    if reg is None:
        raise ValueError(f"{field}: {val!r} is not a valid register name")
    return reg


def parse_optional_register(d: dict[str, Any], key: str, field: str) -> Register | None:
    """Parse d[key] as a register, or return None when key is absent.

    Invalid values raise ValueError identifying the QICK field.
    """
    if key not in d:
        return None
    reg = parse_register(d[key])
    if reg is None:
        raise ValueError(f"{field}: {d[key]!r} is not a valid register name")
    return reg


def _require_immediate(val: Any, field: str) -> Immediate:
    imm = parse_immediate(val)
    if imm is None:
        raise ValueError(f"{field}: {val!r} is not a valid immediate")
    return imm


def parse_optional_immediate(
    d: dict[str, Any], key: str, field: str
) -> Immediate | None:
    """Parse d[key] as an immediate, or return None when key is absent.

    Invalid values raise ValueError identifying the QICK field.
    """
    if key not in d:
        return None
    return _require_immediate(d[key], field)


def require_alu_expr(val: str, field: str) -> AluExpr:
    """Parse val as an ALU expression, or raise ValueError identifying the QICK field."""
    expr = parse_alu_expr(val)
    if expr is None:
        raise ValueError(f"{field}: {val!r} is not a valid ALU expression")
    return expr


def parse_port_dst(val: str) -> Register | ImmValue:
    """Parse a QICK DST register or bare integer port number; invalid values raise ValueError."""
    reg = parse_register(val)
    if reg is not None:
        return reg
    imm = parse_imm_value(val)
    if imm is not None:
        return imm
    raise ValueError(f"DST: {val!r} is not a register or port number")


def parse_mem_addr_field(val: str, field: str) -> Register | MemAddr:
    """Parse a register or &N numeric address; invalid values raise ValueError naming field."""
    reg = parse_register(val)
    if reg is not None:
        return reg
    addr = parse_mem_addr(val)
    if addr is not None:
        return addr
    raise ValueError(f"{field}: {val!r} is not a register or memory address")


def parse_optional_addr(d: dict[str, Any], key: str, field: str) -> AddrType | None:
    """Parse d[key] as an address, or return None when key is absent.

    Invalid values raise ValueError identifying the QICK field.
    """
    if key not in d:
        return None
    addr = parse_addr(d[key])
    if addr is None:
        raise ValueError(f"{field}: {d[key]!r} is not a valid address")
    return addr


def parse_optional_time(d: dict[str, Any], key: str, field: str) -> TimeType | None:
    """Parse d[key] as a time operand, or return None when key is absent.

    Invalid operands or @ literals raise ValueError identifying field.
    """
    if key not in d:
        return None
    raw = d[key]
    time = parse_time(raw)
    if time is None:
        raise ValueError(f"{field}: {raw!r} is not a valid time operand")
    if (
        isinstance(raw, str)
        and raw.strip().startswith("@")
        and not isinstance(time, TimeOffset)
    ):
        raise ValueError(f"{field}: {raw!r} is not a valid time literal")
    return time


def parse_optional_src(d: dict[str, Any], key: str, field: str) -> SrcType | None:
    """Parse d[key] as a source operand, or return None when key is absent.

    Invalid values raise ValueError identifying the QICK field.
    """
    if key not in d:
        return None
    src = parse_src(d[key])
    if src is None:
        raise ValueError(f"{field}: {d[key]!r} is not a valid source")
    return src


def parse_optional_side_write(
    d: dict[str, Any], key: str, field: str
) -> SideWrite | None:
    """Parse d[key] as a side-write, or return None when key is absent.

    Invalid values raise ValueError identifying the QICK field.
    """
    if key not in d:
        return None
    wr = parse_side_write(d[key])
    if wr is None:
        raise ValueError(f"{field}: {d[key]!r} is not a valid side-write")
    return wr


def parse_wp_port(d: dict[str, Any], cmd: str) -> tuple[str | None, str | None]:
    """Return optional WP/PORT strings from the QICK dictionary d.

    Unpaired fields, packed WP or a noninteger port raise ValueError
    identifying the instruction cmd.
    """
    wp_raw = d.get("WP")
    port_raw = d.get("PORT")
    wp = None if wp_raw is None else str(wp_raw)
    port = None if port_raw is None else str(port_raw)

    if wp is not None and len(wp.split()) != 1:
        raise ValueError(f"{cmd}.WP: packed WP/PORT form is not supported")
    if wp is None and port is not None:
        raise ValueError(f"{cmd}.WP: PORT requires an explicit WP field")
    if wp is not None and port is None:
        raise ValueError(f"{cmd}.PORT: WP requires an explicit PORT field")
    if port is not None and parse_imm_value(port) is None:
        raise ValueError(f"{cmd}.PORT: {port!r} is not a valid port number")
    return wp, port


# DPORT_WR.SRC selects immediate or register DATA.
DportSrc = Literal["imm", "reg"]


def parse_dport_src_keyword(val: Any) -> DportSrc:
    """Return the DPORT_WR SRC keyword imm/reg, or raise ValueError if absent or invalid."""
    if val is None:
        raise ValueError("DPORT_WR.SRC is required")
    raw = str(val)
    if raw not in ("imm", "reg"):
        raise ValueError(f"DPORT_WR.SRC must be 'imm' or 'reg', got {raw!r}")
    return raw


def parse_dport_data_field(val: Any, src: DportSrc) -> Register | ImmValue:
    """Parse DPORT_WR DATA as an integer for imm or a register for reg.

    Absent or invalid values raise ValueError.
    """
    if val is None:
        raise ValueError("DPORT_WR.DATA is required")
    if src == "imm":
        data = parse_imm_value(val)
        if data is not None:
            return data
        raise ValueError(f"DPORT_WR.DATA: {val!r} is not a bare integer for SRC='imm'")
    reg = parse_register(val)
    if reg is not None:
        return reg
    raise ValueError(f"DPORT_WR.DATA: {val!r} is not a register for SRC='reg'")


# DMEM_WR.SRC selects literal or ALU-operation data.
DmemSrc = Literal["imm", "op"]


def parse_dmem_src_keyword(val: str) -> DmemSrc:
    """Return the DMEM_WR SRC keyword imm/op, or raise ValueError for an invalid value."""
    if val not in ("imm", "op"):
        raise ValueError(f"DMEM_WR.SRC must be 'imm' or 'op', got {val!r}")
    return val


def remap_label_ref(ref: LabelRef | None, remap: dict[str, Label]) -> LabelRef | None:
    """Remap ref using remap keys containing label names.

    Preserve None, pseudo-label identity and references absent from remap.
    """
    if ref is None or ref.is_pseudo():
        return ref
    new = remap.get(str(ref.as_label()))
    return LabelRef(new) if new is not None else ref
