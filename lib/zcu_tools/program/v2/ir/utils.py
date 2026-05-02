import re
from typing import Any

_REG_TOKEN_RE = re.compile(r"\b[srw]\d+\b|s_[A-Za-z0-9_]+|temp_reg_\d+")


def regs_from_value(value: Any) -> set[str]:
    if not isinstance(value, str):
        return set()
    if value.startswith("#"):
        return set()
    return set(_REG_TOKEN_RE.findall(value))


def strip_write_modifier(value: str) -> str:
    return value.split(" ", 1)[0]
