"""Static dmem tables that LoadValue / LoadWord fill, read per sweep point."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from zcu_tools.program.v2.modules.base import Module
from zcu_tools.program.v2.modules.dmem import LoadValue, LoadWord

from .errors import UnsupportedModuleError


@dataclass(frozen=True)
class DmemTable:
    values: list[int]
    idx_reg: str
    raw_word: bool
    source_name: str


def collect_dmem_values(
    modules: Sequence[Module],
) -> dict[str, DmemTable]:
    """Scan the module list for dmem tables, keyed by their ``val_reg``.

    Also records the table's ``idx_reg`` (the sweep axis name) so a later
    register-driven DelayAuto can map its register back to the correct sweep
    axis.  ``auto_compress`` tables would change the stored layout; the T1 path
    uses ``auto_compress=False`` so values are stored verbatim, which is the only
    scalar layout this lowering supports — anything compressed fast-fails.
    """

    tables: dict[str, DmemTable] = {}
    for module in modules:
        if isinstance(module, LoadValue):
            if module.is_compressed:
                raise UnsupportedModuleError(
                    f"LoadValue {module.name!r} is compressed; the Bloch lowering "
                    f"only supports verbatim (auto_compress=False) dmem tables"
                )
            _add_dmem_table(
                tables,
                module.val_reg,
                DmemTable(
                    values=list(module.values),
                    idx_reg=module.idx_reg,
                    raw_word=False,
                    source_name=module.name,
                ),
            )
        elif isinstance(module, LoadWord):
            _add_dmem_table(
                tables,
                module.val_reg,
                DmemTable(
                    values=list(module.values),
                    idx_reg=module.idx_reg,
                    raw_word=True,
                    source_name=module.name,
                ),
            )
    return tables


def _add_dmem_table(
    tables: dict[str, DmemTable], val_reg: str, table: DmemTable
) -> None:
    if val_reg in tables:
        raise UnsupportedModuleError(
            f"dmem register {val_reg!r} is populated by multiple LoadValue/LoadWord "
            "modules; SimEngine requires one producer per register"
        )
    tables[val_reg] = table


def dmem_table_value(val_reg: str, table: DmemTable, point: dict[str, int]) -> int:
    if table.idx_reg not in point:
        raise UnsupportedModuleError(
            f"cannot determine which sweep axis indexes dmem register {val_reg!r}"
        )

    idx = point[table.idx_reg]
    if not 0 <= idx < len(table.values):
        raise UnsupportedModuleError(
            f"dmem index {idx} out of range for register {val_reg!r} "
            f"(table size {len(table.values)})"
        )

    return table.values[idx]


def load_value(
    val_reg: str,
    tables: dict[str, DmemTable],
    point: dict[str, int],
    *,
    consumer: str,
) -> int:
    """Read the integer a LoadValue table holds for ``val_reg`` at one point."""

    if val_reg not in tables:
        raise UnsupportedModuleError(
            f"{consumer} reads register {val_reg!r} but no LoadValue populates it"
        )
    table = tables[val_reg]
    if table.raw_word:
        raise UnsupportedModuleError(
            f"{consumer} reads register {val_reg!r} from LoadWord "
            f"{table.source_name!r}; raw hardware register words are not integers"
        )
    return dmem_table_value(val_reg, table, point)
