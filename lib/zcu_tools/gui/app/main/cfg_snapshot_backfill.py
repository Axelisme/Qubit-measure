from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from zcu_tools.gui.app.main.adapter import CfgSchema

if TYPE_CHECKING:
    from zcu_tools.meta_tool import ModuleLibrary
    from zcu_tools.meta_tool.metadict import MetaDict


@dataclass(frozen=True)
class CfgSnapshotBackfillRequest:
    adapter_name: str
    base_schema: CfgSchema
    cfg_snapshot: object
    md: MetaDict | None = None
    ml: ModuleLibrary | None = None


@dataclass(frozen=True)
class CfgSnapshotBackfillResult:
    schema: CfgSchema
    warnings: tuple[str, ...] = ()
    lost_refs: bool = False


def cfg_snapshot_to_schema(
    request: CfgSnapshotBackfillRequest,
) -> CfgSnapshotBackfillResult:
    del request
    raise NotImplementedError(
        "cfg_snapshot backfill is not implemented in the first load-data release"
    )
