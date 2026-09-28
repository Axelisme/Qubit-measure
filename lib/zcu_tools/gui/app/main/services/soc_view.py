"""Project GUI-owned SoC connection metadata and QICK hardware information."""

from __future__ import annotations

import json
from collections.abc import Mapping

from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.session.types import SocCfgHandle
from zcu_tools.program import describe_soc


def project_soc_info(
    soccfg: SocCfgHandle | None,
    *,
    is_mock: bool,
    endpoint: Mapping[str, str | int | None],
    include_cfg: bool,
) -> dict[str, object]:
    """Summarize the connected board; serialize full QICK cfg only on request."""
    if soccfg is None:
        raise FailedPreconditionError("No SoC connected")
    info: dict[str, object] = {
        "description": describe_soc(soccfg),
        "is_mock": is_mock,
        **endpoint,
    }
    if include_cfg:
        info["cfg"] = json.loads(soccfg.dump_cfg())
    return info
