from __future__ import annotations

from typing import Any, Optional, TypeVar

from zcu_tools.experiment.base import AbsExperiment
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.gui.adapter import RunRequest, SocCfgHandle, SocHandle

T_Result = TypeVar("T_Result")
T_Config = TypeVar("T_Config", bound=ExpCfgModel)


def require_soc_handles(req: RunRequest) -> tuple[SocHandle, SocCfgHandle]:
    if req.soc is None:
        raise RuntimeError("RunRequest.soc is required for real experiment adapters")
    if req.soccfg is None:
        raise RuntimeError("RunRequest.soccfg is required for real experiment adapters")
    return req.soc, req.soccfg


def save_with_last_state(
    *,
    exp_cls: type[AbsExperiment[T_Result, T_Config]],
    cfg: T_Config,
    result: T_Result,
    filepath: str,
    comment: Optional[str] = None,
    **kwargs: Any,
) -> None:
    exp = exp_cls()
    exp.last_cfg = cfg
    exp.last_result = result
    exp.save(filepath=filepath, result=result, comment=comment, **kwargs)  # type: ignore[attr-defined]
