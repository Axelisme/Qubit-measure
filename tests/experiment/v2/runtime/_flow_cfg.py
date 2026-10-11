"""Shared program and host-loop configuration used by runtime seam tests."""

from pydantic import ConfigDict
from zcu_tools.experiment import ExpCfgModel
from zcu_tools.program.v2 import ProgramV2Cfg


class FlowCfg(ProgramV2Cfg, ExpCfgModel):
    """Program cfg allowing extra attributes for host-loop assertions."""

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


def make_flow_cfg(**values: int | str) -> FlowCfg:
    """Create FlowCfg and assign named program or extra int/string fields.

    Assignment validation errors from the model propagate unchanged.
    """
    cfg = FlowCfg()
    for name, value in values.items():
        setattr(cfg, name, value)
    return cfg
