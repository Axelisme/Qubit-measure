"""Shared program and host-loop configuration used by runtime seam tests."""

from zcu_tools.cfg_model import ConfigBase


class FlowCfg(ConfigBase):
    """Synthetic host-loop inputs; program controls are passed explicitly."""

    reps: int = 1
    rounds: int = 1
    initial_delay: float = 1.0
    relax_delay: float = 1.0
    value: float = 0.0
    marker: str = ""


def make_flow_cfg(**values: int | str) -> FlowCfg:
    """Create FlowCfg and assign named program or extra int/string fields.

    Assignment validation errors from the model propagate unchanged.
    """
    cfg = FlowCfg()
    for name, value in values.items():
        setattr(cfg, name, value)
    return cfg
