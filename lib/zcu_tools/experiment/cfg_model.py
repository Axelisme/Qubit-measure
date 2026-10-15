from __future__ import annotations

from zcu_tools.cfg_model import ConfigBase


class ExpCfgModel(ConfigBase):
    """Framework base for experiment-owned inputs.

    Concrete experiment cfgs declare their device role-name mappings and units
    when needed. This base carries neither device selections nor settings.
    Drivers are resolved by the experiment through RunContext.devices at use.
    """
