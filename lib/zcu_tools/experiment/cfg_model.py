from __future__ import annotations

from zcu_tools.cfg_model import ConfigBase


class ExpCfgModel(ConfigBase):
    """Framework base for experiment-owned inputs.

    Concrete experiment cfgs declare their device role-name mappings and units
    when needed. This base carries neither device selections nor settings.
    Drivers are resolved by the experiment through RunContext.devices at use.
    """


class ProgramExpCfg(ExpCfgModel):
    """Base for zcu_lab experiment inputs that also configure a v2 program.

    Inherits reps, rounds and initial/relax delay (us), with the same defaults
    and validation as ProgramV2Cfg. Declare experiment-owned fields in a subclass.
    Use ExpCfgModel instead when program settings are constructed only at run time.
    """
