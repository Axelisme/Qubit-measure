from __future__ import annotations

from pydantic import ConfigDict
from zcu_tools.experiment.cfg_model import ExpCfgModel


class DictCfg(ExpCfgModel):
    """Flexible cfg for unit tests — allows arbitrary extra fields."""

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)
