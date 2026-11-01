"""Explicit FTF environment and author-only numeric composition.

Import ftf to build the shared GUI/Notebook bundle. plan_template exposes numeric
inputs for independent author diagnostics, not normal GUI/MCP calibration truth.
"""

from zcu_tools.simulators.ftf.environment.acquisition import FTFAcquisitionOptions
from zcu_tools.simulators.ftf.environment.readout import (
    FTFMuxReadoutConfig,
    FTFReadoutConfig,
    FTFReceiverConfig,
)
from zcu_tools.simulators.ftf.environment.replay import replay_round
from zcu_tools.simulators.ftf.environment.runtime import (
    FTFAcquire,
    FTFNumericUse,
    FTFSource,
    ftf,
)
from zcu_tools.simulators.ftf.environment.template import (
    FTFModelFacts,
    FTFReadoutPlan,
    FTFTemplatePlan,
    plan_template,
)
