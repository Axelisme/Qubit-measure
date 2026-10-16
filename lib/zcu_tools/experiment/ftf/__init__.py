"""Explicit FTF environment and author-only numeric composition.

Import ftf to build the shared GUI/Notebook bundle. plan_template exposes numeric
inputs for independent author diagnostics, not normal GUI/MCP calibration truth.
"""

from .replay import replay_round
from .runtime import FTFAcquire, FTFNumericUse, FTFSource, ftf
from .template import FTFTemplatePlan, plan_template
