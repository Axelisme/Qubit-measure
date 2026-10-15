"""Explicit FTF environment and author-only numeric composition.

Import ftf to build the shared GUI/Notebook bundle. plan_point exposes numeric
inputs for independent author diagnostics, not normal GUI/MCP calibration truth.
"""

from .database import FTFBuildPlan, build_database
from .point import FTFPointPlan, plan_point
from .runtime import FTFSource, ftf
