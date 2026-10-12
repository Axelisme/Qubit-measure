"""Private mechanics shared by multiple measure experiment adapters."""

from zcu_lab.v2._support.measure.analyze_results import (
    FigureOnlyAnalyzeResult,
    run_figure_only_analyze,
)
from zcu_lab.v2._support.measure.defaults.helpers import make_trig_offset
from zcu_lab.v2._support.measure.defaults.role_factories import (
    ROLE_FACTORIES,
    RoleFactorySpec,
)
from zcu_lab.v2._support.measure.seeds import (
    flux_range,
    qub_freq_range,
    res_freq_range,
)
from zcu_lab.v2._support.measure.writeback_helpers import (
    READOUT_DPM_PULSE_TAIL_US,
    pulse_readout_module_writeback_items,
    readout_dpm_writeback_items,
)

__all__ = [
    # context-free schema definition
    "flux_range",
    "qub_freq_range",
    "res_freq_range",
    # shared analyze-result shapes
    "FigureOnlyAnalyzeResult",
    "run_figure_only_analyze",
    # interactive flux-pick analysis (shared by onetone/twotone flux_dep)
    # Role factory table (single source for TemplateCatalog + MeasureCfgDefinition)
    "ROLE_FACTORIES",
    "RoleFactorySpec",
    # Module defaults (low-level)
    "make_trig_offset",
    # Spec helpers
    # Shared readout module writeback helpers
    "READOUT_DPM_PULSE_TAIL_US",
    "pulse_readout_module_writeback_items",
    "readout_dpm_writeback_items",
]
