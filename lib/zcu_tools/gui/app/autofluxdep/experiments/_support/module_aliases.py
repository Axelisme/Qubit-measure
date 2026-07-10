"""ModuleLibrary fallback names shared by autofluxdep nodes.

Workflow-produced modules keep short logical names such as ``readout`` and
``pi_pulse``. Persisted ModuleLibrary presets often use calibrated measure-gui
names, so ModuleDep declarations use these lists only for library fallback.
"""

from __future__ import annotations

READOUT_LIBRARY_ALIASES = ("readout_dpm", "readout_rf", "readout", "res_readout")
PI_PULSE_LIBRARY_ALIASES = ("pi_len", "pi_amp", "pi_pulse")
PI2_PULSE_LIBRARY_ALIASES = ("pi2_amp", "pi2_len", "pi_amp", "pi_len", "pi2_pulse")
