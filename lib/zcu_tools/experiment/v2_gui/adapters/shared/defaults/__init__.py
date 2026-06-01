"""Per-role default factories.

Each role lives in its own module and exposes a blank factory
(``make_<role>_default``) and a library-aware ref factory
(``make_<role>_ref_default`` — prefers a named library entry, falls back to the
blank one). This package re-exports them flat. See ADR-0009.
"""

from .helpers import (
    make_default_value,
    make_trig_offset,
    patch_pulse_fields,
    patch_ro_cfg_fields,
    select_named_module_value,
)
from .module_defaults import NamedModuleValue
from .pi2_pulse import make_pi2_pulse_default, make_pi2_pulse_ref_default
from .pi_pulse import make_pi_pulse_default, make_pi_pulse_ref_default
from .qub_probe import make_qub_probe_default, make_qub_probe_ref_default
from .qub_waveform import make_qub_waveform_default, make_qub_waveform_ref_default
from .readout import (
    make_direct_readout_default,
    make_pulse_readout_default,
    make_readout_default,
    make_readout_dpm_default,
    make_readout_ref_default,
)
from .res_probe import make_res_probe_default, make_res_probe_ref_default
from .res_waveform import make_res_waveform_default, make_res_waveform_ref_default
from .reset import (
    make_bath_reset_default,
    make_none_reset_default,
    make_pulse_reset_default,
    make_reset_default,
    make_reset_ref_default,
    make_two_pulse_reset_default,
)

__all__ = [
    # shared helpers
    "make_default_value",
    "make_trig_offset",
    "patch_pulse_fields",
    "patch_ro_cfg_fields",
    "select_named_module_value",
    "NamedModuleValue",
    # qub_probe
    "make_qub_probe_default",
    "make_qub_probe_ref_default",
    # res_probe
    "make_res_probe_default",
    "make_res_probe_ref_default",
    # pi_pulse
    "make_pi_pulse_default",
    "make_pi_pulse_ref_default",
    # pi2_pulse
    "make_pi2_pulse_default",
    "make_pi2_pulse_ref_default",
    # readout
    "make_pulse_readout_default",
    "make_direct_readout_default",
    "make_readout_default",
    "make_readout_dpm_default",
    "make_readout_ref_default",
    # reset
    "make_none_reset_default",
    "make_pulse_reset_default",
    "make_two_pulse_reset_default",
    "make_bath_reset_default",
    "make_reset_default",
    "make_reset_ref_default",
    # qub_waveform
    "make_qub_waveform_default",
    "make_qub_waveform_ref_default",
    # res_waveform
    "make_res_waveform_default",
    "make_res_waveform_ref_default",
]
