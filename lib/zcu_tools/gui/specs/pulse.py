"""Static CfgSectionSpec definitions for pulse module."""

from __future__ import annotations

from zcu_tools.gui.adapter import CfgSectionSpec, ScalarSpec, WaveformRefSpec
from zcu_tools.gui.specs.waveform import (
    ARB_WAVEFORM_SPEC,
    CONST_WAVEFORM_SPEC,
    COSINE_WAVEFORM_SPEC,
    DRAG_WAVEFORM_SPEC,
    FLAT_TOP_WAVEFORM_SPEC,
    GAUSS_WAVEFORM_SPEC,
)

PULSE_SPEC = CfgSectionSpec(
    label="Pulse",
    fields={
        "type": ScalarSpec(label="Type", type=str, hidden=True),
        "waveform": WaveformRefSpec(
            allowed=[
                CONST_WAVEFORM_SPEC,
                COSINE_WAVEFORM_SPEC,
                GAUSS_WAVEFORM_SPEC,
                DRAG_WAVEFORM_SPEC,
                ARB_WAVEFORM_SPEC,
                FLAT_TOP_WAVEFORM_SPEC,
            ],
            label="Waveform",
        ),
        "ch": ScalarSpec(label="Gen ch", type=int),
        "nqz": ScalarSpec(label="NQZ", type=int, choices=[1, 2]),
        "freq": ScalarSpec(label="Freq (MHz)", type=float, decimals=2),
        "phase": ScalarSpec(label="Phase (deg)", type=float, decimals=2),
        "gain": ScalarSpec(label="Gain", type=float, decimals=4),
        "pre_delay": ScalarSpec(label="Pre-delay (us)", type=float, decimals=3),
        "post_delay": ScalarSpec(label="Post-delay (us)", type=float, decimals=3),
    },
)
