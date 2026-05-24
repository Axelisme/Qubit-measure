"""Fresh CfgSectionSpec factories for pulse modules."""

from __future__ import annotations

from zcu_tools.gui.adapter import (
    CfgSectionSpec,
    LiteralSpec,
    ScalarSpec,
    WaveformRefSpec,
)
from zcu_tools.gui.specs.waveform import (
    make_arb_waveform_spec,
    make_const_waveform_spec,
    make_cosine_waveform_spec,
    make_drag_waveform_spec,
    make_flat_top_waveform_spec,
    make_gauss_waveform_spec,
)


def make_pulse_spec(label: str = "Pulse") -> CfgSectionSpec:
    return CfgSectionSpec(
        label=label,
        fields={
            "type": LiteralSpec("pulse"),
            "waveform": WaveformRefSpec(
                allowed=[
                    make_const_waveform_spec(),
                    make_cosine_waveform_spec(),
                    make_gauss_waveform_spec(),
                    make_drag_waveform_spec(),
                    make_arb_waveform_spec(),
                    make_flat_top_waveform_spec(),
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
