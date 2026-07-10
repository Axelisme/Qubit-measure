"""Fresh CfgSectionSpec factories for pulse modules."""

from __future__ import annotations

from zcu_tools.gui.app.main.adapter import (
    CfgSectionSpec,
    LiteralSpec,
    ReferenceSpec,
    ScalarSpec,
)

from .waveform import (
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
            "waveform": ReferenceSpec(
                kind="waveform",
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
            # nqz / phase are rarely tuned per-experiment → Advanced group (still
            # required/normal fields, only the rendering is grouped).
            "nqz": ScalarSpec(label="NQZ", type=int, choices=[1, 2], group="Advanced"),
            "freq": ScalarSpec(label="Freq (MHz)", type=float, decimals=2),
            "gain": ScalarSpec(label="Gain", type=float, decimals=4),
            "phase": ScalarSpec(
                label="Phase (deg)", type=float, decimals=2, group="Advanced"
            ),
            "pre_delay": ScalarSpec(
                label="Pre-delay (us)", type=float, decimals=3, group="Advanced"
            ),
            "post_delay": ScalarSpec(
                label="Post-delay (us)", type=float, decimals=3, group="Advanced"
            ),
            # Optional NCO mixer frequency (PulseCfg.mixer_freq) — only relevant
            # for generator channels that have a mixer; left empty (None) → no
            # mixer. Tucked under an "Advanced" group to keep the common form
            # uncluttered.
            "mixer_freq": ScalarSpec(
                label="Mixer freq (MHz)",
                type=float,
                decimals=2,
                optional=True,
                group="Advanced",
            ),
        },
    )
