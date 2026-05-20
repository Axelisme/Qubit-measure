"""Static CfgSectionSpec definitions for waveform types."""

from __future__ import annotations

from zcu_tools.gui.adapter import CfgSectionSpec, ScalarSpec, WaveformRefSpec

CONST_WAVEFORM_SPEC = CfgSectionSpec(
    label="Const",
    fields={
        "style": ScalarSpec(label="Style", type=str, hidden=True),
        "length": ScalarSpec(label="Length (us)", type=float, decimals=3),
    },
)

COSINE_WAVEFORM_SPEC = CfgSectionSpec(
    label="Cosine",
    fields={
        "style": ScalarSpec(label="Style", type=str, hidden=True),
        "length": ScalarSpec(label="Length (us)", type=float, decimals=3),
    },
)

GAUSS_WAVEFORM_SPEC = CfgSectionSpec(
    label="Gauss",
    fields={
        "style": ScalarSpec(label="Style", type=str, hidden=True),
        "length": ScalarSpec(label="Length (us)", type=float, decimals=3),
        "sigma": ScalarSpec(label="Sigma (us)", type=float, decimals=3),
    },
)

DRAG_WAVEFORM_SPEC = CfgSectionSpec(
    label="DRAG",
    fields={
        "style": ScalarSpec(label="Style", type=str, hidden=True),
        "length": ScalarSpec(label="Length (us)", type=float, decimals=3),
        "sigma": ScalarSpec(label="Sigma (us)", type=float, decimals=3),
        "delta": ScalarSpec(label="Delta (MHz)", type=float, decimals=2),
        "alpha": ScalarSpec(label="Alpha", type=float, decimals=4),
    },
)

ARB_WAVEFORM_SPEC = CfgSectionSpec(
    label="Arb",
    fields={
        "style": ScalarSpec(label="Style", type=str, hidden=True),
        "length": ScalarSpec(label="Length (us)", type=float, decimals=3),
        "data": ScalarSpec(label="Data key", type=str),
    },
)

# Raise waveform for flat_top: only non-flat styles are valid
_RAISE_WAVEFORM_SPEC = WaveformRefSpec(
    allowed=[COSINE_WAVEFORM_SPEC, GAUSS_WAVEFORM_SPEC, DRAG_WAVEFORM_SPEC],
    label="Raise Waveform",
)

FLAT_TOP_WAVEFORM_SPEC = CfgSectionSpec(
    label="FlatTop",
    fields={
        "style": ScalarSpec(label="Style", type=str, hidden=True),
        "length": ScalarSpec(label="Length (us)", type=float, decimals=3),
        "raise_waveform": _RAISE_WAVEFORM_SPEC,
    },
)

# Lookup by style string
WAVEFORM_SPEC_BY_STYLE: dict[str, CfgSectionSpec] = {
    "const": CONST_WAVEFORM_SPEC,
    "cosine": COSINE_WAVEFORM_SPEC,
    "gauss": GAUSS_WAVEFORM_SPEC,
    "drag": DRAG_WAVEFORM_SPEC,
    "arb": ARB_WAVEFORM_SPEC,
    "flat_top": FLAT_TOP_WAVEFORM_SPEC,
}
