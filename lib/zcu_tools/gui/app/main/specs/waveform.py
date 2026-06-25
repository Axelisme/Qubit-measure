"""Fresh CfgSectionSpec factories for waveform types."""

from __future__ import annotations

from zcu_tools.gui.app.main.adapter import (
    CfgSectionSpec,
    LiteralSpec,
    ScalarSpec,
    WaveformRefSpec,
)


def make_const_waveform_spec() -> CfgSectionSpec:
    return CfgSectionSpec(
        label="Const",
        fields={
            "style": LiteralSpec("const"),
            "length": ScalarSpec(label="Length (us)", type=float, decimals=3),
        },
    )


def make_cosine_waveform_spec() -> CfgSectionSpec:
    return CfgSectionSpec(
        label="Cosine",
        fields={
            "style": LiteralSpec("cosine"),
            "length": ScalarSpec(label="Length (us)", type=float, decimals=3),
        },
    )


def make_gauss_waveform_spec() -> CfgSectionSpec:
    return CfgSectionSpec(
        label="Gauss",
        fields={
            "style": LiteralSpec("gauss"),
            "length": ScalarSpec(label="Length (us)", type=float, decimals=3),
            "sigma": ScalarSpec(label="Sigma (us)", type=float, decimals=3),
        },
    )


def make_drag_waveform_spec() -> CfgSectionSpec:
    return CfgSectionSpec(
        label="DRAG",
        fields={
            "style": LiteralSpec("drag"),
            "length": ScalarSpec(label="Length (us)", type=float, decimals=3),
            "sigma": ScalarSpec(label="Sigma (us)", type=float, decimals=3),
            "delta": ScalarSpec(label="Delta (MHz)", type=float, decimals=2),
            "alpha": ScalarSpec(label="Alpha", type=float, decimals=4),
        },
    )


def make_arb_waveform_spec() -> CfgSectionSpec:
    return CfgSectionSpec(
        label="Arb",
        fields={
            "style": LiteralSpec("arb"),
            "length": ScalarSpec(label="Length (us)", type=float, decimals=3),
            "data": ScalarSpec(
                label="Data key",
                type=str,
                choices_source="arb_waveforms",
            ),
        },
    )


def make_flat_top_waveform_spec() -> CfgSectionSpec:
    raise_waveform_spec = WaveformRefSpec(
        allowed=[
            make_cosine_waveform_spec(),
            make_gauss_waveform_spec(),
            make_drag_waveform_spec(),
            make_arb_waveform_spec(),
        ],
        label="Raise Waveform",
    )
    return CfgSectionSpec(
        label="FlatTop",
        fields={
            "style": LiteralSpec("flat_top"),
            "length": ScalarSpec(label="Length (us)", type=float, decimals=3),
            "raise_waveform": raise_waveform_spec,
        },
    )


def make_waveform_spec_by_style(style: str) -> CfgSectionSpec:
    factories = {
        "const": make_const_waveform_spec,
        "cosine": make_cosine_waveform_spec,
        "gauss": make_gauss_waveform_spec,
        "drag": make_drag_waveform_spec,
        "arb": make_arb_waveform_spec,
        "flat_top": make_flat_top_waveform_spec,
    }
    factory = factories.get(style, make_const_waveform_spec)
    return factory()
