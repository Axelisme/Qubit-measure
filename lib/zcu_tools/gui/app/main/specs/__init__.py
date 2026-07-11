"""Main-app policy adapter for canonical program cfg shapes."""

from __future__ import annotations

from zcu_tools.gui.cfg import CfgSectionSpec
from zcu_tools.gui.measure_cfg import PROGRAM_SHAPES, ProgramSpecPolicy

MAIN_PROGRAM_SPEC_POLICY = ProgramSpecPolicy(
    arb_data_choices_source="arb_waveforms",
    enable_readout_shape_inheritance=True,
)


def _module(discriminator: str, *, label: str | None = None) -> CfgSectionSpec:
    return PROGRAM_SHAPES.module(discriminator).make_spec(
        MAIN_PROGRAM_SPEC_POLICY, label=label
    )


def _waveform(style: str) -> CfgSectionSpec:
    return PROGRAM_SHAPES.waveform(style).make_spec(MAIN_PROGRAM_SPEC_POLICY)


def make_pulse_spec(label: str = "Pulse") -> CfgSectionSpec:
    return _module("pulse", label=label)


def make_direct_readout_spec() -> CfgSectionSpec:
    return _module("readout/direct")


def make_pulse_readout_spec() -> CfgSectionSpec:
    return _module("readout/pulse")


def make_none_reset_spec() -> CfgSectionSpec:
    return _module("reset/none")


def make_pulse_reset_spec() -> CfgSectionSpec:
    return _module("reset/pulse")


def make_two_pulse_reset_spec() -> CfgSectionSpec:
    return _module("reset/two_pulse")


def make_bath_reset_spec() -> CfgSectionSpec:
    return _module("reset/bath")


def make_const_waveform_spec() -> CfgSectionSpec:
    return _waveform("const")


def make_cosine_waveform_spec() -> CfgSectionSpec:
    return _waveform("cosine")


def make_gauss_waveform_spec() -> CfgSectionSpec:
    return _waveform("gauss")


def make_drag_waveform_spec() -> CfgSectionSpec:
    return _waveform("drag")


def make_arb_waveform_spec() -> CfgSectionSpec:
    return _waveform("arb")


def make_flat_top_waveform_spec() -> CfgSectionSpec:
    return _waveform("flat_top")


def make_waveform_spec_by_style(style: str) -> CfgSectionSpec:
    return _waveform(style)


__all__ = [
    "MAIN_PROGRAM_SPEC_POLICY",
    "make_arb_waveform_spec",
    "make_bath_reset_spec",
    "make_const_waveform_spec",
    "make_cosine_waveform_spec",
    "make_direct_readout_spec",
    "make_drag_waveform_spec",
    "make_flat_top_waveform_spec",
    "make_gauss_waveform_spec",
    "make_none_reset_spec",
    "make_pulse_readout_spec",
    "make_pulse_reset_spec",
    "make_pulse_spec",
    "make_two_pulse_reset_spec",
    "make_waveform_spec_by_style",
]
