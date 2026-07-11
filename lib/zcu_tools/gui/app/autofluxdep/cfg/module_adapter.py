"""Autoflux policy adapter for program cfg materialization and shape lookup."""

from __future__ import annotations

from typing import Any

from zcu_tools.gui.cfg import CfgSectionSpec, CfgSectionValue, ReferenceSpec
from zcu_tools.gui.measure_cfg import (
    PROGRAM_SHAPES,
    ProgramMaterializationPolicy,
    ProgramSpecPolicy,
    UnknownProgramShapeError,
    materialize_program_module,
    materialize_program_waveform,
)
from zcu_tools.program.v2.modules.base import AbsModuleCfg
from zcu_tools.program.v2.modules.waveform import AbsWaveformCfg

AUTOFLUX_PROGRAM_SPEC_POLICY = ProgramSpecPolicy()
AUTOFLUX_PROGRAM_MATERIALIZATION_POLICY = ProgramMaterializationPolicy(
    spec_policy=AUTOFLUX_PROGRAM_SPEC_POLICY,
    allowed_module_discriminators=frozenset({"pulse", "readout/pulse"}),
    allowed_waveform_styles=frozenset(
        shape.discriminator for shape in PROGRAM_SHAPES.waveforms()
    ),
)


def _normalize_waveform_cfg(cfg_input: object) -> dict[str, Any]:
    if isinstance(cfg_input, AbsWaveformCfg):
        return cfg_input.to_dict()
    if isinstance(cfg_input, dict):
        return cfg_input
    raise TypeError(f"Expected dict or AbsWaveformCfg, got {type(cfg_input)}")


def _normalize_module_cfg(cfg_input: object) -> dict[str, Any]:
    if isinstance(cfg_input, (AbsModuleCfg, AbsWaveformCfg)):
        return cfg_input.to_dict()
    if isinstance(cfg_input, dict):
        return cfg_input
    raise TypeError(f"Expected dict or ModuleCfg, got {type(cfg_input)}")


def _make_pulse_spec(label: str = "Pulse") -> CfgSectionSpec:
    return PROGRAM_SHAPES.module("pulse").make_spec(
        AUTOFLUX_PROGRAM_SPEC_POLICY,
        label=label,
    )


def _make_pulse_readout_spec() -> CfgSectionSpec:
    return PROGRAM_SHAPES.module("readout/pulse").make_spec(
        AUTOFLUX_PROGRAM_SPEC_POLICY
    )


def pulse_module_ref_spec(
    label: str = "Pulse",
    optional: bool = False,
) -> ReferenceSpec:
    return ReferenceSpec(
        kind="module",
        allowed=[_make_pulse_spec()],
        label=label,
        optional=optional,
    )


def pulse_readout_module_ref_spec(
    label: str = "Readout",
    optional: bool = False,
) -> ReferenceSpec:
    return ReferenceSpec(
        kind="module",
        allowed=[_make_pulse_readout_spec()],
        label=label,
        optional=optional,
    )


def waveform_cfg_to_value(
    cfg_input: object,
) -> tuple[CfgSectionSpec, CfgSectionValue]:
    return materialize_program_waveform(
        _normalize_waveform_cfg(cfg_input),
        AUTOFLUX_PROGRAM_MATERIALIZATION_POLICY,
    )


def module_cfg_to_value(
    cfg_input: object,
) -> tuple[CfgSectionSpec, CfgSectionValue]:
    """Convert only Pulse and Pulse Readout module roots."""

    return materialize_program_module(
        _normalize_module_cfg(cfg_input),
        AUTOFLUX_PROGRAM_MATERIALIZATION_POLICY,
    )


def module_cfg_shape_label(cfg_input: object) -> str:
    """Resolve every legal module discriminator without materializing it."""

    cfg = _normalize_module_cfg(cfg_input)
    if "style" in cfg:
        return waveform_cfg_shape_label(cfg)
    type_value = _read_discriminator(cfg, "type", missing="")
    try:
        return PROGRAM_SHAPES.module(type_value).label
    except UnknownProgramShapeError as exc:
        raise RuntimeError(f"Unsupported module type {type_value!r}") from exc


def waveform_cfg_shape_label(cfg_input: object) -> str:
    cfg = _normalize_waveform_cfg(cfg_input)
    style = _read_discriminator(cfg, "style", missing="const")
    try:
        return PROGRAM_SHAPES.waveform(style).label
    except UnknownProgramShapeError as exc:
        raise RuntimeError(f"Unsupported waveform style {style!r}") from exc


def _read_discriminator(
    raw: dict[str, Any],
    key: str,
    *,
    missing: str,
) -> str:
    if key not in raw:
        return missing
    value = raw[key]
    if not isinstance(value, str):
        raise TypeError(
            f"Program discriminator {key!r} must be str, got {type(value).__name__}"
        )
    return value
