"""Canonical program module/waveform catalog contract."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from typing import cast

import pytest
from zcu_tools.gui.cfg import (
    CfgNodeSpec,
    CfgSectionSpec,
    LiteralSpec,
    ReferenceSpec,
    ScalarSpec,
)
from zcu_tools.gui.measure_cfg import (
    PROGRAM_SHAPES,
    ProgramCfgKind,
    ProgramSpecPolicy,
    UnknownProgramShapeError,
)
from zcu_tools.program.v2.modules import (
    ArbWaveformCfg,
    BathResetCfg,
    ConstWaveformCfg,
    CosineWaveformCfg,
    DirectReadoutCfg,
    DragWaveformCfg,
    FlatTopWaveformCfg,
    GaussWaveformCfg,
    NoneResetCfg,
    PulseCfg,
    PulseReadoutCfg,
    PulseResetCfg,
    TwoPulseResetCfg,
)

_MAIN_POLICY = ProgramSpecPolicy(
    arb_data_choices_source="arb_waveforms",
    enable_readout_shape_inheritance=True,
)
_AUTOFLUX_POLICY = ProgramSpecPolicy()


def test_catalog_has_exact_ordered_closed_vocabulary() -> None:
    assert [
        (shape.discriminator, shape.label) for shape in PROGRAM_SHAPES.modules()
    ] == [
        ("pulse", "Pulse"),
        ("readout/direct", "Direct Readout"),
        ("readout/pulse", "Pulse Readout"),
        ("reset/none", "None Reset"),
        ("reset/pulse", "Pulse Reset"),
        ("reset/two_pulse", "Two-Pulse Reset"),
        ("reset/bath", "Bath Reset"),
    ]


@pytest.mark.parametrize("attribute", ["_modules", "_waveforms", "_by_kind"])
def test_catalog_singleton_rejects_internal_collection_reassignment(
    attribute: str,
) -> None:
    with pytest.raises(FrozenInstanceError):
        setattr(PROGRAM_SHAPES, attribute, ())
    assert [
        (shape.discriminator, shape.label) for shape in PROGRAM_SHAPES.waveforms()
    ] == [
        ("const", "Const"),
        ("cosine", "Cosine"),
        ("gauss", "Gauss"),
        ("drag", "DRAG"),
        ("arb", "Arb"),
        ("flat_top", "FlatTop"),
    ]


@pytest.mark.parametrize(
    ("kind", "discriminator", "allowed"),
    [
        (
            "module",
            "unknown",
            "pulse, readout/direct, readout/pulse, reset/none, reset/pulse, "
            "reset/two_pulse, reset/bath",
        ),
        (
            "waveform",
            "unknown",
            "const, cosine, gauss, drag, arb, flat_top",
        ),
    ],
)
def test_catalog_unknown_discriminator_fast_fails(
    kind: str, discriminator: str, allowed: str
) -> None:
    with pytest.raises(UnknownProgramShapeError) as exc_info:
        PROGRAM_SHAPES.get(cast(ProgramCfgKind, kind), discriminator)

    assert str(exc_info.value) == (
        f"Unknown {kind} program shape {discriminator!r}; allowed: {allowed}"
    )


def test_catalog_matches_explicit_program_v2_runtime_discriminators() -> None:
    module_classes = (
        PulseCfg,
        DirectReadoutCfg,
        PulseReadoutCfg,
        NoneResetCfg,
        PulseResetCfg,
        TwoPulseResetCfg,
        BathResetCfg,
    )
    waveform_classes = (
        ConstWaveformCfg,
        CosineWaveformCfg,
        GaussWaveformCfg,
        DragWaveformCfg,
        ArbWaveformCfg,
        FlatTopWaveformCfg,
    )

    assert {cls.model_fields["type"].default for cls in module_classes} == {
        shape.discriminator for shape in PROGRAM_SHAPES.modules()
    }
    assert {cls.model_fields["style"].default for cls in waveform_classes} == {
        shape.discriminator for shape in PROGRAM_SHAPES.waveforms()
    }


def test_nested_allowed_sets_match_runtime_shape_rules() -> None:
    pulse = PROGRAM_SHAPES.module("pulse").make_spec(_AUTOFLUX_POLICY)
    waveform = cast(ReferenceSpec, pulse.fields["waveform"])
    assert _literal_values(waveform, "style") == [
        "const",
        "cosine",
        "gauss",
        "drag",
        "arb",
        "flat_top",
    ]

    flat_top = PROGRAM_SHAPES.waveform("flat_top").make_spec(_AUTOFLUX_POLICY)
    raise_waveform = cast(ReferenceSpec, flat_top.fields["raise_waveform"])
    assert _literal_values(raise_waveform, "style") == [
        "cosine",
        "gauss",
        "drag",
        "arb",
    ]


def test_catalog_specs_are_deep_fresh() -> None:
    for shape in (*PROGRAM_SHAPES.modules(), *PROGRAM_SHAPES.waveforms()):
        first = shape.make_spec(_MAIN_POLICY)
        second = shape.make_spec(_MAIN_POLICY)

        assert _normalize_policy_fields(first) == _normalize_policy_fields(second)
        assert _mutable_ids(first).isdisjoint(_mutable_ids(second))


def test_pulse_root_label_override_does_not_create_another_shape() -> None:
    pulse_shape = PROGRAM_SHAPES.module("pulse")

    assert pulse_shape.make_spec(_MAIN_POLICY, label="Pulse 1").label == "Pulse 1"
    assert pulse_shape.label == "Pulse"


def test_cross_app_specs_differ_only_by_two_policy_fields() -> None:
    for shape in (*PROGRAM_SHAPES.modules(), *PROGRAM_SHAPES.waveforms()):
        main = shape.make_spec(_MAIN_POLICY)
        autoflux = shape.make_spec(_AUTOFLUX_POLICY)

        assert _normalize_policy_fields(main) == _normalize_policy_fields(autoflux)

    main_arb = PROGRAM_SHAPES.waveform("arb").make_spec(_MAIN_POLICY)
    autoflux_arb = PROGRAM_SHAPES.waveform("arb").make_spec(_AUTOFLUX_POLICY)
    assert cast(ScalarSpec, main_arb.fields["data"]).choices_source == "arb_waveforms"
    assert cast(ScalarSpec, autoflux_arb.fields["data"]).choices_source == ""

    for discriminator in ("readout/direct", "readout/pulse"):
        assert (
            PROGRAM_SHAPES.module(discriminator).make_spec(_MAIN_POLICY).inherit_hook
            is not None
        )
        assert (
            PROGRAM_SHAPES.module(discriminator)
            .make_spec(_AUTOFLUX_POLICY)
            .inherit_hook
            is None
        )


def _literal_values(reference: ReferenceSpec, key: str) -> list[object]:
    return [cast(LiteralSpec, spec.fields[key]).value for spec in reference.allowed]


def _mutable_ids(spec: CfgSectionSpec) -> set[int]:
    ids = {id(spec.fields)}
    for node in spec.fields.values():
        if isinstance(node, CfgSectionSpec):
            ids.update(_mutable_ids(node))
        elif isinstance(node, ReferenceSpec):
            ids.add(id(node.allowed))
            for allowed in node.allowed:
                ids.update(_mutable_ids(allowed))
        elif isinstance(node, ScalarSpec) and node.choices is not None:
            ids.add(id(node.choices))
    return ids


def _normalize_policy_fields(spec: CfgSectionSpec) -> CfgSectionSpec:
    fields: dict[str, CfgNodeSpec] = {}
    style = spec.fields.get("style")
    is_arb = isinstance(style, LiteralSpec) and style.value == "arb"
    for key, node in spec.fields.items():
        if isinstance(node, CfgSectionSpec):
            fields[key] = _normalize_policy_fields(node)
        elif isinstance(node, ReferenceSpec):
            fields[key] = replace(
                node,
                allowed=[_normalize_policy_fields(item) for item in node.allowed],
            )
        elif isinstance(node, ScalarSpec):
            fields[key] = (
                replace(node, choices_source="") if is_arb and key == "data" else node
            )
        else:
            fields[key] = node
    return replace(spec, fields=fields, inherit_hook=None)
