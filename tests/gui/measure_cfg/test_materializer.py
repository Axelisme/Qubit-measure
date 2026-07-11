from __future__ import annotations

from typing import cast

import pytest
from zcu_tools.gui.cfg import (
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    LiteralSpec,
    ReferenceSpec,
    ReferenceValue,
    ScalarSpec,
    parse_custom_reference_key,
)
from zcu_tools.gui.measure_cfg import (
    PROGRAM_SHAPES,
    ProgramMaterializationPolicy,
    ProgramSpecPolicy,
    materialize_program_module,
    materialize_program_waveform,
)

_ALL_MODULES = frozenset(shape.discriminator for shape in PROGRAM_SHAPES.modules())
_ALL_WAVEFORMS = frozenset(shape.discriminator for shape in PROGRAM_SHAPES.waveforms())
_POLICY = ProgramMaterializationPolicy(
    spec_policy=ProgramSpecPolicy(),
    allowed_module_discriminators=_ALL_MODULES,
    allowed_waveform_styles=_ALL_WAVEFORMS,
)


class _StringImpostor:
    def __init__(self, text: str) -> None:
        self._text = text

    def __str__(self) -> str:
        return self._text


def _serialize_value(value: object) -> object:
    if isinstance(value, DirectValue):
        return value.value
    if isinstance(value, CfgSectionValue):
        return {key: _serialize_value(child) for key, child in value.fields.items()}
    if isinstance(value, ReferenceValue):
        return {
            "chosen_key": value.chosen_key,
            "value": _serialize_value(value.value),
        }
    if value is None:
        return None
    raise AssertionError(type(value).__name__)


_NEUTRAL_CONST = {"style": "const", "length": 0.0}
_NEUTRAL_PULSE = {
    "type": "pulse",
    "waveform": {"chosen_key": "<Custom:Const>", "value": _NEUTRAL_CONST},
    "ch": 0,
    "nqz": 1,
    "freq": 0.0,
    "gain": 0.0,
    "phase": 0.0,
    "pre_delay": 0.0,
    "post_delay": 0.0,
    "mixer_freq": None,
}
_ROOT_PULSE = {
    **_NEUTRAL_PULSE,
    "nqz": None,
    "freq": None,
    "gain": None,
    "phase": None,
    "pre_delay": None,
    "post_delay": None,
}
_NEUTRAL_DIRECT_READOUT = {
    "type": "readout/direct",
    "ro_ch": 0,
    "ro_freq": 0.0,
    "ro_length": 0.0,
    "trig_offset": 0.0,
    "gen_ch": None,
}
_PULSE_REFERENCE = {
    "chosen_key": "<Custom:Pulse>",
    "value": _NEUTRAL_PULSE,
}


@pytest.mark.parametrize(
    ("kind", "raw", "expected"),
    [
        ("module", {"type": "pulse"}, _ROOT_PULSE),
        (
            "module",
            {"type": "readout/direct"},
            {
                "type": "readout/direct",
                "ro_ch": 0,
                "ro_freq": None,
                "ro_length": None,
                "trig_offset": None,
                "gen_ch": None,
            },
        ),
        (
            "module",
            {"type": "readout/pulse"},
            {
                "type": "readout/pulse",
                "pulse_cfg": _NEUTRAL_PULSE,
                "ro_cfg": _NEUTRAL_DIRECT_READOUT,
            },
        ),
        ("module", {"type": "reset/none"}, {"type": "reset/none"}),
        (
            "module",
            {"type": "reset/pulse"},
            {"type": "reset/pulse", "pulse_cfg": _NEUTRAL_PULSE},
        ),
        (
            "module",
            {"type": "reset/two_pulse"},
            {
                "type": "reset/two_pulse",
                "pulse1_cfg": _NEUTRAL_PULSE,
                "pulse2_cfg": _NEUTRAL_PULSE,
            },
        ),
        (
            "module",
            {"type": "reset/bath"},
            {
                "type": "reset/bath",
                "cavity_tone_cfg": _PULSE_REFERENCE,
                "qubit_tone_cfg": _PULSE_REFERENCE,
                "pi2_cfg": _PULSE_REFERENCE,
            },
        ),
        ("waveform", {"style": "const"}, {"style": "const", "length": None}),
        (
            "waveform",
            {"style": "cosine"},
            {"style": "cosine", "length": None},
        ),
        (
            "waveform",
            {"style": "gauss"},
            {"style": "gauss", "length": None, "sigma": None},
        ),
        (
            "waveform",
            {"style": "drag"},
            {
                "style": "drag",
                "length": None,
                "sigma": None,
                "delta": None,
                "alpha": None,
            },
        ),
        (
            "waveform",
            {"style": "flat_top"},
            {
                "style": "flat_top",
                "length": None,
                "raise_waveform": {
                    "chosen_key": "<Custom:Cosine>",
                    "value": {"style": "cosine", "length": 0.0},
                },
            },
        ),
        ("waveform", {"style": "arb"}, {"style": "arb", "data": None}),
    ],
)
def test_all_program_roots_have_exact_serialized_value_golden(
    kind: str,
    raw: dict[str, object],
    expected: dict[str, object],
) -> None:
    if kind == "module":
        _, value = materialize_program_module(raw, _POLICY)
    else:
        _, value = materialize_program_waveform(raw, _POLICY)

    assert _serialize_value(value) == expected


def _assert_spec_value_shape(
    spec: CfgSectionSpec,
    value: CfgSectionValue,
) -> None:
    assert tuple(value.fields) == tuple(spec.fields)
    for key, node_spec in spec.fields.items():
        node_value = value.fields[key]
        if isinstance(node_spec, (LiteralSpec, ScalarSpec)):
            assert isinstance(node_value, DirectValue)
        elif isinstance(node_spec, CfgSectionSpec):
            assert isinstance(node_value, CfgSectionValue)
            _assert_spec_value_shape(node_spec, node_value)
        elif isinstance(node_spec, ReferenceSpec):
            assert isinstance(node_value, ReferenceValue)
            label = parse_custom_reference_key(node_value.chosen_key)
            selected = next(item for item in node_spec.allowed if item.label == label)
            _assert_spec_value_shape(selected, node_value.value)
        else:  # pragma: no cover - program catalog has no other spec kinds
            raise AssertionError(type(node_spec).__name__)


@pytest.mark.parametrize(
    "discriminator",
    [shape.discriminator for shape in PROGRAM_SHAPES.modules()],
)
def test_program_module_missing_defaults_are_complete(discriminator: str) -> None:
    spec, value = materialize_program_module({"type": discriminator}, _POLICY)

    _assert_spec_value_shape(spec, value)
    if discriminator == "pulse":
        assert value.fields["ch"] == DirectValue(0)
        assert value.fields["freq"] == DirectValue(None)
        waveform = cast(ReferenceValue, value.fields["waveform"])
        assert parse_custom_reference_key(waveform.chosen_key) == "Const"
        assert waveform.value.fields["length"] == DirectValue(0.0)
    elif discriminator == "readout/direct":
        assert value.fields["ro_ch"] == DirectValue(0)
        assert value.fields["ro_freq"] == DirectValue(None)
    elif discriminator == "readout/pulse":
        pulse = cast(CfgSectionValue, value.fields["pulse_cfg"])
        readout = cast(CfgSectionValue, value.fields["ro_cfg"])
        assert pulse.fields["freq"] == DirectValue(0.0)
        assert readout.fields["ro_freq"] == DirectValue(0.0)


@pytest.mark.parametrize(
    "style",
    [shape.discriminator for shape in PROGRAM_SHAPES.waveforms()],
)
def test_program_waveform_missing_defaults_are_complete(style: str) -> None:
    spec, value = materialize_program_waveform({"style": style}, _POLICY)

    _assert_spec_value_shape(spec, value)
    for key, node_spec in spec.fields.items():
        if isinstance(node_spec, ScalarSpec):
            assert value.fields[key] == DirectValue(None)
    if style == "flat_top":
        raise_waveform = cast(ReferenceValue, value.fields["raise_waveform"])
        assert parse_custom_reference_key(raise_waveform.chosen_key) == "Cosine"
        assert raise_waveform.value.fields["length"] == DirectValue(0.0)


def test_program_waveform_missing_style_defaults_to_const() -> None:
    spec, value = materialize_program_waveform({}, _POLICY)

    assert spec.label == "Const"
    assert value.fields == {
        "style": DirectValue("const"),
        "length": DirectValue(None),
    }


def test_program_materializer_rejects_explicit_unknown_style() -> None:
    with pytest.raises(RuntimeError, match="Unsupported waveform style 'mystery'"):
        materialize_program_waveform({"style": "mystery"}, _POLICY)


@pytest.mark.parametrize("value", [_StringImpostor("pulse"), None, 7])
def test_program_module_discriminator_must_be_string(value: object) -> None:
    with pytest.raises(TypeError, match=r"'type' must be str"):
        materialize_program_module({"type": value}, _POLICY)


@pytest.mark.parametrize("value", [_StringImpostor("const"), None, 7])
def test_program_waveform_discriminator_must_be_string(value: object) -> None:
    with pytest.raises(TypeError, match=r"'style' must be str"):
        materialize_program_waveform({"style": value}, _POLICY)


def test_program_materializer_rejects_bath_module_relax_delay() -> None:
    with pytest.raises(
        RuntimeError,
        match="reset/bath.*does not accept.*relax_delay.*program root",
    ):
        materialize_program_module(
            {"type": "reset/bath", "relax_delay": 10.0},
            _POLICY,
        )


def test_single_allowed_reference_missing_discriminator_uses_allowed_shape() -> None:
    _, value = materialize_program_module(
        {
            "type": "reset/bath",
            "cavity_tone_cfg": {"freq": 6000.0},
            "qubit_tone_cfg": {"freq": 4000.0},
            "pi2_cfg": {"phase": 90.0},
        },
        _POLICY,
    )

    cavity = cast(ReferenceValue, value.fields["cavity_tone_cfg"])
    assert cavity.value.fields["type"] == DirectValue("pulse")
    assert cavity.value.fields["freq"] == DirectValue(6000.0)


def test_program_materializer_rejects_legal_module_outside_policy_subset() -> None:
    policy = ProgramMaterializationPolicy(
        spec_policy=ProgramSpecPolicy(),
        allowed_module_discriminators=frozenset({"pulse", "readout/pulse"}),
        allowed_waveform_styles=_ALL_WAVEFORMS,
    )

    with pytest.raises(RuntimeError) as exc_info:
        materialize_program_module({"type": "readout/direct"}, policy)

    assert str(exc_info.value) == "Unsupported module type 'readout/direct'"
