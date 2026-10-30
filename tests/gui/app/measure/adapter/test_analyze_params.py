from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Literal

import pytest
from pydantic import Field, TypeAdapter
from pydantic.dataclasses import dataclass as validated_dataclass
from zcu_tools.cfg_model import FieldMeta
from zcu_tools.gui.app.measure.adapter import ParamMeta
from zcu_tools.gui.app.measure.adapter.analyze_params import (
    describe_analyze_params,
    reconstruct_params,
)


@dataclass
class _CoreOptions:
    mode: Annotated[Literal["lor", "sinc"], FieldMeta(label="Model type")] = "lor"
    cutoff: Annotated[float | None, FieldMeta(label="Gain cutoff", decimals=3)] = None
    enabled: Annotated[bool, FieldMeta()] = True


@validated_dataclass(frozen=True)
class _ValidatedOptions:
    skip: Annotated[int, FieldMeta(label="Skip points"), Field(ge=0)] = 0


@dataclass(frozen=True)
class _PlainOptions:
    skip: Annotated[int, FieldMeta(label="Skip points")] = 0


@pytest.mark.parametrize("options_type", [_PlainOptions, _ValidatedOptions])
def test_dataclass_options_share_description_and_reconstruction(options_type):
    assert describe_analyze_params(options_type) == [
        {"name": "skip", "type": "int", "label": "Skip points", "default": 0}
    ]
    assert reconstruct_params(options_type, {"skip": 2}) == options_type(skip=2)


@pytest.mark.parametrize(
    "construct",
    [
        lambda: _ValidatedOptions(skip=-1),
        lambda: reconstruct_params(_ValidatedOptions, {"skip": -1}),
        lambda: TypeAdapter(_ValidatedOptions).validate_python({"skip": -1}),
    ],
)
def test_options_constraints_reject_invalid_values_at_construction(construct):
    with pytest.raises(ValueError, match="greater than or equal to 0"):
        construct()


def test_describe_core_options_uses_field_metadata():
    assert describe_analyze_params(_CoreOptions) == [
        {
            "name": "mode",
            "type": "str",
            "label": "Model type",
            "choices": ["lor", "sinc"],
            "default": "lor",
        },
        {
            "name": "cutoff",
            "type": "float",
            "label": "Gain cutoff",
            "decimals": 3,
            "optional": True,
            "default": None,
        },
        {"name": "enabled", "type": "bool", "label": "enabled", "default": True},
    ]


def test_describe_bool_field():
    @dataclass
    class P:
        flag: bool

    assert describe_analyze_params(P) == [
        {"name": "flag", "type": "bool", "label": "flag"}
    ]


def test_describe_literal_field():
    @dataclass
    class P:
        mode: Annotated[Literal["a", "b"], ParamMeta(label="Mode")]

    assert describe_analyze_params(P) == [
        {
            "name": "mode",
            "type": "str",
            "choices": ["a", "b"],
            "label": "Mode",
        }
    ]


def test_reconstruct_params_basic():
    @dataclass
    class P:
        x: float
        flag: bool

    result = reconstruct_params(P, {"x": 1.5, "flag": True})

    assert result == P(x=1.5, flag=True)


def test_reconstruct_params_extra_key_raises():
    @dataclass
    class P:
        x: float

    with pytest.raises(RuntimeError, match="Unknown analyze params"):
        reconstruct_params(P, {"x": 1.0, "extra": 99})


def test_reconstruct_params_rejects_bool_as_int():
    @dataclass
    class P:
        x: int

    with pytest.raises(RuntimeError, match="expects int"):
        reconstruct_params(P, {"x": True})


def test_describe_mixed_literal_types_raise():
    @dataclass
    class P:
        mode: Literal["a", 1]

    with pytest.raises(TypeError, match="one type"):
        describe_analyze_params(P)


def test_unsupported_annotation_raises(qapp):
    @dataclass
    class P:
        val: list[int]

    from zcu_tools.gui.app.measure.ui.analyze_form import AnalyzeFormWidget

    form = AnalyzeFormWidget()
    with pytest.raises(TypeError, match="Unsupported analyze parameter annotation"):
        form.populate(P(val=[]))


def test_describe_optional_field():
    @dataclass
    class P:
        t0: float | None = None

    assert describe_analyze_params(P) == [
        {
            "name": "t0",
            "type": "float",
            "label": "t0",
            "optional": True,
            "default": None,
        }
    ]


def test_describe_optional_annotated_keeps_meta_and_default():
    @dataclass
    class P:
        t0: Annotated[float | None, ParamMeta(label="T0")] = None

    assert describe_analyze_params(P) == [
        {
            "name": "t0",
            "type": "float",
            "label": "T0",
            "optional": True,
            "default": None,
        }
    ]


def test_describe_non_optional_union_rejected():
    @dataclass
    class P:
        x: int | str

    with pytest.raises(TypeError, match="only Optional"):
        describe_analyze_params(P)


def test_reconstruct_optional_none_passes_through():
    @dataclass
    class P:
        t0: float | None = None

    assert reconstruct_params(P, {"t0": None}) == P(t0=None)


def test_reconstruct_optional_value_is_coerced():
    @dataclass
    class P:
        t0: float | None = None

    result = reconstruct_params(P, {"t0": 2})

    assert result == P(t0=2.0)
    assert type(result.t0) is float
