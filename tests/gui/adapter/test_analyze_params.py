from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Annotated, Literal, Optional, Union, get_type_hints

import pytest
from zcu_tools.gui.app.main.adapter import ParamMeta
from zcu_tools.gui.app.main.adapter.analyze_params import (
    _resolve_field_info,
    describe_analyze_params,
    reconstruct_params,
)


def test_resolve_bool_field():
    @dataclass
    class P:
        flag: bool

    hints = get_type_hints(P, include_extras=True)
    field = dataclasses.fields(P)[0]
    bare, choices, label, decimals, optional = _resolve_field_info(field, hints)

    assert bare is bool
    assert choices is None
    assert label == "flag"
    assert decimals is None
    assert optional is False


def test_resolve_literal_field():
    @dataclass
    class P:
        mode: Annotated[Literal["a", "b"], ParamMeta(label="Mode")]

    hints = get_type_hints(P, include_extras=True)
    field = dataclasses.fields(P)[0]
    bare, choices, label, decimals, optional = _resolve_field_info(field, hints)

    assert bare is str
    assert choices == ["a", "b"]
    assert label == "Mode"
    assert decimals is None
    assert optional is False


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


def test_mixed_literal_types_raise():
    @dataclass
    class P:
        mode: Literal["a", 1]

    hints = get_type_hints(P, include_extras=True)
    field = dataclasses.fields(P)[0]
    with pytest.raises(TypeError, match="one type"):
        _resolve_field_info(field, hints)


def test_unsupported_annotation_raises(qapp):  # noqa: ARG001
    @dataclass
    class P:
        val: list

    from zcu_tools.gui.app.main.ui.analyze_form import AnalyzeFormWidget

    form = AnalyzeFormWidget()
    with pytest.raises(TypeError, match="Unsupported analyze parameter annotation"):
        form.populate(P(val=[]))


# --- optional analyze params (Optional[T]) ---------------------------------


def test_resolve_optional_field():
    @dataclass
    class P:
        t0: Optional[float] = None

    hints = get_type_hints(P, include_extras=True)
    field = dataclasses.fields(P)[0]
    bare, choices, _label, _decimals, optional = _resolve_field_info(field, hints)

    assert bare is float  # the None is stripped, T resolved
    assert choices is None
    assert optional is True


def test_resolve_optional_annotated_keeps_meta():
    @dataclass
    class P:
        t0: Annotated[Optional[float], ParamMeta(label="T0")] = None

    hints = get_type_hints(P, include_extras=True)
    field = dataclasses.fields(P)[0]
    bare, _choices, label, _decimals, optional = _resolve_field_info(field, hints)

    assert bare is float
    assert optional is True
    assert label == "T0"


def test_non_optional_union_rejected():
    @dataclass
    class P:
        x: Union[int, str]

    hints = get_type_hints(P, include_extras=True)
    field = dataclasses.fields(P)[0]
    with pytest.raises(TypeError, match="only Optional"):
        _resolve_field_info(field, hints)


def test_describe_marks_optional_with_default_none():
    @dataclass
    class P:
        t0: Annotated[Optional[float], ParamMeta(label="T0")] = None

    assert describe_analyze_params(P) == [
        {
            "name": "t0",
            "type": "float",
            "label": "T0",
            "optional": True,
            "default": None,
        }
    ]


def test_reconstruct_optional_none_passes_through():
    @dataclass
    class P:
        t0: Optional[float] = None

    assert reconstruct_params(P, {"t0": None}) == P(t0=None)


def test_reconstruct_optional_value_is_coerced():
    @dataclass
    class P:
        t0: Optional[float] = None

    assert reconstruct_params(P, {"t0": 2}) == P(t0=2.0)
