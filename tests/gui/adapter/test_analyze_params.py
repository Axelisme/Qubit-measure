from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Annotated, Literal, get_type_hints

import pytest
from zcu_tools.gui.app.main.adapter import ParamMeta
from zcu_tools.gui.app.main.adapter.analyze_params import (
    _resolve_field_info,
    reconstruct_params,
)


def test_resolve_bool_field():
    @dataclass
    class P:
        flag: bool

    hints = get_type_hints(P, include_extras=True)
    field = dataclasses.fields(P)[0]
    bare, choices, label, decimals = _resolve_field_info(field, hints)

    assert bare is bool
    assert choices is None
    assert label == "flag"
    assert decimals is None


def test_resolve_literal_field():
    @dataclass
    class P:
        mode: Annotated[Literal["a", "b"], ParamMeta(label="Mode")]

    hints = get_type_hints(P, include_extras=True)
    field = dataclasses.fields(P)[0]
    bare, choices, label, decimals = _resolve_field_info(field, hints)

    assert bare is str
    assert choices == ["a", "b"]
    assert label == "Mode"
    assert decimals is None


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
