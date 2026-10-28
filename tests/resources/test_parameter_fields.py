"""Detached candidate writes preserve models, keys and existing list shape."""

import numpy as np
import pytest
from pydantic import BaseModel
from zcu_tools.resources.parameter_fields import set_model_value
from zcu_tools.resources.projection import ProjectionPathError, projection_value


class Tone(BaseModel):
    frequency: float
    center: object = None


class Candidate(BaseModel):
    tones: list[Tone]
    values: list[object]
    mapping: dict[str, object]


def candidate() -> Candidate:
    return Candidate(
        tones=[Tone(frequency=1.0), Tone(frequency=2.0)],
        values=[0.0, 1.0],
        mapping={"0": {"frequency": 3.0}},
    )


def test_setter_updates_existing_list_model_and_scalar_without_replacing_neighbors():
    model = candidate()
    set_model_value(model, "tones.1.frequency", 5.0)
    set_model_value(model, "values.0", 2 + 3j)
    set_model_value(model, "mapping.0.frequency", 4.0)
    assert model.tones == [Tone(frequency=1.0), Tone(frequency=5.0)]
    assert model.values == [{"__complex__": [2.0, 3.0]}, 1.0]
    assert model.mapping == {"0": {"frequency": 4.0}}


@pytest.mark.parametrize("index", ["-1", "01", "2", "", "1:2", "１", "1" * 5000])
def test_setter_and_reader_reject_noncanonical_or_missing_list_indices(index):
    model = candidate()
    before = model.model_copy(deep=True)
    with pytest.raises((AttributeError, ValueError)):
        set_model_value(model, f"tones.{index}.frequency", 5.0)
    with pytest.raises(ProjectionPathError):
        projection_value(model, ("tones", index, "frequency"))
    assert model == before


@pytest.mark.parametrize("container", [(1.0,), np.array([1.0])])
def test_setter_does_not_expand_read_only_sequence_support(container):
    class ReadOnly(BaseModel):
        values: object

    model = ReadOnly(values=container)
    assert projection_value(model, ("values", "0")) == 1.0
    with pytest.raises(AttributeError):
        set_model_value(model, "values.0", 5.0)
