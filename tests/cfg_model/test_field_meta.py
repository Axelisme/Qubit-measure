"""Front-end-neutral field descriptions survive Pydantic type wrappers."""

from typing import Annotated, Union

import pytest
from pydantic import BaseModel, Field


@pytest.mark.parametrize(
    "name",
    ["direct", "outer_optional", "inner_optional", "union_member", "nested"],
)
def test_reads_field_meta_through_annotated_and_union(name: str) -> None:
    from zcu_tools.cfg_model import FieldMeta, get_field_meta

    meta = FieldMeta(
        label="Frequency", unit="MHz", decimals=3, description="Probe frequency."
    )

    class Cfg(BaseModel):
        direct: Annotated[float, meta]
        outer_optional: Annotated[float | None, meta] = None
        inner_optional: Annotated[float, meta] | None = None
        union_member: Union[Annotated[float, meta], str]
        nested: Annotated[Annotated[float, meta] | None, Field(description="Other")] = None

    assert get_field_meta(Cfg.model_fields[name]) == meta


@pytest.mark.parametrize("name", ["plain", "optional", "other_metadata"])
def test_unannotated_fields_have_no_field_meta(name: str) -> None:
    from zcu_tools.cfg_model import get_field_meta

    class Cfg(BaseModel):
        plain: float
        optional: float | str | None = None
        other_metadata: Annotated[float, Field(gt=0, description="Not FieldMeta")]

    assert get_field_meta(Cfg.model_fields[name]) is None
