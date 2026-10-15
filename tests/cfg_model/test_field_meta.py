"""Front-end-neutral field descriptions survive Pydantic type wrappers."""

from typing import Annotated

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
        union_member: Annotated[float, meta] | str
        nested: Annotated[Annotated[float, meta] | None, Field(description="Other")] = (
            None
        )

    assert get_field_meta(Cfg.model_fields[name]) == meta


@pytest.mark.parametrize("name", ["plain", "optional", "other_metadata"])
def test_unannotated_fields_have_no_field_meta(name: str) -> None:
    from zcu_tools.cfg_model import get_field_meta

    class Cfg(BaseModel):
        plain: float
        optional: float | str | None = None
        other_metadata: Annotated[float, Field(gt=0, description="Not FieldMeta")]

    assert get_field_meta(Cfg.model_fields[name]) is None


@pytest.mark.parametrize("group", [None, "Advanced"])
def test_group_is_readable_without_changing_field_values(group: str | None) -> None:
    from zcu_tools.cfg_model import FieldMeta, get_field_meta

    try:
        meta = FieldMeta(group=group)
    except TypeError:
        pytest.fail("FieldMeta must accept a presentation group")

    class Cfg(BaseModel):
        frequency: Annotated[float | None, meta] = None

    read_meta = get_field_meta(Cfg.model_fields["frequency"])
    assert read_meta is not None
    assert read_meta.group == group
    assert Cfg(frequency=125.5).model_dump() == {"frequency": 125.5}
    assert Cfg().model_dump() == {"frequency": None}
    assert FieldMeta().group is None


def test_empty_group_is_rejected_at_construction() -> None:
    from zcu_tools.cfg_model import FieldMeta

    try:
        with pytest.raises(ValueError, match="group"):
            FieldMeta(group="")
    except TypeError:
        pytest.fail("An empty group must be rejected with ValueError")
