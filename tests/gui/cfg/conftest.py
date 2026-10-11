"""Function-scoped reference resources shared by reference-mode contracts."""

import pytest
from zcu_tools.gui.cfg.binding.ports import ResolvedReference
from zcu_tools.gui.cfg.model import (
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    LiteralSpec,
    ReferenceSpec,
    ReferenceValue,
    ScalarSpec,
    SweepSpec,
)
from zcu_tools.gui.cfg.resource import CfgResource

from ._reference_support import Catalog, entry


@pytest.fixture
def reference_resource() -> tuple[CfgResource, Catalog]:
    shape = CfgSectionSpec(fields={"x": ScalarSpec("X", float)}, label="Shape")
    schema = CfgSchema(
        CfgSectionSpec(fields={"ref": ReferenceSpec("test", [shape], optional=True)}),
        CfgSectionValue(
            {"ref": ReferenceValue("first", CfgSectionValue({"x": DirectValue(1.0)}))}
        ),
    )
    catalog = Catalog({"first": entry(1.0), "second": entry(2.0)})
    resource = CfgResource(
        lambda: schema,
        resolution=catalog.snapshot,
    )
    return resource, catalog


@pytest.fixture
def switchable_resource() -> CfgResource:
    first = CfgSectionSpec(
        fields={"type": LiteralSpec("first"), "old": ScalarSpec("Old", float)},
        label="First",
    )
    child = CfgSectionSpec(fields={"x": ScalarSpec("X", float)}, label="Child")
    second = CfgSectionSpec(
        fields={
            "type": LiteralSpec("second"),
            "fixed": LiteralSpec("locked"),
            "number": ScalarSpec("Number", float),
            "optional": ScalarSpec("Optional", float, optional=True),
            "child": ReferenceSpec("test", [child]),
            "range": SweepSpec(),
        },
        label="Second",
    )
    schema = CfgSchema(
        CfgSectionSpec(
            fields={"ref": ReferenceSpec("test", [first, second], discriminator="type")}
        ),
        CfgSectionValue(
            {
                "ref": ReferenceValue(
                    "<Custom:First>",
                    CfgSectionValue(
                        {"type": DirectValue("first"), "old": DirectValue(99.0)}
                    ),
                )
            }
        ),
    )
    return CfgResource(
        lambda: schema,
        resolution=Catalog(
            {
                "child": ResolvedReference(
                    "Child", CfgSectionValue({"x": DirectValue(2.0)})
                )
            }
        ).snapshot,
    )
