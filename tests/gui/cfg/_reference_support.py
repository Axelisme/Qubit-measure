"""Recording reference catalog and nested waveform schemas for resource contracts."""

from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass, field

from zcu_tools.experiment.cfg_editing.catalog import PROGRAM_SHAPES, ProgramSpecPolicy
from zcu_tools.gui.cfg.binding.ports import ResolvedReference
from zcu_tools.gui.cfg.model import (
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    ReferenceSpec,
    ReferenceValue,
)
from zcu_tools.gui.cfg.resource import (
    CfgResolution,
    CfgResource,
)
from zcu_tools.gui.session.expression import validate_scalar_expr


@dataclass
class Catalog:
    entries: dict[str, ResolvedReference] = field(default_factory=dict)
    lookups: list[str] = field(default_factory=list)
    failed_keys: frozenset[str] = frozenset()

    def keys(self, kind: str, allowed_labels: frozenset[str]) -> Sequence[str]:
        return tuple(
            key for key, value in self.entries.items() if value.label in allowed_labels
        )

    def resolve(self, kind: str, key: str) -> ResolvedReference | None:
        self.lookups.append(key)
        if key in self.failed_keys:
            raise RuntimeError("reference resolver defect")
        return self.entries.get(key)

    def snapshot(self) -> CfgResolution:
        frozen = Catalog(deepcopy(self.entries), self.lookups, self.failed_keys)
        return CfgResolution(
            (),
            lambda expression: 0.0,
            lambda source_id: (),
            frozen,
            lambda name: 0.0,
            validate_scalar_expr,
        )


def entry(value: float) -> ResolvedReference:
    return ResolvedReference("Shape", CfgSectionValue({"x": DirectValue(value)}))


def nested_waveform_resource() -> tuple[CfgResource, Catalog]:
    rise = CfgSectionValue(
        {
            "style": DirectValue("gauss"),
            "length": DirectValue(0.4),
            "sigma": DirectValue(0.1),
        }
    )
    parent = CfgSectionValue(
        {
            "style": DirectValue("flat_top"),
            "length": DirectValue(1.0),
            "raise_waveform": ReferenceValue("rise", rise),
        }
    )
    catalog = Catalog(
        {
            "parent": ResolvedReference("FlatTop", parent),
            "rise": ResolvedReference("Gauss", rise),
        }
    )
    return waveform_resource(
        initial_style="flat_top", catalog=catalog, chosen_key="parent", value=parent
    ), catalog


def waveform_resource(
    *,
    initial_style: str = "gauss",
    catalog: Catalog | None = None,
    chosen_key: str | None = None,
    value: CfgSectionValue | None = None,
    resolution: Callable[[], CfgResolution] | None = None,
) -> CfgResource:
    policy = ProgramSpecPolicy()
    allowed = [shape.make_spec(policy) for shape in PROGRAM_SHAPES.waveforms()]
    old = PROGRAM_SHAPES.waveform(initial_style).make_spec(policy)
    initial = (
        value
        if value is not None
        else CfgSectionValue(
            {
                "style": DirectValue(initial_style),
                "length": DirectValue(0.8),
                "sigma": DirectValue(0.2),
            }
        )
    )
    schema = CfgSchema(
        CfgSectionSpec(
            fields={"ref": ReferenceSpec("waveform", allowed, discriminator="style")}
        ),
        CfgSectionValue(
            {"ref": ReferenceValue(chosen_key or f"<Custom:{old.label}>", initial)}
        ),
    )
    return CfgResource(
        lambda: schema,
        resolution=resolution or (catalog or Catalog()).snapshot,
    )
