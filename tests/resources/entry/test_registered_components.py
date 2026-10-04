"""Original registered models in complete documents, with registry custody."""

from collections.abc import Generator, Sequence
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Annotated, Literal

import pytest
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    ValidatorFunctionWrapHandler,
    field_validator,
)
from ruamel.yaml import YAML
from zcu_tools.format_version import YamlMap
from zcu_tools.resources.entry import (
    ComponentRegistry,
    ComponentSchema,
    MissingReferenceError,
    ModuleSlot,
    Ref,
    ResultEntry,
    RoleSpec,
    UnitSpec,
    UnknownFieldError,
    component_registry,
)
from zcu_tools.resources.entry.views import FieldView


class RequiredPhysicalSchema(ComponentSchema):
    rate: Annotated[float, UnitSpec("MHz")]
    title: str


class RequiredTiming(BaseModel):
    model_config = ConfigDict(extra="forbid")
    width: Annotated[float, UnitSpec("µs")]
    label: str


class RequiredNestedSchema(ComponentSchema):
    timing: RequiredTiming


class OptionalNestedSchema(ComponentSchema):
    timing: RequiredTiming | None = None


class PairLinks(BaseModel):
    model_config = ConfigDict(extra="forbid")
    control: str
    target: str
    coupler: str | None = None


class PairSchema(ComponentSchema):
    links: PairLinks


class OptionalPairSchema(ComponentSchema):
    links: PairLinks | None = None


@contextmanager
def registered_model(
    kind: str, model: type[ComponentSchema], *, references: Sequence[str] = ()
) -> Generator[str]:
    component_registry.register(kind, model, references=references)
    try:
        yield kind
    finally:
        component_registry.unregister(kind)


@pytest.fixture
def required_kind(registry_state_guard: None) -> Generator[str]:
    with registered_model("notebook/required", RequiredPhysicalSchema) as kind:
        yield kind


@pytest.fixture
def nested_kind(registry_state_guard: None) -> Generator[str]:
    with registered_model("notebook/nested", RequiredNestedSchema) as kind:
        yield kind


@pytest.fixture
def pair_kind(registry_state_guard: None) -> Generator[str]:
    with registered_model(
        "notebook/pair",
        PairSchema,
        references=("links.control", "links.target", "links.coupler"),
    ) as kind:
        yield kind


def create_entry(tmp_path: Path) -> tuple[ResultEntry, Path, Path]:
    results, database = tmp_path / "results", tmp_path / "Database"
    entry = ResultEntry.create("entry", result_root=results, database_root=database)
    return entry, results, database


def test_nested_wiring_view_returns_maps_and_preserves_snapshot_aliases(
    tmp_path: Path,
) -> None:
    class Pin(BaseModel):
        model_config = ConfigDict(extra="forbid")
        index: int = Field(strict=True, ge=0)

    class Pins(BaseModel):
        model_config = ConfigDict(extra="forbid")
        link: Pin

    class Wired(ComponentSchema):
        wiring: Pins

    with registered_model("notebook/wired", Wired) as kind:
        entry, _, _ = create_entry(tmp_path)
        entry.setup.add_component("N1", kind=kind, wiring={"link": {"index": 2}})
        value = entry.setup.N1.wiring.link
        assert value == {"index": 2}
        assert isinstance(value, dict)
        value["index"] = 99
        assert entry.setup.N1.wiring.link == {"index": 2}
        with entry.setup.edit() as draft:
            draft.set("N1.wiring.link.index", 5)
            assert draft.N1.wiring.link == {"index": 5}
            assert entry.setup.N1.wiring.link == {"index": 2}
        entry.setup.refresh()
        assert entry.setup.N1.wiring.link == {"index": 5}


def test_marked_module_slots_edit_seed_and_reload_without_resolving_paths(
    tmp_path: Path,
) -> None:
    class Programmed(ComponentSchema):
        programs: Annotated[dict[str, str], ModuleSlot()] = Field(default_factory=dict)

    with registered_model("notebook/programmed", Programmed) as kind:
        entry, results, database = create_entry(tmp_path)
        entry.setup.add_component(
            "N1", kind=kind, programs={"drive": "unresolved.drive"}
        )
        slots = entry.setup.N1.programs
        assert isinstance(slots, FieldView)
        assert slots.drive == "unresolved.drive"
        slots.drive = "another.path"
        point = entry.new_point("working")
        with point.edit() as draft:
            draft.set("N1.programs.sense", "unresolved.sense")
        point_slots = point.N1.programs
        assert isinstance(point_slots, FieldView)
        assert point_slots.drive == "another.path"
        assert point_slots.sense == "unresolved.sense"
        with pytest.raises(ValidationError):
            point_slots.drive = 3
        assert point_slots.drive == "another.path"
        reopened = ResultEntry.open(
            "entry", result_root=results, database_root=database
        )
        reopened_slots = reopened.use_point("working").N1.programs
        assert isinstance(reopened_slots, FieldView)
        assert reopened_slots.sense == "unresolved.sense"
        assert slots.drive == "another.path"


def test_nullable_branch_reference_validates_missing_target(tmp_path: Path) -> None:
    class Linked(ComponentSchema):
        link: Annotated[str, Ref()] | None = None

    with registered_model("notebook/nullable-branch-link", Linked) as kind:
        entry, results, _database = create_entry(tmp_path)
        before = (results / "entry/setup.yaml").read_bytes()
        with pytest.raises(MissingReferenceError) as error:
            entry.setup.add_component("L1", kind=kind, link="missing")
        assert error.value.field == "link"
        assert (results / "entry/setup.yaml").read_bytes() == before


def test_nullable_branch_numeric_reference_is_rejected_before_reserving_kind() -> None:
    class InvalidRef(ComponentSchema):
        link: Annotated[int, Ref()] | None = None

    registry = ComponentRegistry()
    with pytest.raises(ValueError, match="reference"):
        registry.register("notebook/branch-ref", InvalidRef)
    registry.register("notebook/branch-ref", ComponentSchema)
    assert registry.get("notebook/branch-ref") is ComponentSchema


def test_nullable_branch_module_slot_is_rejected_before_reserving_kind() -> None:
    class InvalidSlot(ComponentSchema):
        programs: Annotated[dict[str, str], ModuleSlot()] | None = None

    registry = ComponentRegistry()
    with pytest.raises(TypeError, match="ModuleSlot requires"):
        registry.register("notebook/branch-slot", InvalidSlot)
    registry.register("notebook/branch-slot", ComponentSchema)
    assert registry.get("notebook/branch-slot") is ComponentSchema


@pytest.mark.parametrize("inner_marker", [False, True], ids=["outer", "inner"])
def test_marked_reference_validates_add_write_and_reload(
    tmp_path: Path, inner_marker: bool
) -> None:
    class OuterLinked(ComponentSchema):
        link: Annotated[str | None, Ref()] = None

    class InnerLinked(ComponentSchema):
        link: Annotated[str, Ref()] | None = None

    model = InnerLinked if inner_marker else OuterLinked
    with registered_model("notebook/marked-link", model) as kind:
        entry, results, database = create_entry(tmp_path)
        before = (results / "entry/setup.yaml").read_bytes()
        with pytest.raises(MissingReferenceError) as error:
            entry.setup.add_component("L1", kind=kind, link="missing")
        assert error.value.field == "link"
        assert (results / "entry/setup.yaml").read_bytes() == before
        entry.setup.add_component("T1", kind=kind)
        entry.setup.add_component("L1", kind=kind, link="T1")
        point = entry.new_point("working")
        with pytest.raises(MissingReferenceError):
            point.L1.link = "missing"
        assert point.L1.link == "T1"
        with pytest.raises(MissingReferenceError), point.edit() as draft:
            draft.set("L1.link", "missing")
        assert point.L1.link == "T1"
        with point.edit() as draft:
            draft.set("L1.link", None)
        assert point.L1.link is None
        point.L1.link = "T1"
        reopened = ResultEntry.open(
            "entry", result_root=results, database_root=database
        )
        assert reopened.use_point("working").L1.link == "T1"
        roles = component_registry.roles
        with ExitStack() as cleanup:
            for name in ("marker_source", "link", "marker_target"):
                roles.register(name, RoleSpec(kind))
                cleanup.callback(roles.unregister, name)
            same_name = point.resolve(["marker_source", "link"], marker_source="L1")
            assert same_name.components == {"marker_source": "L1", "link": "T1"}
            via = point.resolve(
                {
                    "marker_source": RoleSpec(kind),
                    "marker_target": RoleSpec(kind, via="marker_source.link"),
                },
                marker_source="L1",
            )
            assert via.components == {"marker_source": "L1", "marker_target": "T1"}
        path = results / "entry/points/working/point.yaml"
        yaml = YAML(typ="safe")
        document = yaml.load(path.read_text())
        document["components"]["L1"]["link"] = "missing"
        with path.open("w") as stream:
            yaml.dump(document, stream)
        before = path.read_bytes()
        with pytest.raises(MissingReferenceError):
            point.refresh()
        assert point.L1.link == "T1"
        with pytest.raises(MissingReferenceError):
            reopened.use_point("working")
        assert path.read_bytes() == before


def test_nullable_nested_markers_validate_reference_and_accept_null(
    tmp_path: Path,
) -> None:
    class Links(BaseModel):
        model_config = ConfigDict(extra="forbid")
        target: Annotated[str | None, Ref()] = None

    class Linked(ComponentSchema):
        links: Links | None = None

    with registered_model("notebook/nested-links", Linked) as kind:
        entry, _, _ = create_entry(tmp_path)
        entry.setup.add_component("T1", kind=kind, links=None)
        entry.setup.add_component("N1", kind=kind, links={"target": "T1"})
        point = entry.new_point("working")
        with pytest.raises(MissingReferenceError) as error:
            point.N1.links = {"target": "missing"}
        assert error.value.field == "links.target"
        assert point.N1.links == {"target": "T1"}
        with point.edit() as draft:
            draft.set("N1.links.target", None)
        point.refresh()
        assert point.N1.links == {"target": None}
        point.N1.links = None
        point.refresh()
        assert point.N1.links is None


def test_overridden_extension_annotation_keeps_json_boundary(tmp_path: Path) -> None:
    class Extended(ComponentSchema):
        ext: YamlMap = Field(default_factory=dict)

    with registered_model("notebook/extensions", Extended) as kind:
        entry, _, _ = create_entry(tmp_path)
        with pytest.raises(ValidationError):
            entry.setup.add_component("N1", kind=kind, ext={"bad": float("inf")})
        entry.setup.add_component("N1", kind=kind, ext={"valid": [True, None]})
        assert entry.setup.N1.ext.valid == [True, None]


@pytest.mark.parametrize(
    "write", ["automatic", "attribute", "set", pytest.param("ext", id="whole-field")]
)
def test_leaf_write_runs_owning_field_normalization(tmp_path: Path, write: str) -> None:
    class NormalizeExt(ComponentSchema):
        @field_validator("ext")
        @classmethod
        def normalize_title(cls, value: YamlMap) -> YamlMap:
            value["title"] = str(value["title"]).strip().lower()
            return value

    with registered_model("notebook/normalize-ext", NormalizeExt) as kind:
        entry, results, database = create_entry(tmp_path)
        entry.setup.add_component("N1", kind=kind, ext={"title": "prepared"})
        if write == "automatic":
            entry.setup.N1.ext.title = "  Changed  "
        else:
            with entry.setup.edit() as draft:
                if write == "attribute":
                    draft.N1.ext.title = "  Changed  "
                elif write == "set":
                    draft.set("N1.ext.title", "  Changed  ")
                else:
                    setattr(draft.N1, write, {"title": "  Changed  "})
        assert entry.setup.N1.ext.title == "changed"
        reopened = ResultEntry.open(
            "entry", result_root=results, database_root=database
        )
        assert reopened.setup.N1.ext.title == "changed"


@pytest.mark.parametrize("write", ["attribute", "set"])
def test_caught_owning_field_failure_preserves_draft(
    tmp_path: Path, write: str
) -> None:
    class RejectTitle(ComponentSchema):
        @field_validator("ext")
        @classmethod
        def reject_title(cls, value: YamlMap) -> YamlMap:
            if value.get("title") == "invalid":
                raise ValueError("title rejected")
            return value

    with registered_model("notebook/reject-title", RejectTitle) as kind:
        entry, results, database = create_entry(tmp_path)
        entry.setup.add_component("N1", kind=kind, ext={"title": "prepared"})
        with entry.setup.edit() as draft:
            draft.description = "keep successful edit"
            if write == "attribute":
                with pytest.raises(ValidationError, match="title rejected"):
                    draft.N1.ext.title = "invalid"
            else:
                with pytest.raises(ValidationError, match="title rejected"):
                    draft.set("N1.ext.title", "invalid")
            assert draft.N1.ext.title == "prepared"
        reopened = ResultEntry.open(
            "entry", result_root=results, database_root=database
        )
        assert reopened.setup.description == "keep successful edit"
        assert reopened.setup.N1.ext.title == "prepared"


def test_path_write_runs_nested_owning_field_normalization(tmp_path: Path) -> None:
    class NormalizeTiming(ComponentSchema):
        timing: RequiredTiming

        @field_validator("timing")
        @classmethod
        def normalize_label(cls, value: RequiredTiming) -> RequiredTiming:
            value.label = value.label.strip().lower()
            return value

    with registered_model("notebook/normalize-timing", NormalizeTiming) as kind:
        entry, results, database = create_entry(tmp_path)
        entry.setup.add_component(
            "N1", kind=kind, timing={"label": "prepared", "width": 1.0}
        )
        with entry.setup.edit() as draft:
            draft.set("N1.timing.label", "  Changed  ")
            assert draft.N1.timing == {"label": "changed", "width": 1.0}
        reopened = ResultEntry.open(
            "entry", result_root=results, database_root=database
        )
        assert reopened.setup.N1.timing == {"label": "changed", "width": 1.0}


@pytest.mark.parametrize("nullable", [False, True])
def test_nested_kind_is_required_in_complete_components(
    tmp_path: Path, nullable: bool
) -> None:
    class Details(BaseModel):
        model_config = ConfigDict(extra="forbid")
        kind: str

    class DirectDetails(ComponentSchema):
        details: Details

    class NullableDetails(ComponentSchema):
        details: Details | None

    model = NullableDetails if nullable else DirectDetails
    with registered_model("notebook/nested-kind", model) as kind:
        entry, results, database = create_entry(tmp_path)
        with pytest.raises(ValidationError, match="kind"):
            entry.setup.add_component("N1", kind=kind, details={})
        entry.setup.add_component("N1", kind=kind, details={"kind": "initial"})
        assert entry.setup.N1.details == {"kind": "initial"}
        source = results / "entry" / "setup.yaml"
        document = YAML(typ="safe").load(source.read_text(encoding="utf-8"))
        assert document["components"]["N1"]["details"] == {"kind": "initial"}

        with entry.setup.edit() as draft:
            with pytest.raises(ValidationError, match="string_type"):
                draft.set("N1.details.kind", None)
            assert draft.N1.details == {"kind": "initial"}
            draft.set("N1.details.kind", "auxiliary")
        reopened = ResultEntry.open(
            "entry", result_root=results, database_root=database
        )
        assert reopened.setup.N1.details == {"kind": "auxiliary"}
        if nullable:
            entry.setup.N1.details = None
            reopened.setup.refresh()
            assert reopened.setup.N1.details is None


@pytest.mark.parametrize("validate_assignment", [False, True])
@pytest.mark.parametrize("write", ["set", "ext"])
def test_caught_alias_validation_failure_preserves_shared_draft(
    tmp_path: Path, write: str, validate_assignment: bool
) -> None:
    class RejectMutableExt(ComponentSchema):
        model_config = ConfigDict(
            extra="forbid", validate_assignment=validate_assignment
        )

        @field_validator("ext", mode="before")
        @classmethod
        def mutate_then_reject(cls, value: YamlMap) -> YamlMap:
            if value.get("fail") is True:
                value["touched"] = True
                raise ValueError("mutable payload rejected")
            return value

    with registered_model("notebook/reject-mutable-ext", RejectMutableExt) as kind:
        entry, results, database = create_entry(tmp_path)
        entry.setup.add_component("N1", kind=kind, ext={"payload": {"fail": True}})
        with entry.setup.edit() as draft:
            payload = draft.N1.ext.payload
            if write == "set":
                with pytest.raises(ValidationError, match="mutable payload rejected"):
                    draft.set("N1.ext", payload)
            else:
                with pytest.raises(ValidationError, match="mutable payload rejected"):
                    setattr(draft.N1, write, payload)
            assert draft.N1.ext.payload == {"fail": True}
            draft.description = "keep successful edit"
        reopened = ResultEntry.open(
            "entry", result_root=results, database_root=database
        )
        assert reopened.setup.description == "keep successful edit"
        assert reopened.setup.N1.ext.payload == {"fail": True}


def test_nullable_branch_unit_round_trip(tmp_path: Path) -> None:
    class NullableFrequency(ComponentSchema):
        rate: Annotated[float, UnitSpec("MHz")] | None = None

    with registered_model("notebook/nullable-unit", NullableFrequency) as kind:
        entry, results, database = create_entry(tmp_path)
        entry.setup.add_component("N1", kind=kind, rate=10.0)
        source = results / "entry" / "setup.yaml"
        document = YAML(typ="safe").load(source.read_text(encoding="utf-8"))
        assert document["components"]["N1"]["rate"] == 10.0
        assert entry.setup.N1.rate == 10.0

        entry.setup.N1.rate = 12.0
        reopened = ResultEntry.open(
            "entry", result_root=results, database_root=database
        )
        assert reopened.setup.N1.rate == 12.0
        document = YAML(typ="safe").load(source.read_text(encoding="utf-8"))
        assert document["components"]["N1"]["rate"] == 12.0

        entry.setup.N1.rate = None
        reopened = ResultEntry.open(
            "entry", result_root=results, database_root=database
        )
        assert reopened.setup.N1.rate is None
        document = YAML(typ="safe").load(source.read_text(encoding="utf-8"))
        assert document["components"]["N1"]["rate"] is None


def test_unitspec_integer_output_keeps_exact_canonical_comparison(
    tmp_path: Path, registry_state_guard: None
) -> None:
    class IntegerShiftedSchema(ComponentSchema):
        rate: Annotated[int, UnitSpec("Hz")]

        @field_validator("rate")
        @classmethod
        def shift_frequency(cls, value: int) -> int:
            return value + 1

    with registered_model("notebook/integer-shift", IntegerShiftedSchema) as kind:
        entry, results, _ = create_entry(tmp_path)
        source = results / "entry" / "setup.yaml"
        before = source.read_bytes()
        with pytest.raises(ValidationError) as failure:
            entry.setup.add_component("N1", kind=kind, rate=1_000_000_000_000_000)
        assert failure.value.errors()[0]["loc"] == ("components", "N1", "rate")
        assert source.read_bytes() == before
        with pytest.raises(AttributeError, match="Unknown component 'N1'"):
            _ = entry.setup.N1


def test_nullable_nested_rounding_publishes_working_canonical_values(
    tmp_path: Path, registry_state_guard: None
) -> None:
    class RoundedTiming(BaseModel):
        model_config = ConfigDict(extra="forbid")
        rate: Annotated[float, UnitSpec("MHz")]

        @field_validator("rate")
        @classmethod
        def round_frequency(cls, value: float) -> float:
            return round(value, 2)

    class RoundedSchema(ComponentSchema):
        timing: RoundedTiming | None = None

    with registered_model("notebook/nested-rounded", RoundedSchema) as kind:
        entry, results, database = create_entry(tmp_path)
        entry.setup.add_component("N1", kind=kind, timing={"rate": 0.143})
        source = results / "entry" / "setup.yaml"
        assert entry.setup.N1.timing == {"rate": 0.14}
        assert YAML(typ="safe").load(source)["components"]["N1"]["timing"] == {
            "rate": 0.14
        }
        reopened = ResultEntry.open(
            "entry", result_root=results, database_root=database
        )
        reopened.setup.refresh()
        assert reopened.setup.N1.timing == {"rate": 0.14}
        reopened.setup.N1.timing = None
        assert reopened.setup.N1.timing is None
        assert YAML(typ="safe").load(source)["components"]["N1"]["timing"] is None


def test_in_place_field_conversion_cannot_mutate_away_canonical_drift(
    tmp_path: Path, registry_state_guard: None
) -> None:
    class StampedExtensionSchema(ComponentSchema):
        @field_validator("ext", mode="before")
        @classmethod
        def stamp_title(cls, value: YamlMap) -> YamlMap:
            title = value["title"]
            if not isinstance(title, str):
                raise TypeError("title must be a string")
            value["title"] = title + "!"
            return value

    with registered_model("notebook/in-place-stamp", StampedExtensionSchema) as kind:
        entry, results, _ = create_entry(tmp_path)
        source = results / "entry" / "setup.yaml"
        before = source.read_bytes()
        with pytest.raises(ValidationError) as failure:
            entry.setup.add_component("N1", kind=kind, ext={"title": "prepared"})
        error = failure.value.errors()[0]
        assert error["loc"] == ("components", "N1", "ext")
        assert "prepared!" in error["msg"] and "prepared!!" in error["msg"]
        assert source.read_bytes() == before
        with pytest.raises(AttributeError, match="Unknown component 'N1'"):
            _ = entry.setup.N1


def test_unitless_extension_structure_keeps_exact_canonical_comparison(
    tmp_path: Path, registry_state_guard: None
) -> None:
    class ShiftedExtensionSchema(ComponentSchema):
        @field_validator("ext")
        @classmethod
        def shift_gain(cls, value: YamlMap) -> YamlMap:
            gain = value["gain"]
            if not isinstance(gain, float):
                raise TypeError("gain must be a float")
            return {**value, "gain": gain + 5e-13}

    with registered_model("notebook/extension", ShiftedExtensionSchema) as kind:
        entry, results, _ = create_entry(tmp_path)
        source = results / "entry" / "setup.yaml"
        before = source.read_bytes()
        with pytest.raises(ValidationError) as failure:
            entry.setup.add_component("N1", kind=kind, ext={"gain": 1.0})
        assert failure.value.errors()[0]["loc"] == ("components", "N1", "ext")
        assert source.read_bytes() == before
        with pytest.raises(AttributeError, match="Unknown component 'N1'"):
            _ = entry.setup.N1


def test_idempotent_numeric_rounding_survives_reload_and_publishes_canonical(
    tmp_path: Path, registry_state_guard: None
) -> None:
    class RoundedSchema(ComponentSchema):
        rate: Annotated[float, UnitSpec("MHz")]

        @field_validator("rate")
        @classmethod
        def round_frequency(cls, value: float) -> float:
            return round(value, 2)

    with registered_model("notebook/rounded", RoundedSchema) as kind:
        entry, results, database = create_entry(tmp_path)
        entry.setup.add_component("N1", kind=kind, rate=0.143)
        source = results / "entry" / "setup.yaml"
        assert YAML(typ="safe").load(source)["components"]["N1"]["rate"] == 0.14
        assert entry.setup.N1.rate == 0.14
        reopened = ResultEntry.open(
            "entry", result_root=results, database_root=database
        )
        reopened.setup.refresh()
        assert reopened.setup.N1.rate == 0.14
        with reopened.setup.edit() as draft:
            assert draft.N1.rate == 0.14
            draft.description = "rounded value retained"
        assert reopened.setup.N1.rate == 0.14
        assert reopened.setup.description == "rounded value retained"
        assert YAML(typ="safe").load(source)["components"]["N1"]["rate"] == 0.14


def test_non_idempotent_nested_numeric_edit_discards_the_whole_transaction(
    tmp_path: Path, registry_state_guard: None
) -> None:
    class DoubledTiming(RequiredTiming):
        @field_validator("width")
        @classmethod
        def double_width(cls, value: float) -> float:
            return value * 2

    class DoubledSchema(ComponentSchema):
        timing: DoubledTiming

    with registered_model("notebook/doubled", DoubledSchema) as kind:
        entry, results, _ = create_entry(tmp_path)
        entry.setup.add_component("R1", kind="fake/sensor", rate=10.0)
        entry.setup.add_component(
            "N1", kind=kind, timing={"width": 0.0, "label": "initial"}
        )
        source = results / "entry" / "setup.yaml"
        before = source.read_bytes()

        def edit() -> None:
            with entry.setup.edit() as draft:
                draft.description = "must roll back"
                draft.R1.rate = 15.0
                draft.set("N1.timing.width", 10.0)

        with pytest.raises(ValidationError) as failure:
            edit()
        error = failure.value.errors()[0]
        assert error["loc"] == ("components", "N1", "timing", "width")
        assert "20.0" in error["msg"] and "40.0" in error["msg"]
        assert source.read_bytes() == before
        assert entry.setup.R1.rate == 10.0
        assert entry.setup.N1.timing == {"width": 0.0, "label": "initial"}
        assert entry.setup.description is None


@pytest.mark.parametrize("operation", ["open", "refresh", "edit"])
def test_noncanonical_file_rejects_reload_with_source_and_preserves_snapshot(
    tmp_path: Path,
    registry_state_guard: None,
    operation: Literal["open", "refresh", "edit"],
) -> None:
    class NormalizedSchema(ComponentSchema):
        title: str

        @field_validator("title")
        @classmethod
        def normalize_title(cls, value: str) -> str:
            return value.strip().lower()

    with registered_model("notebook/normalized", NormalizedSchema) as kind:
        entry, results, database = create_entry(tmp_path)
        entry.setup.add_component("N1", kind=kind, title="prepared")
        source = results / "entry" / "setup.yaml"
        yaml = YAML(typ="rt")
        document = yaml.load(source)
        document["components"]["N1"]["title"] = "  Changed  "
        yaml.dump(document, source)
        before = source.read_bytes()
        body_entered = False

        def reload() -> None:
            nonlocal body_entered
            if operation == "open":
                ResultEntry.open("entry", result_root=results, database_root=database)
            elif operation == "refresh":
                entry.setup.refresh()
            else:
                with entry.setup.edit() as draft:
                    body_entered = True
                    draft.description = "must not commit"

        with pytest.raises(ValidationError) as failure:
            reload()
        error = failure.value.errors()[0]
        assert error["loc"] == ("components", "N1", "title")
        assert "  Changed  " in error["msg"] and "changed" in error["msg"]
        assert str(source) in error["msg"]
        assert not body_entered
        assert source.read_bytes() == before
        assert entry.setup.N1.title == "prepared"
        assert entry.setup.description is None


@pytest.mark.parametrize("mode", ["before", "after", "wrap", "plain"])
def test_non_idempotent_field_conversion_rejects_add_without_publishing(
    tmp_path: Path,
    registry_state_guard: None,
    mode: Literal["before", "after", "wrap", "plain"],
) -> None:
    class StampedSchema(ComponentSchema):
        title: str

        if mode == "wrap":

            @field_validator("title", mode="wrap")
            @classmethod
            def stamp_wrapped_title(
                cls, value: object, handler: ValidatorFunctionWrapHandler
            ) -> str:
                return str(handler(value)) + "!"
        else:

            @field_validator("title", mode=mode)
            @classmethod
            def stamp_title(cls, value: str) -> str:
                return value + "!"

    with registered_model("notebook/stamped", StampedSchema) as kind:
        entry, results, _ = create_entry(tmp_path)
        entry.setup.add_component("R1", kind="fake/sensor", rate=10.0)
        source = results / "entry" / "setup.yaml"
        before = source.read_bytes()
        with pytest.raises(ValidationError) as failure:
            entry.setup.add_component("N1", kind=kind, title="prepared")
        error = failure.value.errors()[0]
        assert error["loc"] == ("components", "N1", "title")
        assert "prepared!" in error["msg"] and "prepared!!" in error["msg"]
        assert source.read_bytes() == before
        assert entry.setup.R1.rate == 10.0
        with pytest.raises(AttributeError, match="Unknown component 'N1'"):
            _ = entry.setup.N1


@pytest.mark.parametrize("mode", ["before", "after", "wrap", "plain"])
def test_original_model_runs_field_validators_and_validates_defaults(
    tmp_path: Path,
    registry_state_guard: None,
    mode: Literal["before", "after", "wrap", "plain"],
) -> None:
    def normalize(value: object) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("title must contain text")
        return value.strip().lower()

    class NormalizedSchema(ComponentSchema):
        model_config = ConfigDict(extra="forbid", validate_default=True)
        title: str = Field(default="", validate_default=True)

        if mode == "wrap":

            @field_validator("title", mode="wrap")
            @classmethod
            def normalize_wrapped_title(
                cls, value: object, handler: ValidatorFunctionWrapHandler
            ) -> str:
                return normalize(handler(value))
        else:

            @field_validator("title", mode=mode)
            @classmethod
            def normalize_title(cls, value: object) -> str:
                return normalize(value)

    with registered_model("notebook/normalized", NormalizedSchema) as kind:
        entry, results, database = create_entry(tmp_path)
        source = results / "entry" / "setup.yaml"
        before_add = source.read_bytes()
        with pytest.raises(ValidationError, match="title"):
            entry.setup.add_component("N1", kind=kind)
        assert source.read_bytes() == before_add
        entry.setup.add_component("N1", kind=kind, title="  Initial  ")
        assert entry.setup.N1.title == "initial"
        with entry.setup.edit() as draft:
            draft.set("N1.title", "  Prepared  ")
        reopened = ResultEntry.open(
            "entry", result_root=results, database_root=database
        )
        reopened.setup.refresh()
        assert reopened.setup.N1.title == "prepared"
        assert YAML(typ="safe").load(source)["components"]["N1"]["title"] == "prepared"
        before = source.read_bytes()
        with pytest.raises(ValidationError, match="title"):
            reopened.setup.N1.title = " "
        assert source.read_bytes() == before
        assert reopened.setup.N1.title == "prepared"
        with pytest.raises(ValidationError, match="title"):
            component_registry.get(kind).model_validate({"kind": kind})


@pytest.mark.parametrize("value", ["not a width", None])
def test_optional_container_does_not_make_a_supplied_required_leaf_nullable(
    tmp_path: Path, value: str | None
) -> None:
    with registered_model("notebook/optional-timing", OptionalNestedSchema) as kind:
        entry, results, _database = create_entry(tmp_path)
        entry.setup.add_component(
            "N1", kind=kind, timing={"label": "prepared", "width": 1.0}
        )
        setup_path = results / "entry" / "setup.yaml"
        before = setup_path.read_bytes()
        with pytest.raises(ValidationError, match="width"):
            entry.setup.N1.timing = {"label": "prepared", "width": value}
        assert setup_path.read_bytes() == before
        assert entry.setup.N1.timing == {"label": "prepared", "width": 1.0}
        with entry.setup.edit() as draft:
            with pytest.raises(ValidationError, match="width"):
                draft.set("N1.timing.width", value)
            assert draft.N1.timing == {"label": "prepared", "width": 1.0}
            draft.set("N1.timing.width", 10.0)
        assert entry.setup.N1.timing == {"label": "prepared", "width": 10.0}


def test_optional_nested_references_validate_supplied_targets_and_allow_null(
    tmp_path: Path,
) -> None:
    with registered_model(
        "notebook/optional-pair",
        OptionalPairSchema,
        references=("links.control", "links.target", "links.coupler"),
    ) as kind:
        entry, results, database = create_entry(tmp_path)
        entry.setup.add_component("Q1", kind="fake/drive/a")
        entry.setup.add_component("Q2", kind="fake/drive/a")
        entry.setup.add_component("P0", kind=kind, links=None)
        assert entry.setup.P0.links is None
        entry.setup.add_component(
            "P1", kind=kind, links={"control": "Q1", "target": "Q1"}
        )
        with entry.setup.edit() as draft:
            draft.set("P1.links.target", "Q2")
        reopened = ResultEntry.open(
            "entry", result_root=results, database_root=database
        )
        assert reopened.setup.P1.links == {"control": "Q1", "target": "Q2"}
        setup_path = results / "entry" / "setup.yaml"
        before = setup_path.read_bytes()
        with pytest.raises(MissingReferenceError) as failure:
            reopened.setup.P1.links = {"control": "Q1", "target": "absent"}
        assert failure.value.component == "P1"
        assert failure.value.field == "links.target"
        assert failure.value.target == "absent"
        assert setup_path.read_bytes() == before
        assert reopened.setup.P1.links == {"control": "Q1", "target": "Q2"}
        reopened.setup.P1.links = None
        again = ResultEntry.open("entry", result_root=results, database_root=database)
        assert again.setup.P1.links is None


@pytest.mark.parametrize("operation", ["add", "attribute", "set"])
def test_optional_nested_typos_keep_the_same_path_and_field_suggestion(
    tmp_path: Path, operation: str
) -> None:
    with registered_model("notebook/optional-timing", OptionalNestedSchema) as kind:
        entry, results, _database = create_entry(tmp_path)
        entry.setup.add_component(
            "N1", kind=kind, timing={"label": "prepared", "width": 1.0}
        )
        setup_path = results / "entry" / "setup.yaml"
        before = setup_path.read_bytes()

        def perform_operation() -> None:
            if operation == "add":
                entry.setup.add_component("N2", kind=kind, timing={"widht": 10.0})
            elif operation == "attribute":
                entry.setup.N1.timing = {"widht": 10.0}
            else:
                with entry.setup.edit() as draft:
                    draft.set("N1.timing.widht", 10.0)

        with pytest.raises(UnknownFieldError) as failure:
            perform_operation()
        name = "N2" if operation == "add" else "N1"
        assert failure.value.path == f"{name}.timing.widht"
        assert failure.value.field == "widht"
        assert "width" in failure.value.suggestions
        assert setup_path.read_bytes() == before
        assert entry.setup.N1.timing == {"label": "prepared", "width": 1.0}


def test_optional_nested_complete_values_round_trip_units(
    tmp_path: Path,
) -> None:
    with registered_model("notebook/optional-timing", OptionalNestedSchema) as kind:
        entry, results, database = create_entry(tmp_path)
        entry.setup.add_component(
            "N1", kind=kind, timing={"label": "prepared", "width": 1.0}
        )
        assert entry.setup.N1.timing == {"label": "prepared", "width": 1.0}
        with entry.setup.edit() as draft:
            draft.set("N1.timing.width", 10.0)
            assert draft.N1.timing == {"label": "prepared", "width": 10.0}

        setup_path = results / "entry" / "setup.yaml"
        assert YAML(typ="safe").load(setup_path)["components"]["N1"]["timing"][
            "width"
        ] == pytest.approx(10.0)
        reopened = ResultEntry.open(
            "entry", result_root=results, database_root=database
        )
        assert reopened.setup.N1.timing == {"label": "prepared", "width": 10.0}
        reopened.setup.N1.timing = {"label": "updated", "width": 20.0}
        assert YAML(typ="safe").load(setup_path)["components"]["N1"]["timing"][
            "width"
        ] == pytest.approx(20.0)
        again = ResultEntry.open("entry", result_root=results, database_root=database)
        assert again.setup.N1.timing == {"label": "updated", "width": 20.0}


def test_nested_reference_path_failure_discards_the_shared_draft(
    tmp_path: Path, pair_kind: str
) -> None:
    entry, results, _database = create_entry(tmp_path)
    entry.setup.add_component("Q1", kind="fake/drive/a")
    entry.setup.add_component("Q2", kind="fake/drive/b")
    links: YamlMap = {"control": "Q1", "target": "Q2"}
    entry.setup.add_component("P1", kind=pair_kind, links=links)
    setup_path = results / "entry" / "setup.yaml"
    before = setup_path.read_bytes()

    def perform_operation() -> None:
        with entry.setup.edit() as draft:
            draft.description = "discarded"
            draft.set("P1.links.target", "absent")

    with pytest.raises(MissingReferenceError) as failure:
        perform_operation()
    assert failure.value.field == "links.target"
    assert setup_path.read_bytes() == before
    assert entry.setup.description is None
    assert entry.setup.P1.links == links


def test_nested_model_values_use_yaml_maps_and_dotted_edits_in_working_units(
    tmp_path: Path, nested_kind: str
) -> None:
    entry, results, database = create_entry(tmp_path)
    entry.setup.add_component(
        "N1", kind=nested_kind, timing={"label": "prepared", "width": 1.0}
    )
    setup_path = results / "entry" / "setup.yaml"
    before = setup_path.read_bytes()
    with entry.setup.edit() as draft:
        draft.set("N1.timing.width", 10.0)
        assert draft.N1.timing == {"width": 10.0, "label": "prepared"}
        assert entry.setup.N1.timing == {"label": "prepared", "width": 1.0}
        assert setup_path.read_bytes() == before

    reopened = ResultEntry.open("entry", result_root=results, database_root=database)
    assert reopened.setup.N1.timing == {"width": 10.0, "label": "prepared"}
    assert YAML(typ="safe").load(setup_path)["components"]["N1"]["timing"]["width"] == (
        pytest.approx(10.0)
    )


@pytest.mark.parametrize("field", ["control", "target", "coupler"])
@pytest.mark.parametrize("operation", ["add", "write", "open", "refresh"])
def test_nested_references_reject_missing_targets_with_the_declared_path(
    tmp_path: Path, pair_kind: str, field: str, operation: str
) -> None:
    entry, results, database = create_entry(tmp_path)
    setup_path = results / "entry" / "setup.yaml"
    entry.setup.add_component("Q1", kind="fake/drive/a")
    entry.setup.add_component("Q2", kind="fake/drive/b")
    valid_links: YamlMap = {"control": "Q1", "target": "Q2"}
    entry.setup.add_component("P1", kind=pair_kind, links=valid_links)
    invalid_links = {**valid_links, field: "absent"}
    if operation in ("open", "refresh"):
        document = YAML(typ="safe").load(setup_path)
        document["components"]["P1"]["links"] = invalid_links
        with setup_path.open("w", encoding="utf-8") as stream:
            YAML(typ="rt").dump(document, stream)
    before = setup_path.read_bytes()

    def perform_operation() -> None:
        if operation == "add":
            entry.setup.add_component("P2", kind=pair_kind, links=invalid_links)
        elif operation == "write":
            entry.setup.P1.links = invalid_links
        elif operation == "refresh":
            entry.setup.refresh()
        else:
            ResultEntry.open("entry", result_root=results, database_root=database)

    with pytest.raises(MissingReferenceError) as failure:
        perform_operation()
    assert failure.value.source == setup_path
    assert failure.value.component == ("P2" if operation == "add" else "P1")
    assert failure.value.field == f"links.{field}"
    assert failure.value.target == "absent"
    assert setup_path.read_bytes() == before
    assert entry.setup.P1.links == valid_links
    if operation in ("add", "write"):
        entry.setup.P1.links = valid_links


@pytest.mark.parametrize("value", [None, "not a frequency"])
def test_required_component_values_reject_invalid_edits(
    tmp_path: Path,
    required_kind: str,
    value: str | None,
) -> None:
    entry, results, _ = create_entry(tmp_path)
    entry.setup.add_component("N1", kind=required_kind, rate=5000.0, title="initial")
    setup_file = results / "entry" / "setup.yaml"
    before = setup_file.read_bytes()
    with pytest.raises(ValidationError, match="rate"):
        entry.setup.N1.rate = value
    assert setup_file.read_bytes() == before
    assert entry.setup.N1.rate == 5000.0
