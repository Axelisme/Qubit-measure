"""Original registered models in complete documents, with registry custody."""

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal, Self

import pytest
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    ValidatorFunctionWrapHandler,
    field_validator,
    model_validator,
)
from ruamel.yaml import YAML
from zcu_tools.format_version import YamlMap
from zcu_tools.resources.entry import (
    ComponentSchema,
    ResultEntry,
    component_registry,
)
from zcu_tools.resources.entry.views import FieldView


class RequiredPhysicalSchema(ComponentSchema):
    model_config = ConfigDict(extra="forbid")
    rate: float
    title: str


class RequiredTiming(BaseModel):
    model_config = ConfigDict(extra="forbid")
    width: float
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


class OptionalPairSchema(ComponentSchema):
    links: PairLinks | None = None


@contextmanager
def registered_model(kind: str, model: type[ComponentSchema]) -> Generator[str]:
    component_registry.register(kind, model)
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


def create_entry(tmp_path: Path) -> tuple[ResultEntry, Path, Path]:
    results, database = tmp_path / "results", tmp_path / "Database"
    entry = ResultEntry.create("entry", result_root=results, database_root=database)
    return entry, results, database


def field_view(value: object) -> FieldView:
    """Require an editable container from a public component or child read."""
    assert isinstance(value, FieldView)
    return value


@pytest.mark.parametrize("owner", ["setup", "point"])
@pytest.mark.parametrize("version", ["1.0", "1.2"])
def test_allow_extras_support_child_views_set_meta_seed_and_reload(
    tmp_path: Path,
    registry_state_guard: None,
    owner: str,
    version: str,
) -> None:
    class Extensible(ComponentSchema):
        model_config = ConfigDict(extra="allow")
        title: str

    with registered_model("notebook/extensible", Extensible) as kind:
        entry, results, database = create_entry(tmp_path)
        entry.setup.add_component("N1", kind=kind, title="ready", gain=2)
        view = entry.setup if owner == "setup" else entry.new_point("working")
        source = (
            results / "entry/setup.yaml"
            if owner == "setup"
            else results / "entry/points/working/point.yaml"
        )
        yaml = YAML(typ="rt")
        stored = yaml.load(source)
        stored["format_version"] = version
        stored["components"]["N1"]["future"] = {"level": 3, "untouched": 4}
        yaml.dump(stored, source)
        before = source.read_bytes()
        view.refresh()
        assert source.read_bytes() == before
        assert view.N1.gain == 2
        assert view.meta("N1.future.level") is None
        future = field_view(view.N1.future)
        assert future["level"] == 3
        view.N1.gain = 5
        assert view.N1.gain == 5
        future["level"] = 6
        with view.edit() as draft:
            draft.set("N1.future.level", 7)
            draft.set("N1.gain", 8)
        accepted = view.meta("N1.future.level")
        assert accepted is not None and accepted.source == "manual"
        gain_source = view.meta("N1.gain")
        assert gain_source is not None and gain_source.source == "manual"
        assert future.level == 7 and future.untouched == 4
        if owner == "setup":
            seeded = entry.new_point("seeded")
            assert seeded.N1.gain == 8
            assert seeded.meta("N1.future.level") == accepted
            field_view(seeded.N1.future).level = 9
            assert future.level == 7
        else:
            assert entry.setup.N1.gain == 2
        reopened = ResultEntry.open(
            "entry", result_root=results, database_root=database
        )
        loaded = reopened.setup if owner == "setup" else reopened.use_point("working")
        assert loaded.N1.gain == 8
        assert field_view(loaded.N1.future).level == 7
        assert loaded.meta("N1.future.level") == accepted
        assert YAML(typ="safe").load(source)["format_version"] == version


@pytest.mark.parametrize("owner", ["setup", "point"])
@pytest.mark.parametrize("version", ["1.0", "1.2"])
def test_registered_legacy_rename_preserves_source_until_explicit_acceptance(
    tmp_path: Path,
    registry_state_guard: None,
    owner: str,
    version: str,
) -> None:
    class Renamed(ComponentSchema):
        model_config = ConfigDict(extra="forbid")
        modern: int

        @model_validator(mode="before")
        @classmethod
        def rename_legacy(cls, value: YamlMap) -> YamlMap:
            converted = dict(value)
            if "legacy" in converted:
                converted["modern"] = converted.pop("legacy")
            return converted

    with registered_model("notebook/renamed", Renamed) as kind:
        entry, results, database = create_entry(tmp_path)
        entry.setup.add_component("N1", kind=kind, modern=7)
        view = entry.setup if owner == "setup" else entry.new_point("working")
        source = (
            results / "entry/setup.yaml"
            if owner == "setup"
            else results / "entry/points/working/point.yaml"
        )
        original_source = view.meta("N1.modern")
        assert original_source is not None
        yaml = YAML(typ="rt")
        stored = yaml.load(source)
        stored["format_version"] = version
        component = stored["components"]["N1"]
        component["legacy"] = component.pop("modern")
        if version == "1.2":
            component["future"] = {"quality": "next"}
        yaml.dump(stored, source)
        before = source.read_bytes()
        reopened = ResultEntry.open(
            "entry", result_root=results, database_root=database
        )
        loaded = reopened.setup if owner == "setup" else reopened.use_point("working")
        assert loaded.N1.modern == 7
        assert loaded.meta("N1.modern") == original_source
        assert loaded.meta("N1.legacy") is None
        with pytest.raises(AttributeError):
            _ = loaded.N1.future
        loaded.refresh()
        with loaded.edit():
            pass
        assert source.read_bytes() == before
        loaded.description = "unrelated commit"
        saved = YAML(typ="safe").load(source)
        assert saved["components"]["N1"]["modern"] == 7
        assert "legacy" not in saved["components"]["N1"]
        if version == "1.2":
            assert saved["components"]["N1"]["future"] == {"quality": "next"}
        assert loaded.meta("N1.modern") == original_source
        assert loaded.meta("N1.legacy") is None
        with loaded.edit() as draft:
            draft.set("N1.modern", 8)
        accepted = loaded.meta("N1.modern")
        assert accepted is not None and accepted.source == "manual"
        assert accepted.at != original_source.at
        with pytest.raises(ValidationError), loaded.edit() as draft:
            draft.set("N1.modren", 9)
        assert loaded.N1.modern == 8
        assert loaded.meta("N1.modern") == accepted


def test_nested_model_views_follow_commit_and_refresh_without_leaking_drafts(
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
        entry, results, database = create_entry(tmp_path)
        entry.setup.add_component("N1", kind=kind, wiring={"link": {"index": 2}})
        wiring = entry.setup.N1.wiring
        assert isinstance(wiring, FieldView)
        pin = field_view(wiring.link)
        assert pin.index == 2
        with entry.setup.edit() as draft:
            draft_wiring = field_view(draft.N1.wiring)
            draft_pin = field_view(draft_wiring.link)
            draft_pin.index = 5
            assert draft_pin.index == 5
            assert pin.index == 2
        assert pin.index == 5
        source = entry.setup.meta("N1.wiring.link.index")
        assert source is not None and source.source == "manual"
        other = ResultEntry.open("entry", result_root=results, database_root=database)
        with other.setup.edit() as draft:
            draft.set("N1.wiring.link.index", 8)
        assert pin.index == 5
        entry.setup.refresh()
        assert pin.index == 8


def test_plain_typed_dict_views_edit_seed_and_reload_without_resolving_paths(
    tmp_path: Path,
) -> None:
    class Programmed(ComponentSchema):
        programs: dict[str, str] = Field(default_factory=dict)

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
        source = point.meta("N1.programs.drive")
        before = point_slots.drive
        with pytest.raises(ValidationError):
            point_slots.drive = 3
        assert point_slots.drive == before
        assert point.meta("N1.programs.drive") == source
        reopened = ResultEntry.open(
            "entry", result_root=results, database_root=database
        )
        reopened_slots = reopened.use_point("working").N1.programs
        assert isinstance(reopened_slots, FieldView)
        assert reopened_slots.sense == "unresolved.sense"
        assert slots.drive == "another.path"


@pytest.mark.parametrize("owner", ["setup", "point"])
@pytest.mark.parametrize("invalid", [-1, None, True, "3", 3.5])
def test_recursive_dict_views_preserve_models_constraints_and_sources(
    tmp_path: Path,
    owner: str,
    invalid: object,
) -> None:
    class Pin(BaseModel):
        model_config = ConfigDict(extra="forbid")
        index: int = Field(strict=True, ge=0)

    class Route(BaseModel):
        model_config = ConfigDict(extra="forbid")
        pin: Pin

    class Routed(ComponentSchema):
        routes: dict[str, Route]
        levels: dict[str, dict[str, int]]

    with registered_model("notebook/routed", Routed) as kind:
        entry, results, database = create_entry(tmp_path)
        entry.setup.add_component(
            "N1",
            kind=kind,
            routes={"drive": {"pin": {"index": 2}}},
            levels={"gain": {"left": 1, "right": 2}},
        )
        view = entry.setup if owner == "setup" else entry.new_point("working")
        routes = field_view(view.N1.routes)
        route = field_view(routes["drive"])
        pin = field_view(route.pin)
        pin["index"] = 3
        source = view.meta("N1.routes.drive.pin.index")
        assert source is not None and source.source == "manual"
        with view.edit() as draft:
            draft_routes = field_view(draft.N1.routes)
            draft_route = field_view(draft_routes.drive)
            draft_pin = field_view(draft_route.pin)
            with pytest.raises(ValidationError):
                draft_pin.index = invalid
            assert draft_pin.index == 3
            levels = field_view(draft.N1.levels)
            gain = field_view(levels["gain"])
            gain["left"] = 5
            assert gain.right == 2
            assert pin.index == 3
            assert view.meta("N1.routes.drive.pin.index") == source
        assert pin.index == 3
        assert view.meta("N1.routes.drive.pin.index") == source
        gain_source = view.meta("N1.levels.gain.left")
        assert gain_source is not None and gain_source.source == "manual"
        if owner == "point":
            setup_routes = field_view(entry.setup.N1.routes)
            setup_route = field_view(setup_routes.drive)
            setup_pin = field_view(setup_route.pin)
            assert setup_pin.index == 2
        reopened = ResultEntry.open(
            "entry", result_root=results, database_root=database
        )
        loaded = reopened.setup if owner == "setup" else reopened.use_point("working")
        with loaded.edit() as draft:
            draft.set("N1.routes.drive.pin.index", 6)
        loaded_routes = field_view(loaded.N1.routes)
        loaded_route = field_view(loaded_routes.drive)
        loaded_pin = field_view(loaded_route.pin)
        assert loaded_pin.index == 6
        loaded_levels = field_view(loaded.N1.levels)
        loaded_gain = field_view(loaded_levels.gain)
        assert loaded_gain.left == 5


@pytest.mark.parametrize("owner", ["setup", "point"])
def test_child_writes_validate_the_complete_parent_and_preserve_rejected_sources(
    tmp_path: Path,
    owner: str,
) -> None:
    class Bounds(BaseModel):
        model_config = ConfigDict(extra="forbid")
        low: int
        high: int

    class Bounded(ComponentSchema):
        box: Bounds
        initialized: bool = False

        def model_post_init(self, context: object) -> None:
            self.initialized = True

        @model_validator(mode="after")
        def check_bounds(self) -> Self:
            if self.box.low > self.box.high:
                raise ValueError("low must not exceed high")
            return self

    with registered_model("notebook/bounded", Bounded) as kind:
        entry, results, database = create_entry(tmp_path)
        entry.setup.add_component("N1", kind=kind, box={"low": 1, "high": 5})
        view = entry.setup if owner == "setup" else entry.new_point("working")
        assert view.N1.initialized is True
        box = field_view(view.N1.box)
        source = view.meta("N1.box.low")
        with pytest.raises(ValidationError, match="low must not exceed high"):
            box.low = 7
        assert box.low == 1
        assert view.meta("N1.box.low") == source
        with view.edit() as draft:
            candidate = field_view(draft.N1.box)
            with pytest.raises(ValidationError, match="low must not exceed high"):
                candidate["low"] = 7
            assert candidate.low == 1
            candidate.high = 9
            draft.set("N1.box.low", 8)
            assert candidate.low == 8
            assert box.low == 1
        assert box.low == 8 and box.high == 9
        accepted = view.meta("N1.box.low")
        assert accepted is not None and accepted.source == "manual"
        reopened = ResultEntry.open(
            "entry", result_root=results, database_root=database
        )
        loaded = reopened.setup if owner == "setup" else reopened.use_point("working")
        loaded_box = field_view(loaded.N1.box)
        assert loaded_box.low == 8 and loaded_box.high == 9
        assert loaded.N1.initialized is True


@pytest.mark.parametrize(
    "write", ["automatic", "attribute", "set", pytest.param("ext", id="whole-field")]
)
def test_leaf_write_runs_owning_field_normalization(tmp_path: Path, write: str) -> None:
    class NormalizeExt(ComponentSchema):
        ext: YamlMap = Field(default_factory=dict)

        @field_validator("ext")
        @classmethod
        def normalize_title(cls, value: YamlMap) -> YamlMap:
            value["title"] = str(value["title"]).strip().lower()
            return value

    with registered_model("notebook/normalize-ext", NormalizeExt) as kind:
        entry, results, database = create_entry(tmp_path)
        entry.setup.add_component("N1", kind=kind, ext={"title": "prepared"})
        if write == "automatic":
            field_view(entry.setup.N1.ext).title = "  Changed  "
        else:
            with entry.setup.edit() as draft:
                if write == "attribute":
                    field_view(draft.N1.ext).title = "  Changed  "
                elif write == "set":
                    draft.set("N1.ext.title", "  Changed  ")
                else:
                    setattr(draft.N1, write, {"title": "  Changed  "})
        assert field_view(entry.setup.N1.ext).title == "changed"
        reopened = ResultEntry.open(
            "entry", result_root=results, database_root=database
        )
        assert field_view(reopened.setup.N1.ext).title == "changed"


@pytest.mark.parametrize("write", ["attribute", "set"])
def test_caught_owning_field_failure_preserves_draft(
    tmp_path: Path, write: str
) -> None:
    class RejectTitle(ComponentSchema):
        ext: YamlMap = Field(default_factory=dict)

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
                    field_view(draft.N1.ext).title = "invalid"
            else:
                with pytest.raises(ValidationError, match="title rejected"):
                    draft.set("N1.ext.title", "invalid")
            assert field_view(draft.N1.ext).title == "prepared"
        reopened = ResultEntry.open(
            "entry", result_root=results, database_root=database
        )
        assert reopened.setup.description == "keep successful edit"
        assert field_view(reopened.setup.N1.ext).title == "prepared"


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
            assert (
                field_view(draft.N1.timing).label == "changed"
                and field_view(draft.N1.timing).width == 1.0
            )
        reopened = ResultEntry.open(
            "entry", result_root=results, database_root=database
        )
        assert (
            field_view(reopened.setup.N1.timing).label == "changed"
            and field_view(reopened.setup.N1.timing).width == 1.0
        )


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
        assert field_view(entry.setup.N1.details).kind == "initial"
        source = results / "entry" / "setup.yaml"
        document = YAML(typ="safe").load(source.read_text(encoding="utf-8"))
        assert document["components"]["N1"]["details"] == {"kind": "initial"}

        with entry.setup.edit() as draft:
            with pytest.raises(ValidationError, match="string_type"):
                draft.set("N1.details.kind", None)
            assert field_view(draft.N1.details).kind == "initial"
            draft.set("N1.details.kind", "auxiliary")
        reopened = ResultEntry.open(
            "entry", result_root=results, database_root=database
        )
        assert field_view(reopened.setup.N1.details).kind == "auxiliary"
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
        ext: YamlMap = Field(default_factory=dict)

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
            payload: YamlMap = {"fail": True}
            if write == "set":
                with pytest.raises(ValidationError, match="mutable payload rejected"):
                    draft.set("N1.ext", payload)
            else:
                with pytest.raises(ValidationError, match="mutable payload rejected"):
                    setattr(draft.N1, write, payload)
            assert payload == {"fail": True}
            assert field_view(field_view(draft.N1.ext).payload).fail is True
            draft.description = "keep successful edit"
        reopened = ResultEntry.open(
            "entry", result_root=results, database_root=database
        )
        assert reopened.setup.description == "keep successful edit"
        assert field_view(field_view(reopened.setup.N1.ext).payload).fail is True


def test_nullable_numeric_field_round_trip(tmp_path: Path) -> None:
    class NullableFrequency(ComponentSchema):
        rate: float | None = None

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


def test_nullable_nested_rounding_publishes_normalized_values(
    tmp_path: Path, registry_state_guard: None
) -> None:
    class RoundedTiming(BaseModel):
        model_config = ConfigDict(extra="forbid")
        rate: float

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
        assert field_view(entry.setup.N1.timing).rate == 0.14
        assert YAML(typ="safe").load(source)["components"]["N1"]["timing"] == {
            "rate": 0.14
        }
        reopened = ResultEntry.open(
            "entry", result_root=results, database_root=database
        )
        reopened.setup.refresh()
        assert field_view(reopened.setup.N1.timing).rate == 0.14
        reopened.setup.N1.timing = None
        assert reopened.setup.N1.timing is None
        assert YAML(typ="safe").load(source)["components"]["N1"]["timing"] is None


def test_numeric_rounding_survives_reload(
    tmp_path: Path, registry_state_guard: None
) -> None:
    class RoundedSchema(ComponentSchema):
        rate: float

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
        assert (
            field_view(entry.setup.N1.timing).label == "prepared"
            and field_view(entry.setup.N1.timing).width == 1.0
        )
        with entry.setup.edit() as draft:
            with pytest.raises(ValidationError, match="width"):
                draft.set("N1.timing.width", value)
            assert (
                field_view(draft.N1.timing).label == "prepared"
                and field_view(draft.N1.timing).width == 1.0
            )
            draft.set("N1.timing.width", 10.0)
        assert (
            field_view(entry.setup.N1.timing).label == "prepared"
            and field_view(entry.setup.N1.timing).width == 10.0
        )


def test_optional_nested_strings_accept_null_and_round_trip_without_reference_checks(
    tmp_path: Path,
) -> None:
    with registered_model("notebook/optional-pair", OptionalPairSchema) as kind:
        entry, results, database = create_entry(tmp_path)
        entry.setup.add_component("P0", kind=kind, links=None)
        assert entry.setup.P0.links is None
        entry.setup.add_component(
            "P1", kind=kind, links={"control": "first", "target": "second"}
        )
        with entry.setup.edit() as draft:
            draft.set("P1.links.target", "unresolved")
        reopened = ResultEntry.open(
            "entry", result_root=results, database_root=database
        )
        links = field_view(reopened.setup.P1.links)
        assert links.control == "first" and links.target == "unresolved"
        reopened.setup.P1.links = None
        again = ResultEntry.open("entry", result_root=results, database_root=database)
        assert again.setup.P1.links is None


@pytest.mark.parametrize("operation", ["add", "attribute", "set"])
def test_optional_nested_typos_preserve_values_sources_and_disk(
    tmp_path: Path, operation: str
) -> None:
    with registered_model("notebook/optional-timing", OptionalNestedSchema) as kind:
        entry, results, _database = create_entry(tmp_path)
        entry.setup.add_component(
            "N1", kind=kind, timing={"label": "prepared", "width": 1.0}
        )
        setup_path = results / "entry" / "setup.yaml"
        before = setup_path.read_bytes()
        accepted = entry.setup.meta("N1.timing.width")
        assert accepted is not None

        def perform_operation() -> None:
            if operation == "add":
                entry.setup.add_component("N2", kind=kind, timing={"widht": 10.0})
            elif operation == "attribute":
                entry.setup.N1.timing = {"widht": 10.0}
            else:
                with entry.setup.edit() as draft:
                    draft.set("N1.timing.widht", 10.0)

        with pytest.raises(ValidationError):
            perform_operation()
        assert setup_path.read_bytes() == before
        timing = field_view(entry.setup.N1.timing)
        assert timing.label == "prepared" and timing.width == 1.0
        assert entry.setup.meta("N1.timing.width") == accepted


def test_optional_nested_complete_values_round_trip_in_working_units(
    tmp_path: Path,
) -> None:
    with registered_model("notebook/optional-timing", OptionalNestedSchema) as kind:
        entry, results, database = create_entry(tmp_path)
        entry.setup.add_component(
            "N1", kind=kind, timing={"label": "prepared", "width": 1.0}
        )
        assert (
            field_view(entry.setup.N1.timing).label == "prepared"
            and field_view(entry.setup.N1.timing).width == 1.0
        )
        with entry.setup.edit() as draft:
            draft.set("N1.timing.width", 10.0)
            assert (
                field_view(draft.N1.timing).label == "prepared"
                and field_view(draft.N1.timing).width == 10.0
            )

        setup_path = results / "entry" / "setup.yaml"
        assert YAML(typ="safe").load(setup_path)["components"]["N1"]["timing"][
            "width"
        ] == pytest.approx(10.0)
        reopened = ResultEntry.open(
            "entry", result_root=results, database_root=database
        )
        assert (
            field_view(reopened.setup.N1.timing).label == "prepared"
            and field_view(reopened.setup.N1.timing).width == 10.0
        )
        reopened.setup.N1.timing = {"label": "updated", "width": 20.0}
        assert YAML(typ="safe").load(setup_path)["components"]["N1"]["timing"][
            "width"
        ] == pytest.approx(20.0)
        again = ResultEntry.open("entry", result_root=results, database_root=database)
        assert (
            field_view(again.setup.N1.timing).label == "updated"
            and field_view(again.setup.N1.timing).width == 20.0
        )


def test_nested_model_views_and_dotted_edits_preserve_draft_isolation(
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
        assert (
            field_view(draft.N1.timing).width == 10.0
            and field_view(draft.N1.timing).label == "prepared"
        )
        assert (
            field_view(entry.setup.N1.timing).label == "prepared"
            and field_view(entry.setup.N1.timing).width == 1.0
        )
        assert setup_path.read_bytes() == before

    reopened = ResultEntry.open("entry", result_root=results, database_root=database)
    assert (
        field_view(reopened.setup.N1.timing).width == 10.0
        and field_view(reopened.setup.N1.timing).label == "prepared"
    )
    assert YAML(typ="safe").load(setup_path)["components"]["N1"]["timing"]["width"] == (
        pytest.approx(10.0)
    )


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
