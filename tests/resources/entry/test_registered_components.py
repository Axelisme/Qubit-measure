"""Registered notebook models in partial setup documents, with registry custody."""

from collections.abc import Generator, Sequence
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Annotated, Literal, Self

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
from zcu_tools.resources.document_store import UnitSpec
from zcu_tools.resources.entry import (
    ComponentSchema,
    MissingReferenceError,
    ResultEntry,
    UnknownFieldError,
    component_registry,
)


class RequiredPhysicalSchema(ComponentSchema):
    freq: Annotated[float, UnitSpec("Hz", "MHz")]
    title: str


class RequiredTiming(BaseModel):
    model_config = ConfigDict(extra="forbid")
    width: Annotated[float, UnitSpec("s", "us")]
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


@pytest.fixture(scope="module", autouse=True)
def registry_module_guard() -> Generator[None]:
    before = deepcopy(vars(component_registry))
    yield
    assert vars(component_registry) == before, (
        "component registry polluted by this module"
    )


@pytest.fixture(autouse=True)
def registry_state_guard(
    request: pytest.FixtureRequest,
) -> Generator[None]:
    before = deepcopy(vars(component_registry))
    yield
    assert vars(component_registry) == before, (
        f"registry polluter: {request.node.nodeid}"
    )


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


@pytest.mark.parametrize(
    ("initial", "delta", "accepted"),
    [(1.0, 5e-13, True), (1.0, 2e-12, False), (0.0, 5e-15, False)],
)
def test_unitspec_float_drift_has_relative_tolerance_without_absolute_tolerance(
    tmp_path: Path,
    registry_state_guard: None,
    initial: float,
    delta: float,
    accepted: bool,
) -> None:
    class ShiftedSchema(ComponentSchema):
        freq: Annotated[float, UnitSpec("Hz", "MHz")]

        @field_validator("freq")
        @classmethod
        def shift_frequency(cls, value: float) -> float:
            return value + delta

    with registered_model("notebook/shifted", ShiftedSchema) as kind:
        entry, results, database = create_entry(tmp_path)
        source = results / "entry" / "setup.yaml"
        before = source.read_bytes()
        if accepted:
            entry.setup.add_component("N1", kind=kind, freq=initial)
            assert entry.setup.N1.freq == pytest.approx(
                1.000000000001, rel=1e-15, abs=0.0
            )
            reopened = ResultEntry.open(
                "entry", result_root=results, database_root=database
            )
            assert reopened.setup.N1.freq == entry.setup.N1.freq
        else:
            with pytest.raises(ValidationError) as failure:
                entry.setup.add_component("N1", kind=kind, freq=initial)
            assert failure.value.errors()[0]["loc"] == ("components", "N1", "freq")
            assert source.read_bytes() == before
            with pytest.raises(AttributeError, match="Unknown component 'N1'"):
                _ = entry.setup.N1


def test_unitspec_integer_output_keeps_exact_canonical_comparison(
    tmp_path: Path, registry_state_guard: None
) -> None:
    class IntegerShiftedSchema(ComponentSchema):
        freq: Annotated[int, UnitSpec("Hz", "Hz")]

        @field_validator("freq")
        @classmethod
        def shift_frequency(cls, value: int) -> int:
            return value + 1

    with registered_model("notebook/integer-shift", IntegerShiftedSchema) as kind:
        entry, results, _ = create_entry(tmp_path)
        source = results / "entry" / "setup.yaml"
        before = source.read_bytes()
        with pytest.raises(ValidationError) as failure:
            entry.setup.add_component("N1", kind=kind, freq=1_000_000_000_000_000)
        assert failure.value.errors()[0]["loc"] == ("components", "N1", "freq")
        assert source.read_bytes() == before
        with pytest.raises(AttributeError, match="Unknown component 'N1'"):
            _ = entry.setup.N1


def test_nullable_nested_rounding_publishes_working_canonical_values(
    tmp_path: Path, registry_state_guard: None
) -> None:
    class RoundedTiming(BaseModel):
        model_config = ConfigDict(extra="forbid")
        freq: Annotated[float, UnitSpec("Hz", "MHz")]

        @field_validator("freq")
        @classmethod
        def round_frequency(cls, value: float) -> float:
            return round(value, 2)

    class RoundedSchema(ComponentSchema):
        timing: RoundedTiming | None = None

    with registered_model("notebook/nested-rounded", RoundedSchema) as kind:
        entry, results, database = create_entry(tmp_path)
        entry.setup.add_component("N1", kind=kind, timing={"freq": 0.143})
        source = results / "entry" / "setup.yaml"
        assert entry.setup.N1.timing == {"freq": 0.14}
        assert YAML(typ="safe").load(source)["components"]["N1"]["timing"] == {
            "freq": 140000.0
        }
        reopened = ResultEntry.open(
            "entry", result_root=results, database_root=database
        )
        reopened.setup.refresh()
        assert reopened.setup.N1.timing == {"freq": 0.14}
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


def test_idempotent_numeric_rounding_survives_si_round_trip_and_publishes_canonical(
    tmp_path: Path, registry_state_guard: None
) -> None:
    class RoundedSchema(ComponentSchema):
        freq: Annotated[float, UnitSpec("Hz", "MHz")]

        @field_validator("freq")
        @classmethod
        def round_frequency(cls, value: float) -> float:
            return round(value, 2)

    with registered_model("notebook/rounded", RoundedSchema) as kind:
        entry, results, database = create_entry(tmp_path)
        entry.setup.add_component("N1", kind=kind, freq=0.143)
        source = results / "entry" / "setup.yaml"
        assert YAML(typ="safe").load(source)["components"]["N1"]["freq"] == 140000.0
        assert entry.setup.N1.freq == 0.14
        reopened = ResultEntry.open(
            "entry", result_root=results, database_root=database
        )
        reopened.setup.refresh()
        assert reopened.setup.N1.freq == 0.14
        with reopened.setup.edit() as draft:
            assert draft.N1.freq == 0.14
            draft.description = "rounded value retained"
        assert reopened.setup.N1.freq == 0.14
        assert reopened.setup.description == "rounded value retained"
        assert YAML(typ="safe").load(source)["components"]["N1"]["freq"] == 140000.0


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
        entry.setup.add_component("R1", kind="resonator", freq=10.0)
        entry.setup.add_component("N1", kind=kind, timing={})
        source = results / "entry" / "setup.yaml"
        before = source.read_bytes()

        def edit() -> None:
            with entry.setup.edit() as draft:
                draft.description = "must roll back"
                draft.R1.freq = 15.0
                draft.set("N1.timing.width", 10.0)

        with pytest.raises(ValidationError) as failure:
            edit()
        error = failure.value.errors()[0]
        assert error["loc"] == ("components", "N1", "timing", "width")
        assert "20.0" in error["msg"] and "40.0" in error["msg"]
        assert source.read_bytes() == before
        assert entry.setup.R1.freq == 10.0
        assert entry.setup.N1.timing == {}
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
        entry.setup.add_component("R1", kind="resonator", freq=10.0)
        source = results / "entry" / "setup.yaml"
        before = source.read_bytes()
        with pytest.raises(ValidationError) as failure:
            entry.setup.add_component("N1", kind=kind, title="prepared")
        error = failure.value.errors()[0]
        assert error["loc"] == ("components", "N1", "title")
        assert "prepared!" in error["msg"] and "prepared!!" in error["msg"]
        assert source.read_bytes() == before
        assert entry.setup.R1.freq == 10.0
        with pytest.raises(AttributeError, match="Unknown component 'N1'"):
            _ = entry.setup.N1


@pytest.mark.parametrize("mode", ["before", "after", "wrap", "plain"])
def test_partial_setup_runs_supplied_field_validators_and_skips_invalid_defaults(
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
        entry.setup.add_component("N1", kind=kind)
        source = results / "entry" / "setup.yaml"
        assert "title" not in YAML(typ="safe").load(source)["components"]["N1"]
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


@pytest.mark.parametrize("nullable", [False, True])
def test_partial_nested_setup_defers_model_after_validators(
    tmp_path: Path,
    registry_state_guard: None,
    nullable: bool,
) -> None:
    class TimingSchema(RequiredTiming):
        @model_validator(mode="after")
        def check_width(self) -> Self:
            if self.width <= 0:
                raise ValueError("timing width must be positive")
            return self

    class NestedSchema(ComponentSchema):
        timing: TimingSchema

    class NullableSchema(ComponentSchema):
        timing: TimingSchema | None = None

    model = NullableSchema if nullable else NestedSchema
    with registered_model("notebook/nested-invariant", model) as kind:
        entry, results, database = create_entry(tmp_path)
        entry.setup.add_component("N1", kind=kind, timing={"label": "draft"})
        with entry.setup.edit() as draft:
            draft.set("N1.timing.width", -2.0)
        reopened = ResultEntry.open(
            "entry", result_root=results, database_root=database
        )
        reopened.setup.refresh()
        assert reopened.setup.N1.timing == {"label": "draft", "width": -2.0}
        persisted = YAML(typ="safe").load(results / "entry" / "setup.yaml")
        assert persisted["components"]["N1"]["timing"]["width"] == pytest.approx(-2e-6)
        with pytest.raises(ValidationError, match="timing width"):
            component_registry.get(kind).model_validate(
                {"kind": kind, "timing": {"label": "draft", "width": -2e-6}}
            )


def test_partial_setup_defers_model_invariants_but_preserves_field_constraints(
    tmp_path: Path,
    registry_state_guard: None,
) -> None:
    class LinewidthSchema(ComponentSchema):
        freq: Annotated[float, UnitSpec("Hz", "MHz"), Field(gt=0)]
        kappa: Annotated[float, UnitSpec("Hz", "MHz")]

        @model_validator(mode="after")
        def check_linewidth(self) -> Self:
            if self.kappa >= self.freq:
                raise ValueError("linewidth must be below frequency")
            return self

    with registered_model("notebook/linewidth", LinewidthSchema) as kind:
        entry, results, database = create_entry(tmp_path)
        entry.setup.add_component("R1", kind=kind)
        entry.setup.R1.freq = 5000.0
        entry.setup.R1.kappa = 6000.0
        with entry.setup.edit() as draft:
            draft.set("R1.freq", 4900.0)
        reopened = ResultEntry.open(
            "entry", result_root=results, database_root=database
        )
        reopened.setup.refresh()
        assert reopened.setup.R1.freq == pytest.approx(4900.0)
        assert reopened.setup.R1.kappa == pytest.approx(6000.0)
        source = results / "entry" / "setup.yaml"
        persisted = YAML(typ="safe").load(source)
        assert persisted["components"]["R1"]["freq"] == pytest.approx(4.9e9)
        assert persisted["components"]["R1"]["kappa"] == pytest.approx(6e9)
        before = source.read_bytes()
        with pytest.raises(ValidationError, match="freq"):
            reopened.setup.R1.freq = -1.0
        assert source.read_bytes() == before
        assert reopened.setup.R1.freq == pytest.approx(4900.0)
        original_model = component_registry.get(kind)
        with pytest.raises(ValidationError, match="linewidth"):
            original_model.model_validate({"kind": kind, "freq": 4.9e9, "kappa": 6e9})


@pytest.mark.parametrize("value", ["not a width", None])
def test_optional_container_does_not_make_a_supplied_required_leaf_nullable(
    tmp_path: Path, value: str | None
) -> None:
    with registered_model("notebook/optional-timing", OptionalNestedSchema) as kind:
        entry, results, _database = create_entry(tmp_path)
        entry.setup.add_component("N1", kind=kind, timing={"label": "prepared"})
        setup_path = results / "entry" / "setup.yaml"
        before = setup_path.read_bytes()
        with pytest.raises(ValidationError, match="width"):
            entry.setup.N1.timing = {"label": "prepared", "width": value}
        assert setup_path.read_bytes() == before
        assert entry.setup.N1.timing == {"label": "prepared"}
        with entry.setup.edit() as draft:
            with pytest.raises(ValidationError, match="width"):
                draft.set("N1.timing.width", value)
            assert draft.N1.timing == {"label": "prepared"}
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
        entry.setup.add_component("Q1", kind="qubit/transmon")
        entry.setup.add_component("Q2", kind="qubit/transmon")
        entry.setup.add_component("P0", kind=kind, links=None)
        assert entry.setup.P0.links is None
        entry.setup.add_component("P1", kind=kind, links={"control": "Q1"})
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
        entry.setup.add_component("N1", kind=kind, timing={"label": "prepared"})
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
        assert entry.setup.N1.timing == {"label": "prepared"}


def test_optional_nested_setup_fields_defer_missing_values_and_round_trip_units(
    tmp_path: Path,
) -> None:
    with registered_model("notebook/optional-timing", OptionalNestedSchema) as kind:
        entry, results, database = create_entry(tmp_path)
        entry.setup.add_component("N1", kind=kind, timing={"label": "prepared"})
        assert entry.setup.N1.timing == {"label": "prepared"}
        with entry.setup.edit() as draft:
            draft.set("N1.timing.width", 10.0)
            assert draft.N1.timing == {"label": "prepared", "width": 10.0}

        setup_path = results / "entry" / "setup.yaml"
        assert YAML(typ="safe").load(setup_path)["components"]["N1"]["timing"][
            "width"
        ] == pytest.approx(1e-5)
        reopened = ResultEntry.open(
            "entry", result_root=results, database_root=database
        )
        assert reopened.setup.N1.timing == {"label": "prepared", "width": 10.0}
        reopened.setup.N1.timing = {"label": "updated", "width": 20.0}
        assert YAML(typ="safe").load(setup_path)["components"]["N1"]["timing"][
            "width"
        ] == pytest.approx(2e-5)
        again = ResultEntry.open("entry", result_root=results, database_root=database)
        assert again.setup.N1.timing == {"label": "updated", "width": 20.0}


def test_nested_reference_path_failure_discards_the_shared_draft(
    tmp_path: Path, pair_kind: str
) -> None:
    entry, results, _database = create_entry(tmp_path)
    entry.setup.add_component("Q1", kind="qubit/transmon")
    entry.setup.add_component("Q2", kind="qubit/fluxonium")
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
    entry.setup.add_component("N1", kind=nested_kind, timing={"label": "prepared"})
    setup_path = results / "entry" / "setup.yaml"
    before = setup_path.read_bytes()
    with entry.setup.edit() as draft:
        draft.set("N1.timing.width", 10.0)
        assert draft.N1.timing == {"width": 10.0, "label": "prepared"}
        assert entry.setup.N1.timing == {"label": "prepared"}
        assert setup_path.read_bytes() == before

    reopened = ResultEntry.open("entry", result_root=results, database_root=database)
    assert reopened.setup.N1.timing == {"width": 10.0, "label": "prepared"}
    assert YAML(typ="safe").load(setup_path)["components"]["N1"]["timing"]["width"] == (
        pytest.approx(1e-5)
    )


@pytest.mark.parametrize("field", ["control", "target", "coupler"])
@pytest.mark.parametrize("operation", ["add", "write", "open", "refresh"])
def test_nested_references_reject_missing_targets_with_the_declared_path(
    tmp_path: Path, pair_kind: str, field: str, operation: str
) -> None:
    entry, results, database = create_entry(tmp_path)
    setup_path = results / "entry" / "setup.yaml"
    entry.setup.add_component("Q1", kind="qubit/transmon")
    entry.setup.add_component("Q2", kind="qubit/fluxonium")
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


def test_nested_required_references_can_be_filled_incrementally_and_reopened(
    tmp_path: Path, pair_kind: str
) -> None:
    entry, results, database = create_entry(tmp_path)
    entry.setup.add_component("Q1", kind="qubit/transmon")
    entry.setup.add_component("Q2", kind="qubit/fluxonium")
    entry.setup.add_component("P1", kind=pair_kind, links={"control": "Q1"})
    setup_path = results / "entry" / "setup.yaml"
    assert YAML(typ="safe").load(setup_path)["components"]["P1"]["links"] == {
        "control": "Q1"
    }
    entry.setup.P1.links = {"control": "Q1", "target": "Q2", "coupler": None}
    reopened = ResultEntry.open("entry", result_root=results, database_root=database)
    assert reopened.setup.P1.kind == pair_kind
    assert YAML(typ="safe").load(setup_path)["components"]["P1"]["links"] == {
        "control": "Q1",
        "target": "Q2",
        "coupler": None,
    }


@pytest.mark.parametrize("value", [None, "not a frequency"])
def test_partial_setup_still_validates_supplied_required_values(
    tmp_path: Path,
    required_kind: str,
    value: str | None,
) -> None:
    entry, results, _ = create_entry(tmp_path)
    entry.setup.add_component("N1", kind=required_kind)
    setup_file = results / "entry" / "setup.yaml"
    before = setup_file.read_bytes()
    with pytest.raises(ValidationError, match="freq"):
        entry.setup.N1.freq = value
    assert setup_file.read_bytes() == before
    with pytest.raises(AttributeError, match="not set"):
        _ = entry.setup.N1.freq


def test_nested_required_fields_are_deferred_and_preserve_nested_unit_metadata(
    tmp_path: Path,
    nested_kind: str,
) -> None:
    entry, results, database = create_entry(tmp_path)
    entry.setup.add_component("N1", kind=nested_kind, timing={"label": "prepared"})
    document = YAML(typ="safe").load(results / "entry" / "setup.yaml")
    assert "width" not in document["components"]["N1"]["timing"]

    entry.setup.N1.timing = {"width": 10.0, "label": "prepared"}
    reopened = ResultEntry.open("entry", result_root=results, database_root=database)
    assert reopened.setup.N1.kind == nested_kind
    document = YAML(typ="safe").load(results / "entry" / "setup.yaml")
    assert document["components"]["N1"]["timing"]["width"] == pytest.approx(1e-5)


def test_required_registered_fields_can_be_filled_incrementally_in_setup(
    tmp_path: Path,
    required_kind: str,
) -> None:
    entry, results, database = create_entry(tmp_path)
    entry.setup.add_component("N1", kind=required_kind)
    with pytest.raises(AttributeError, match="not set"):
        _ = entry.setup.N1.freq
    document = YAML(typ="safe").load(results / "entry" / "setup.yaml")
    assert "freq" not in document["components"]["N1"]
    assert "title" not in document["components"]["N1"]

    entry.setup.N1.freq = 5000.0
    entry.setup.N1.title = "prepared later"
    reopened = ResultEntry.open("entry", result_root=results, database_root=database)
    assert reopened.setup.N1.freq == pytest.approx(5000.0)
    assert reopened.setup.N1.title == "prepared later"
    document = YAML(typ="safe").load(results / "entry" / "setup.yaml")
    assert document["components"]["N1"]["freq"] == pytest.approx(5e9)
    assert component_registry.get(required_kind) is RequiredPhysicalSchema
