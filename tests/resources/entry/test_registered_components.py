"""Registered notebook models in partial setup documents, with registry custody."""

from collections.abc import Generator, Sequence
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Annotated

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError
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
