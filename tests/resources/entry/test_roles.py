"""Role resolution at the bound-point seam, without GUI/session wiring."""

from collections.abc import Generator, MutableMapping
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from pathlib import Path
from typing import cast

import pytest
from pydantic import BaseModel, ConfigDict
from zcu_tools.resources.entry import (
    ComponentSchema,
    PointView,
    ResultEntry,
    RoleRegistry,
    RoleResolutionError,
    RoleSpec,
    component_registry,
    role_registry,
)


@contextmanager
def registered_role(name: str, spec: RoleSpec) -> Generator[str]:
    role_registry.register(name, spec)
    try:
        yield name
    finally:
        role_registry.unregister(name)


def make_point(tmp_path: Path) -> PointView:
    entry = ResultEntry.create(
        "roles", result_root=tmp_path / "results", database_root=tmp_path / "database"
    )
    entry.setup.add_component("R1", kind="fake/sensor", rate=6500.0)
    entry.setup.add_component("R2", kind="fake/sensor", rate=7100.0)
    entry.setup.add_component("Q1", kind="fake/drive/a", rate=4300.0, sense="R1")
    entry.setup.add_component("Q2", kind="fake/drive/a", rate=5100.0, sense="R2")
    return entry.new_point("working")


class PairLinks(BaseModel):
    model_config = ConfigDict(extra="forbid")
    control: str
    target: str
    coupler: str


class PairSchema(ComponentSchema):
    links: PairLinks


@pytest.fixture
def pair_point(tmp_path: Path, registry_state_guard: None) -> Generator[PointView]:
    with ExitStack() as stack:
        component_registry.register(
            "notebook/pair",
            PairSchema,
            references=("links.control", "links.target", "links.coupler"),
        )
        stack.callback(component_registry.unregister, "notebook/pair")
        component_registry.register("coupler/notebook", ComponentSchema)
        stack.callback(component_registry.unregister, "coupler/notebook")
        for name, spec in {
            "pair": RoleSpec("notebook/pair"),
            "control_readout": RoleSpec("fake/sensor", via="control.sense"),
            "target_readout": RoleSpec("fake/sensor", via="target.sense"),
        }.items():
            stack.enter_context(registered_role(name, spec))
        point = make_point(tmp_path)
        point.add_component("C12", kind="coupler/notebook")
        point.add_component(
            "CZ12",
            kind="notebook/pair",
            links={"control": "Q1", "target": "Q2", "coupler": "C12"},
        )
        yield point


@pytest.mark.parametrize("declaration", ["explicit", "shorthand"])
def test_declared_kind_patterns_choose_unique_focus_and_derive_reference(
    tmp_path: Path, declaration: str
) -> None:
    class Beam(ComponentSchema):
        detector: str

    before = deepcopy(vars(role_registry))
    with ExitStack() as stack:
        stack.callback(vars(role_registry).update, before)
        for kind, model in {
            "beam/source": Beam,
            "beam/detector": ComponentSchema,
        }.items():
            component_registry.register(
                kind, model, references=("detector",) if model is Beam else ()
            )
            stack.callback(component_registry.unregister, kind)
        stack.enter_context(registered_role("beam", RoleSpec("beam/source")))
        stack.enter_context(registered_role("detector", RoleSpec("beam/detector")))
        role_registry.configure(
            shorthand=("beam", "detector"), focus_kinds=("beam/source",)
        )
        entry = ResultEntry.create(
            "beam",
            result_root=tmp_path / "results",
            database_root=tmp_path / "database",
        )
        entry.setup.add_component("Sensor", kind="beam/detector")
        entry.setup.add_component("Emitter", kind="beam/source", detector="Sensor")
        point = entry.new_point("working")
        resolved = point.resolve(
            ["beam", "detector"] if declaration == "explicit" else None
        )
        assert resolved.components == {"beam": "Emitter", "detector": "Sensor"}
        assert resolved.beam.detector == "Sensor"


def test_explicit_choice_precedes_matching_focus(tmp_path: Path) -> None:
    point = make_point(tmp_path)
    roles = point.resolve(["driver"], focus="Q1", driver="Q2")
    assert roles.components == {"driver": "Q2"}
    assert roles.driver.rate == 5100.0


def test_matching_focus_precedes_same_name_reference(tmp_path: Path) -> None:
    point = make_point(tmp_path)
    roles = point.resolve(["driver", "sense"], focus="R2", driver="Q1")
    assert roles.components == {"driver": "Q1", "sense": "R2"}
    assert roles.sense.rate == 7100.0


def test_same_name_reference_uses_resolved_explicit_component(tmp_path: Path) -> None:
    point = make_point(tmp_path)
    roles = point.resolve(["driver", "sense"], focus="Q1", driver="Q2")
    assert roles.components == {"driver": "Q2", "sense": "R2"}
    assert roles.sense.rate == 7100.0


def test_unique_qubit_in_bound_point_is_default_focus(tmp_path: Path) -> None:
    entry = ResultEntry.create(
        "roles", result_root=tmp_path / "results", database_root=tmp_path / "database"
    )
    entry.setup.add_component("R1", kind="fake/sensor", rate=6500.0)
    entry.setup.add_component("Q1", kind="fake/drive/a", rate=4300.0, sense="R1")
    point = entry.new_point("one_qubit")
    entry.setup.add_component("Q2", kind="fake/drive/b", rate=700.0)
    entry.new_point("two_qubits")
    roles = point.resolve()
    assert roles.components == {"driver": "Q1", "sense": "R1"}
    assert roles.driver.rate == 4300.0


@pytest.mark.parametrize("declaration", ["sequence", "mapping"])
def test_unregistered_roles_fail_with_resolution_context(
    tmp_path: Path, declaration: str
) -> None:
    point = make_point(tmp_path)
    roles = (
        ["unknown_role"]
        if declaration == "sequence"
        else {"unknown_role": RoleSpec("fake/drive/*")}
    )
    with pytest.raises(RoleResolutionError) as error:
        point.resolve(roles, focus="Q1", unknown_role="Q2")
    assert error.value.role == "unknown_role"
    assert error.value.focus == "Q1"
    assert "unknown" in error.value.reason.lower()


def test_explicit_choices_cannot_add_undeclared_requirements(tmp_path: Path) -> None:
    point = make_point(tmp_path)
    with pytest.raises(RoleResolutionError) as error:
        point.resolve(["driver"], focus="Q1", sense="R1")
    assert error.value.role == "sense"
    assert error.value.focus == "Q1"
    assert "declared" in error.value.reason


def test_mapping_via_override_traces_an_already_resolved_role(tmp_path: Path) -> None:
    point = make_point(tmp_path)
    with registered_role("control_readout", RoleSpec("fake/sensor")):
        roles = point.resolve(
            {
                "control": RoleSpec("fake/drive/*"),
                "control_readout": RoleSpec("fake/sensor", via="control.sense"),
            },
            focus="Q1",
        )
        assert roles.components == {"control": "Q1", "control_readout": "R1"}
        assert roles.control_readout.rate == 6500.0


def test_sequence_uses_registered_via_declaration(tmp_path: Path) -> None:
    point = make_point(tmp_path)
    with registered_role(
        "control_readout", RoleSpec("fake/sensor", via="control.sense")
    ):
        roles = point.resolve(["control", "control_readout"], focus="Q2")
        assert roles.components == {"control": "Q2", "control_readout": "R2"}
        assert roles.control_readout.rate == 7100.0


def test_pair_focus_resolves_both_qubits_coupler_and_readouts(
    pair_point: PointView,
) -> None:
    roles = pair_point.resolve(
        {
            "pair": RoleSpec("notebook/pair"),
            "control": RoleSpec("fake/drive/*", via="pair.links.control"),
            "target": RoleSpec("fake/drive/*", via="pair.links.target"),
            "coupler": RoleSpec("coupler/*", via="pair.links.coupler"),
            "control_readout": role_registry.get("control_readout"),
            "target_readout": role_registry.get("target_readout"),
        },
        focus="CZ12",
    )
    assert roles.components == {
        "pair": "CZ12",
        "control": "Q1",
        "target": "Q2",
        "coupler": "C12",
        "control_readout": "R1",
        "target_readout": "R2",
    }
    assert roles.control.rate == 4300.0
    assert roles.target.rate == 5100.0
    assert roles.control_readout.rate == 6500.0
    assert roles.target_readout.rate == 7100.0


def test_role_mapping_stays_fixed_while_values_follow_bound_point(
    tmp_path: Path,
) -> None:
    point = make_point(tmp_path)
    roles = point.resolve(driver="Q2")
    with pytest.raises(TypeError):
        # Deliberately bypass the readonly type to exercise the runtime guard.
        cast(MutableMapping[str, str], roles.components)["driver"] = "Q1"
    roles.driver.rate = 5250.0
    assert point.Q2.rate == 5250.0
    point.Q2.sense = "R1"
    another_resolution = point.resolve(driver="Q1")
    assert another_resolution.components == {"driver": "Q1", "sense": "R1"}
    assert roles.components == {"driver": "Q2", "sense": "R2"}
    assert roles.sense.rate == 7100.0
    other = ResultEntry.open(
        "roles", result_root=tmp_path / "results", database_root=tmp_path / "database"
    ).use_point("working")
    other.R2.rate = 7200.0
    assert roles.sense.rate == 7100.0
    point.refresh()
    assert roles.sense.rate == 7200.0
    with pytest.raises(AttributeError):
        _ = roles.coupler


@pytest.mark.parametrize("explicit", ["missing", "R1"])
def test_invalid_explicit_choice_does_not_fall_back_to_focus(
    tmp_path: Path, explicit: str
) -> None:
    point = make_point(tmp_path)
    with pytest.raises(RoleResolutionError) as error:
        point.resolve(["driver"], focus="Q1", driver=explicit)
    assert error.value.role == "driver"
    assert error.value.focus == "Q1"
    assert error.value.required_kind == "fake/drive/*"
    assert explicit in error.value.reason


@pytest.mark.parametrize("roles", [("control", "target"), ("target", "control")])
def test_focus_only_fills_first_matching_role(
    tmp_path: Path, roles: tuple[str, str]
) -> None:
    point = make_point(tmp_path)
    with pytest.raises(RoleResolutionError) as error:
        point.resolve(roles, focus="Q1")
    assert error.value.role == roles[1]
    assert error.value.focus == "Q1"
    assert error.value.required_kind == "fake/drive/*"
    assert roles[1] in str(error.value) and "Q1" in str(error.value)
    assert point.resolve(roles, focus="Q1", **{roles[1]: "Q1"}).components == {
        roles[0]: "Q1",
        roles[1]: "Q1",
    }


def test_multiple_qubits_do_not_choose_default_focus(tmp_path: Path) -> None:
    point = make_point(tmp_path)
    with pytest.raises(RoleResolutionError) as error:
        point.resolve(["driver"])
    assert error.value.role == "driver"
    assert error.value.focus is None


@pytest.mark.parametrize("shared_reference", [False, True])
def test_same_name_references_require_one_distinct_candidate(
    tmp_path: Path, shared_reference: bool
) -> None:
    point = make_point(tmp_path)
    if shared_reference:
        point.Q2.sense = "R1"
        roles = point.resolve(["control", "target", "sense"], control="Q1", target="Q2")
        assert roles.sense.rate == 6500.0
    else:
        with pytest.raises(RoleResolutionError) as error:
            point.resolve(["control", "target", "sense"], control="Q1", target="Q2")
        assert error.value.role == "sense"
        assert "R1" in error.value.reason and "R2" in error.value.reason


def test_registry_lifecycle_allows_explicit_replacement() -> None:
    registry = RoleRegistry()
    original = RoleSpec("fake/drive/*")
    registry.register("drive", original)
    assert registry.get("drive") == original
    with pytest.raises(ValueError, match="already registered"):
        registry.register("drive", RoleSpec("fake/sensor"))
    assert registry.get("drive") == original
    registry.unregister("drive")
    with pytest.raises(ValueError, match="Unknown role.*drive"):
        registry.get("drive")
    replacement = RoleSpec("fake/sensor", via="driver.sense")
    registry.register("drive", replacement)
    assert registry.get("drive") == replacement


@pytest.mark.parametrize("role", ["components", "_private", "not.a.role"])
def test_registry_rejects_names_that_cannot_be_role_attributes(role: str) -> None:
    registry = RoleRegistry()
    with pytest.raises(ValueError, match="Invalid role name"):
        registry.register(role, RoleSpec("fake/drive/*"))
    with pytest.raises(ValueError, match="Unknown role"):
        registry.get(role)


@pytest.mark.parametrize("via", ["control", "control..sense", "control._private"])
def test_invalid_via_does_not_reserve_role_name(via: str) -> None:
    registry = RoleRegistry()
    with pytest.raises(ValueError, match="Invalid role reference path"):
        registry.register("drive", RoleSpec("fake/sensor", via=via))
    registry.register("drive", RoleSpec("fake/drive/*"))
    assert registry.get("drive").kind == "fake/drive/*"


@pytest.mark.parametrize("explicit", [False, True])
def test_explicit_and_focus_precede_via_even_without_source_role(
    tmp_path: Path, explicit: bool
) -> None:
    point = make_point(tmp_path)
    with registered_role(
        "control_readout", RoleSpec("fake/sensor", via="control.sense")
    ):
        choices = {"control_readout": "R1"} if explicit else {}
        roles = point.resolve(["control_readout"], focus="R2", **choices)
        assert roles.components == {"control_readout": "R1" if explicit else "R2"}


@pytest.mark.parametrize("via", ["target.sense", "Q1.sense", "control.rate"])
def test_via_requires_resolved_role_and_registered_reference(
    tmp_path: Path, via: str
) -> None:
    point = make_point(tmp_path)
    with registered_role("control_readout", RoleSpec("fake/sensor", via=via)):
        with pytest.raises(RoleResolutionError) as error:
            point.resolve(["control", "control_readout"], focus="Q1")
        assert error.value.role == "control_readout"
        assert error.value.focus == "Q1"
        assert via in error.value.reason


def test_via_cannot_use_a_role_declared_later(tmp_path: Path) -> None:
    point = make_point(tmp_path)
    with registered_role(
        "control_readout", RoleSpec("fake/sensor", via="control.sense")
    ):
        with pytest.raises(RoleResolutionError) as error:
            point.resolve(["control_readout", "control"], focus="Q1")
        assert error.value.role == "control_readout"
        assert "control.sense" in error.value.reason


def test_same_name_string_field_is_not_an_implicit_reference(tmp_path: Path) -> None:
    class UnlinkedQubit(ComponentSchema):
        sense: str

    component_registry.register("fake/drive/unlinked", UnlinkedQubit)
    try:
        point = make_point(tmp_path)
        point.add_component("Q3", kind="fake/drive/unlinked", sense="R1")
        with pytest.raises(RoleResolutionError) as error:
            point.resolve(["driver", "sense"], focus="Q3")
        assert error.value.role == "sense"
        assert error.value.focus == "Q3"
    finally:
        component_registry.unregister("fake/drive/unlinked")


@pytest.mark.parametrize("reference", [None, "Q2"])
def test_missing_or_wrong_kind_reference_reports_required_role(
    tmp_path: Path, reference: str | None
) -> None:
    point = make_point(tmp_path)
    point.Q1.sense = reference
    with pytest.raises(RoleResolutionError) as error:
        point.resolve(["driver", "sense"], focus="Q1")
    assert error.value.role == "sense"
    assert error.value.focus == "Q1"
    assert error.value.required_kind == "fake/sensor"
    if reference is not None:
        assert reference in error.value.reason


def test_kind_pattern_matches_whole_kind_not_just_prefix(tmp_path: Path) -> None:
    point = make_point(tmp_path)
    assert point.resolve(
        {"driver": RoleSpec("fake/drive/a*")}, focus="Q1"
    ).components == {
        "driver": "Q1",
    }
    with pytest.raises(RoleResolutionError) as error:
        point.resolve({"driver": RoleSpec("fake/drive/a/*")}, focus="Q1")
    assert error.value.required_kind == "fake/drive/a/*"
