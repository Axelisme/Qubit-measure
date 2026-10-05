"""Recursive component value/path editing, sources, and isolated snapshots."""

from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from ruamel.yaml import YAML
from zcu_tools.resources.entry import ComponentSchema, ResultEntry
from zcu_tools.resources.entry.views import FieldView

from .view_support import create_entry, field_view, registered_model


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
@pytest.mark.parametrize("version", ["1.0", "1.2"])
def test_collision_extra_edit_abort_preserves_both_components(
    tmp_path: Path, registry_state_guard: None, owner: str, version: str
) -> None:
    class Colliding(ComponentSchema):
        model_config = ConfigDict(extra="allow")

    original_config = Colliding.model_config.copy()
    try:
        with registered_model("notebook/collision", Colliding) as kind:
            entry, results, database = create_entry(tmp_path)
            entry.setup.add_component("N1", kind=kind, model_config={"marker": "first"})
            entry.setup.add_component(
                "N2", kind=kind, model_config={"marker": "second"}
            )
            view = entry.setup if owner == "setup" else entry.new_point("working")
            source = (
                results / "entry/setup.yaml"
                if owner == "setup"
                else results / "entry/points/working/point.yaml"
            )
            yaml = YAML(typ="rt")
            stored = yaml.load(source)
            stored["format_version"] = version
            yaml.dump(stored, source)
            view.refresh()
            before = source.read_bytes()

            def abort_edit() -> None:
                with view.edit() as draft:
                    draft.set("N1.model_config.marker", "changed")
                    raise RuntimeError("abort requested")

            with pytest.raises(RuntimeError, match="abort requested"):
                abort_edit()
            assert field_view(view.N1.model_config).marker == "first"
            assert field_view(view.N2.model_config).marker == "second"
            assert source.read_bytes() == before
            reopened = ResultEntry.open(
                "entry", result_root=results, database_root=database
            )
            loaded = (
                reopened.setup if owner == "setup" else reopened.use_point("working")
            )
            assert field_view(loaded.N1.model_config).marker == "first"
            assert field_view(loaded.N2.model_config).marker == "second"
    finally:
        # Restore third-party class state even when the original bug mutates it.
        Colliding.model_config = original_config


@pytest.mark.parametrize("owner", ["setup", "point"])
@pytest.mark.parametrize("write", ["set", "property"])
def test_model_mapping_replacement_accepts_values_and_leaf_sources(
    tmp_path: Path, registry_state_guard: None, owner: str, write: str
) -> None:
    class Pin(BaseModel):
        model_config = ConfigDict(extra="forbid")
        index: int = Field(strict=True, ge=0)

    class Route(BaseModel):
        model_config = ConfigDict(extra="forbid")
        pin: Pin

    class Routed(ComponentSchema):
        routes: dict[str, Route]

    with registered_model("notebook/container", Routed) as kind:
        entry, results, database = create_entry(tmp_path)
        entry.setup.add_component(
            "N1", kind=kind, routes={"drive": {"pin": {"index": 2}}}
        )
        view = entry.setup if owner == "setup" else entry.new_point("working")
        if write == "set":
            with view.edit() as draft:
                draft.set("N1.routes", {"drive": {"pin": {"index": 6}}})
        else:
            view.N1.routes = {"drive": {"pin": {"index": 6}}}
        route = field_view(field_view(view.N1.routes).drive)
        assert field_view(route.pin).index == 6
        accepted = view.meta("N1.routes.drive.pin.index")
        assert accepted is not None and accepted.source == "manual"
        with view.edit() as draft:
            if write == "set":
                with pytest.raises(ValidationError):
                    draft.set("N1.routes", {"drive": {"pin": {"index": -1}}})
            else:
                with pytest.raises(ValidationError):
                    draft.N1.routes = {"drive": {"pin": {"index": -1}}}
            draft_route = field_view(field_view(draft.N1.routes).drive)
            assert field_view(draft_route.pin).index == 6
        assert view.meta("N1.routes.drive.pin.index") == accepted

        reopened = ResultEntry.open(
            "entry", result_root=results, database_root=database
        )
        loaded = reopened.setup if owner == "setup" else reopened.use_point("working")
        loaded_route = field_view(field_view(loaded.N1.routes).drive)
        assert field_view(loaded_route.pin).index == 6
        assert loaded.meta("N1.routes.drive.pin.index") == accepted


@pytest.mark.parametrize("owner", ["setup", "point"])
@pytest.mark.parametrize("write", ["set", "property"])
def test_model_list_replacement_accepts_values_and_source(
    tmp_path: Path, registry_state_guard: None, owner: str, write: str
) -> None:
    class Pin(BaseModel):
        index: int = Field(strict=True, ge=0)

    class Route(BaseModel):
        pin: Pin

    class Routed(ComponentSchema):
        routes: list[Route]

    with registered_model("notebook/list-container", Routed) as kind:
        entry, results, database = create_entry(tmp_path)
        entry.setup.add_component("N1", kind=kind, routes=[{"pin": {"index": 2}}])
        view = entry.setup if owner == "setup" else entry.new_point("working")
        if write == "set":
            with view.edit() as draft:
                draft.set("N1.routes", [{"pin": {"index": 6}}])
        else:
            view.N1.routes = [{"pin": {"index": 6}}]
        assert view.N1.routes == [{"pin": {"index": 6}}]
        accepted = view.meta("N1.routes")
        assert accepted is not None and accepted.source == "manual"
        with view.edit() as draft:
            if write == "set":
                with pytest.raises(ValidationError):
                    draft.set("N1.routes", [{"pin": {"index": -1}}])
            else:
                with pytest.raises(ValidationError):
                    draft.N1.routes = [{"pin": {"index": -1}}]
            assert draft.N1.routes == [{"pin": {"index": 6}}]
        assert view.meta("N1.routes") == accepted
        reopened = ResultEntry.open(
            "entry", result_root=results, database_root=database
        )
        loaded = reopened.setup if owner == "setup" else reopened.use_point("working")
        assert loaded.N1.routes == [{"pin": {"index": 6}}]
        assert loaded.meta("N1.routes") == accepted


@pytest.mark.parametrize("owner", ["setup", "point"])
@pytest.mark.parametrize("version", ["1.0", "1.2"])
def test_collision_extras_read_write_meta_and_reload(
    tmp_path: Path, registry_state_guard: None, owner: str, version: str
) -> None:
    class Colliding(ComponentSchema):
        model_config = ConfigDict(extra="allow")

    original_config = Colliding.model_config.copy()
    try:
        with registered_model("notebook/collision-values", Colliding) as kind:
            entry, results, database = create_entry(tmp_path)
            entry.setup.add_component(
                "N1", kind=kind, model_config={"marker": "first"}, model_dump="first"
            )
            entry.setup.add_component(
                "N2", kind=kind, model_config={"marker": "second"}
            )
            view = entry.setup if owner == "setup" else entry.new_point("working")
            source = (
                results / "entry/setup.yaml"
                if owner == "setup"
                else results / "entry/points/working/point.yaml"
            )
            yaml = YAML(typ="rt")
            stored = yaml.load(source)
            stored["format_version"] = version
            yaml.dump(stored, source)
            before = source.read_bytes()
            view.refresh()
            assert source.read_bytes() == before
            config = field_view(view.N1.model_config)
            assert config.marker == "first"
            assert view.N1.model_dump == "first"
            untouched = view.meta("N2.model_config.marker")
            config.marker = "child"
            assert config.marker == "child"
            view.N1.model_config = {"marker": "whole"}
            with view.edit() as draft:
                assert field_view(draft.N1.model_config).marker == "whole"
                draft.set("N1.model_config.marker", "accepted")
                draft.N1.model_dump = "accepted"
            accepted = view.meta("N1.model_config.marker")
            assert accepted is not None and accepted.source == "manual"
            method_source = view.meta("N1.model_dump")
            assert method_source is not None and method_source.source == "manual"
            assert config.marker == "accepted"
            assert view.N1.model_dump == "accepted"
            assert field_view(view.N2.model_config).marker == "second"
            assert view.meta("N2.model_config.marker") == untouched
            reopened = ResultEntry.open(
                "entry", result_root=results, database_root=database
            )
            loaded = (
                reopened.setup if owner == "setup" else reopened.use_point("working")
            )
            assert field_view(loaded.N1.model_config).marker == "accepted"
            assert loaded.N1.model_dump == "accepted"
            assert loaded.meta("N1.model_config.marker") == accepted
            assert loaded.meta("N1.model_dump") == method_source
            assert YAML(typ="safe").load(source)["format_version"] == version
    finally:
        Colliding.model_config = original_config
