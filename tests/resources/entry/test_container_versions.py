"""Container versions leave registered component evolution to the model definer."""

from collections.abc import Generator
from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from ruamel.yaml import YAML
from zcu_tools.format_version import FormatError, VersionError, YamlMap
from zcu_tools.resources.entry import (
    ComponentSchema,
    PointView,
    ResultEntry,
    SetupView,
    document_meta,
    document_value,
    snapshot,
    source_path,
)
from zcu_tools.resources.entry.schema import PointDocument, SetupDocument

from .view_support import registered_model


class StrictTiming(BaseModel):
    model_config = ConfigDict(extra="forbid")
    delay: float = Field(default=1.2, ge=0)


class StrictSensor(ComponentSchema):
    model_config = ConfigDict(extra="forbid")
    rate: float = Field(default=6500.0, ge=0)
    timing: StrictTiming = Field(default_factory=StrictTiming)

    @field_validator("rate")
    @classmethod
    def round_rate(cls, value: float) -> float:
        return round(value, 1)


class IgnoredTiming(StrictTiming):
    model_config = ConfigDict(extra="ignore")


class IgnoredSensor(StrictSensor):
    model_config = ConfigDict(extra="ignore")
    timing: IgnoredTiming = Field(default_factory=IgnoredTiming)


class OpenTiming(StrictTiming):
    model_config = ConfigDict(extra="allow")


class OpenSensor(StrictSensor):
    model_config = ConfigDict(extra="allow")
    timing: OpenTiming = Field(default_factory=OpenTiming)


@pytest.fixture
def strict_kind() -> Generator[str]:
    with registered_model("model-owned/strict", StrictSensor) as kind:
        yield kind


def bound_view(entry: ResultEntry, owner: str, kind: str) -> SetupView | PointView:
    entry.setup.add_component("S1", kind=kind)
    return entry.setup if owner == "setup" else entry.new_point("a")


def add_future_data(path: Path, location: str = "component") -> bytes:
    yaml = YAML(typ="rt")
    stored = yaml.load(path)
    stored["format_version"] = "1.7"
    fields = stored["components"]["S1"]
    (fields if location == "component" else fields["timing"])["future"] = 17
    yaml.dump(stored, path)
    return path.read_bytes()


@pytest.mark.parametrize("schema", [SetupDocument, PointDocument])
@pytest.mark.parametrize("location", ["component", "nested"])
def test_document_minor_never_relaxes_original_model_validation(
    schema: type[SetupDocument] | type[PointDocument], location: str, strict_kind: str
) -> None:
    general: YamlMap = {"created_at": "2026-10-10T00:00:00Z"}
    if schema is SetupDocument:
        general["entry_id"] = "00000000-0000-0000-0000-000000000001"
    for minor in (0, 7):
        fields: YamlMap = {
            "kind": strict_kind,
            "rate": 6500.04,
            "timing": {"delay": 1.2},
        }
        if location == "component":
            fields["future"] = 17
        else:
            fields["timing"] = {"delay": 1.2, "future": 17}
        document: YamlMap = {
            "format": "zcu.parameter-container",
            "format_version": f"1.{minor}",
            "general": general,
            "components": {"S1": fields},
        }
        before = deepcopy(document)
        error = None
        try:
            schema.model_validate(document)
        except ValidationError as caught:
            error = caught
        assert error is not None, "container minor must not override model extra=forbid"
        assert "future" in str(error)
        assert document == before
        fields.pop("future", None)
        fields["timing"] = {"delay": 1.2}
        validated = schema.model_validate(document)
        assert document_value(validated, "S1.rate") == 6500.0
        assert document_value(validated, "S1.timing.delay") == 1.2


@pytest.mark.parametrize("owner", ["setup", "point"])
@pytest.mark.parametrize("location", ["component", "nested"])
def test_refresh_minor_rejects_forbidden_data_without_publishing(
    entry: ResultEntry, owner: str, location: str, strict_kind: str
) -> None:
    view = bound_view(entry, owner, strict_kind)
    cached = snapshot(view)
    metadata = document_meta(cached, "S1.rate")
    path = source_path(view)
    bad_bytes = add_future_data(path, location)
    error = None
    try:
        view.refresh()
    except ValidationError as caught:
        error = caught
    assert error is not None, "refresh must validate the original model"
    assert "future" in str(error)
    assert snapshot(view) == cached
    assert document_meta(snapshot(view), "S1.rate") == metadata
    assert path.read_bytes() == bad_bytes
    if owner == "setup":
        with pytest.raises(ValidationError, match="future"):
            ResultEntry.open(
                "entry",
                result_root=entry.result_path.parent,
                database_root=entry.database_path.parent,
            )
    else:
        with pytest.raises(ValidationError, match="future"):
            entry.use_point("a")
    assert path.read_bytes() == bad_bytes
    if isinstance(cached, PointDocument):
        candidate = cached.model_copy(deep=True)
        candidate.format_version = "1.7"
        before = candidate.model_dump()
        with pytest.raises(ValidationError, match="future"):
            entry.updated_component(candidate, "S2", strict_kind, {"future": 17})
        with pytest.raises(ValidationError, match="future"):
            entry.updated_point(candidate, "S1.timing", {"delay": 1.2, "future": 17})
        with pytest.raises(ValidationError):
            entry.updated_point(candidate, "S1.rate", -1.0)
        assert candidate.model_dump() == before
        assert snapshot(view) == cached
        assert path.read_bytes() == bad_bytes


@pytest.mark.parametrize("owner", ["setup", "point"])
def test_commit_rejects_latest_forbidden_minor_data_before_publication(
    entry: ResultEntry, owner: str, strict_kind: str
) -> None:
    view = bound_view(entry, owner, strict_kind)
    cached = snapshot(view)
    path = source_path(view)
    bad_bytes = b""
    error = None
    try:
        with view.edit() as draft:
            draft.set("S1.rate", 6600.0)
            bad_bytes = add_future_data(path)
    except ValidationError as caught:
        error = caught
    assert error is not None, "commit must reject the latest illegal component"
    assert "future" in str(error)
    assert path.read_bytes() == bad_bytes
    assert snapshot(view) == cached
    assert document_meta(snapshot(view), "S1.rate") == document_meta(cached, "S1.rate")


@pytest.mark.parametrize("owner", ["setup", "point"])
def test_minor_commit_keeps_container_unknowns_but_applies_component_extra_policy(
    entry: ResultEntry, owner: str
) -> None:
    with (
        registered_model("model-owned/ignore", IgnoredSensor) as ignored,
        registered_model("model-owned/allow", OpenSensor) as allowed,
    ):
        view = bound_view(entry, owner, ignored)
        view.add_component("S2", kind=allowed)
        path = source_path(view)
        yaml = YAML(typ="rt")
        stored = yaml.load(path)
        stored["format_version"] = "1.7"
        stored["future_top"] = {"keep": [1, "two"]}
        stored["general"]["future_general"] = "keep"
        for name in ("S1", "S2"):
            stored["components"][name]["future"] = 17
            stored["components"][name]["timing"]["future"] = 19
        stored["provenance"]["S2.rate"]["future_source"] = "keep"
        yaml.dump(stored, path)
        before = path.read_bytes()
        view.refresh()
        assert path.read_bytes() == before
        with pytest.raises(AttributeError, match="future"):
            document_value(snapshot(view), "S1.future")
        assert document_value(snapshot(view), "S2.future") == 17
        assert document_value(snapshot(view), "S2.timing.future") == 19
        with view.edit() as draft:
            draft.set("S1.rate", 6600.04)
        final = yaml.load(path)
        assert "future" not in final["components"]["S1"], (
            "ignore must discard unknown component fields at any minor"
        )
        assert "future" not in final["components"]["S1"]["timing"]
        assert final["components"]["S2"]["future"] == 17
        assert final["components"]["S2"]["timing"]["future"] == 19
        assert final["future_top"] == {"keep": [1, "two"]}
        assert final["general"]["future_general"] == "keep"
        assert final["format_version"] == "1.7"
        assert document_value(snapshot(view), "S1.rate") == 6600.0
        metadata = document_meta(snapshot(view), "S1.rate")
        assert metadata is not None
        assert metadata.source == "manual"
        assert final["provenance"]["S2.rate"]["future_source"] == "keep"
        reloaded = ResultEntry.open(
            "entry",
            result_root=entry.result_path.parent,
            database_root=entry.database_path.parent,
        )
        again = reloaded.setup if owner == "setup" else reloaded.use_point("a")
        assert snapshot(again) == snapshot(view)
        for header, value, expected in (
            ("format", "wrong", FormatError),
            ("format_version", "2.0", VersionError),
        ):
            final[header] = value
            yaml.dump(final, path)
            bad_bytes = path.read_bytes()
            cached = snapshot(view)
            with pytest.raises(expected):
                view.refresh()
            assert path.read_bytes() == bad_bytes
            assert snapshot(view) == cached
            final[header] = "zcu.parameter-container" if header == "format" else "1.7"
