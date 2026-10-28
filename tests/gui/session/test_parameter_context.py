"""Fixed parameter projection, role capture and read capability contracts."""

from collections.abc import Generator, MutableMapping
from pathlib import Path
from typing import Literal, Self

import pytest
from pydantic import Field, field_validator, model_validator
from zcu_tools.cfg_model import ConstWaveformCfg, PulseSpec
from zcu_tools.gui.expected_error import ExpectedErrorCategory, InvalidInputError
from zcu_tools.gui.session.aliases import AliasRow, AliasTable
from zcu_tools.gui.session.context_types import (
    ContextSourceFault,
    ContextSourceUnavailableError,
    SavedPoint,
)
from zcu_tools.gui.session.parameter_capture import (
    capture_parameters,
    restore_parameters,
)
from zcu_tools.gui.session.types import ParameterContext, RoleBinding
from zcu_tools.resources.context.library import (
    ComponentLibrary,
    LibrarySnapshot,
    ModuleLibraryDocument,
)
from zcu_tools.resources.document_store import DocumentValue
from zcu_tools.resources.entry import (
    ComponentSchema,
    RoleResolutionError,
    RoleSpec,
    component_registry,
    role_registry,
)
from zcu_tools.resources.entry.schema import (
    ParameterContent,
    PointDocument,
    PointGeneral,
)


class CaptureDrive(ComponentSchema):
    rate: float
    sense: str
    module: dict[str, str | None]


class CaptureSensor(ComponentSchema):
    rate: float


def test_component_projection_enforces_cross_field_checks_without_blocking_leaf_reads(
    source: tuple[PointDocument, LibrarySnapshot],
) -> None:
    class Bounded(ComponentSchema):
        low: float
        high: float

        @model_validator(mode="after")
        def bounds(self) -> Self:
            if self.low > self.high:
                raise ValueError("low exceeds high")
            return self

    component_registry.register("capture/bounded", Bounded)
    try:
        parameters = ParameterContext(
            point=ParameterContent(
                {
                    "components": {
                        "pair": {"kind": "capture/bounded", "low": 3.0, "high": 2.0},
                        "unused": ["bad component"],
                    }
                }
            ),
            library=source[1],
            target=None,
            roles=None,
        )
        assert parameters.value("pair.low") == 3.0
        with pytest.raises(ValueError, match="(?s)pair.*low exceeds high"):
            parameters.component("pair")
    finally:
        component_registry.unregister("capture/bounded")


@pytest.fixture
def source() -> Generator[tuple[PointDocument, LibrarySnapshot]]:
    before = component_registry.kinds()
    component_registry.register("capture/drive", CaptureDrive)
    component_registry.register("capture/sensor", CaptureSensor)
    role_registry.register("capture_driver", RoleSpec("capture/drive"))
    role_registry.register(
        "capture_sensor", RoleSpec("capture/sensor", "capture_driver.sense")
    )
    try:
        # No backing point file exists; every read must use the capture.
        document = PointDocument(
            format="zcu.parameter-container",
            format_version="2.0",
            schema_version=1,
            general=PointGeneral(
                created_at="2026-10-09T00:00:00Z",
                ext={"nullable": None, "nested": {"number": 2.0}},
            ),
        )
        for name, rate, sense in (("Q1", 4300.0, "R1"), ("Q2", 5100.0, "R2")):
            document.components[name] = CaptureDrive(
                kind="capture/drive",
                rate=rate,
                sense=sense,
                module={
                    "x180": "$capture_driver.pulses.pi",
                    "empty": None,
                    "dangling": "Q1.pulses.missing",
                },
            )
        for name in ("R1", "R2"):
            document.components[name] = CaptureSensor(
                kind="capture/sensor", rate=6500.0
            )
        library = LibrarySnapshot(
            ModuleLibraryDocument(
                format="zcu.module-library",
                format_version="2.0",
                components={
                    "Q1": ComponentLibrary(
                        pulses={
                            "pi": PulseSpec(
                                ch=0,
                                nqz=1,
                                freq=100.0,
                                gain=0.25,
                                waveform="Q1.waveforms.base",
                            )
                        },
                        waveforms={"base": ConstWaveformCfg(length=0.05)},
                    )
                },
            )
        )
        yield document, library
    finally:
        role_registry.unregister("capture_sensor")
        role_registry.unregister("capture_driver")
        component_registry.unregister("capture/sensor")
        component_registry.unregister("capture/drive")
        assert component_registry.kinds() == before


def capture(
    source: tuple[PointDocument, LibrarySnapshot],
    *,
    roles: RoleBinding | None = None,
    faults: tuple[ContextSourceFault, ...] = (),
) -> ParameterContext:
    point, library = source
    return ParameterContext(
        point=point,
        library=library,
        target=SavedPoint(entry_name="entry", point="working"),
        roles=roles,
        source_faults=faults,
    )


def declarations() -> dict[str, RoleSpec]:
    return {
        "capture_driver": RoleSpec("capture/drive"),
        "capture_sensor": RoleSpec("capture/sensor", "capture_driver.sense"),
    }


def test_empty_analysis_environment_has_no_persistent_identity_or_metadata() -> None:
    context = ParameterContext.empty()
    assert context.target is None
    assert context.storage is None
    assert context.document().general is None
    assert context.document().metadata == {}
    assert context.document().components == {}
    assert context.lookup("general.created_at") == DocumentValue(False, None)
    assert context.library.module_keys() == ()
    assert context.aliases == ()


def test_parameter_snapshot_round_trip_keeps_content_library_and_authored_aliases(
    source: tuple[PointDocument, LibrarySnapshot],
) -> None:
    from pydantic import TypeAdapter

    original = capture(source).bind_aliases(
        (AliasRow("drive", "Q1"), AliasRow("sensor", "$drive.sense")), {}
    )
    snapshot = capture_parameters(original)
    codec = TypeAdapter(type(snapshot))
    encoded = codec.dump_json(snapshot)
    source[0].components.clear()
    restored = restore_parameters(codec.validate_json(encoded))
    assert restored.target is None
    assert restored.storage is None
    assert restored.value("$drive.rate") == 4300.0
    assert restored.value("$sensor.rate") == 6500.0
    assert restored.value("$drive.pulses.pi.gain") == 0.25
    assert restored.aliases == original.aliases
    assert restored.roles == original.roles
    assert restored.document().metadata == original.document().metadata


def test_snapshot_preserves_consumed_factory_defaults_without_validating_unused_fields(
    source: tuple[PointDocument, LibrarySnapshot],
) -> None:
    from itertools import count

    from zcu_tools.experiment.metadata import ParameterSnapshot

    sequence = count(1)

    class Notebook(ComponentSchema):
        title: str
        token: str = Field(default_factory=lambda: f"capture-{next(sequence)}")
        bad: float

        @field_validator("title", mode="before")
        @classmethod
        def strip_title(cls, value: object) -> object:
            return value.strip().lower() if isinstance(value, str) else value

    component_registry.register("capture/notebook", Notebook)
    try:
        original = ParameterContext(
            point=ParameterContent(
                {
                    "components": {
                        "note": {
                            "kind": "capture/notebook",
                            "title": "  SAVED  ",
                            "bad": [],
                        }
                    }
                }
            ),
            library=source[1],
            target=None,
            roles=None,
        )
        assert original.value("note.token") == "capture-1"
        assert original.value("note.title") == "saved"
        encoded = capture_parameters(original).model_dump_json()
        restored = restore_parameters(ParameterSnapshot.model_validate_json(encoded))
        assert restored.value("note.token") == "capture-1"
        assert restored.value("note.token") == "capture-1"
        assert restored.value("note.title") == "saved"
        with pytest.raises(ValueError, match="note.bad"):
            restored.value("note.bad")
        with pytest.raises(ValueError, match="note"):
            restored.component("note")
    finally:
        component_registry.unregister("capture/notebook")


def test_snapshot_preserves_nested_factory_defaults_through_capture(
    source: tuple[PointDocument, LibrarySnapshot],
) -> None:
    from itertools import count

    from pydantic import BaseModel
    from zcu_tools.experiment.metadata import ParameterSnapshot

    sequence = count(1)

    class Nested(BaseModel):
        token: str = Field(default_factory=lambda: f"nested-{next(sequence)}")
        optional: float | None = None

    class Notebook(ComponentSchema):
        nested: Nested = Field(default_factory=Nested)

    component_registry.register("capture/nested-notebook", Notebook)
    try:
        original = ParameterContext.from_content(
            point=ParameterContent(
                {"components": {"note": {"kind": "capture/nested-notebook"}}}
            ),
            library=source[1],
            aliases=None,
            names={},
        )
        assert original.value("note.nested.token") == "nested-1"
        assert original.value("note.nested.token") == "nested-1"
        encoded = capture_parameters(original).model_dump_json()
        restored = restore_parameters(ParameterSnapshot.model_validate_json(encoded))
        assert restored.value("note.nested.token") == "nested-1"
        assert restored.component("note").model_dump()["nested"] == {
            "token": "nested-1",
            "optional": None,
        }
    finally:
        component_registry.unregister("capture/nested-notebook")


@pytest.mark.parametrize("defect", ["missing", "different_format", "extra"])
def test_snapshot_validates_only_values_actually_read(
    source: tuple[PointDocument, LibrarySnapshot],
    defect: str,
) -> None:
    import json

    from pydantic import TypeAdapter

    original = capture(source).bind_aliases((AliasRow("drive", "Q1"),), {})
    snapshot = capture_parameters(original)
    codec = TypeAdapter(type(snapshot))
    data = json.loads(codec.dump_json(snapshot))
    data["point"]["components"]["unused"] = ["not a component"]
    data["point"]["components"]["Q1"]["unrelated"] = ["old format"]
    data["point"]["components"]["Q1"]["sense"] = ["wrong field type"]
    data["point"]["general"] = "unrelated old format"
    data["aliases"]["unused"] = {"not": "an alias binding"}
    if defect == "missing":
        del data["library"]
    elif defect == "different_format":
        data["library"] = ["unrelated old format"]
    else:
        data["extra"] = ["unknown block"]
        data["library"]["extra"] = {"anything": True}
        data["library"]["components"]["Q1"]["pulses"]["pi"]["unused"] = []
    restored = restore_parameters(codec.validate_json(json.dumps(data)))
    assert restored.value("$drive.rate") == 4300.0
    if defect == "extra":
        assert (
            restored.library.entry_value("Q1.pulses.pi", kind="module")["gain"] == 0.25
        )
        assert "Q1.pulses.pi" in restored.library.module_keys()
    else:
        with pytest.raises((KeyError, ValueError), match="library"):
            restored.library.module_spec("Q1.pulses.pi")
        with pytest.raises((KeyError, ValueError), match="library"):
            restored.library.module_keys()
    with pytest.raises((ValueError, InvalidInputError), match="unused|alias"):
        restored.value("$unused")
    with pytest.raises(ValueError, match="valid string"):
        restored.value("Q1.sense")


@pytest.mark.parametrize("components", [None, ["old shape"], {"unrelated": 1}])
def test_snapshot_general_reads_do_not_require_component_or_alias_sections(
    components: object,
) -> None:
    import json

    from zcu_tools.experiment.metadata import ParameterSnapshot

    data = {
        "point": {"general": {"description": "recorded"}, "components": components},
        "aliases": 42,
    }
    snapshot = ParameterSnapshot.model_validate_json(json.dumps(data))
    restored = restore_parameters(snapshot)
    assert restored.value("general.description") == "recorded"
    with pytest.raises(ValueError, match="alias"):
        restored.value("$unused")


def test_point_capture_is_detached_and_preserves_target(
    source: tuple[PointDocument, LibrarySnapshot],
) -> None:
    context = capture(source)
    source[0].general.ext.clear()
    copied = context.document()
    copied.components.clear()
    assert context.value("Q1.rate") == 4300.0
    assert context.value("general.ext.nested.number") == 2.0
    assert context.target == SavedPoint(entry_name="entry", point="working")
    assert context.roles is None
    assert context.source_faults == ()


@pytest.mark.parametrize(
    ("explicit", "expected"),
    [
        ({}, {"capture_driver": "Q1", "capture_sensor": "R1"}),
        (
            {"capture_driver": "Q2"},
            {"capture_driver": "Q2", "capture_sensor": "R2"},
        ),
        ({"capture_driver": None}, {}),
        (
            {"capture_driver": None, "capture_sensor": "R2"},
            {"capture_sensor": "R2"},
        ),
        (
            {"capture_sensor": "R2"},
            {"capture_driver": "Q1", "capture_sensor": "R2"},
        ),
        ({"capture_sensor": None}, {"capture_driver": "Q1"}),
    ],
)
def test_manual_capture_reinfers_only_nonmanual_roles_from_fixed_focus(
    source: tuple[PointDocument, LibrarySnapshot],
    explicit: dict[str, str | None],
    expected: dict[str, str],
) -> None:
    original = capture(source).bind_roles(declarations(), focus="Q1", explicit={})
    updated = original.bind_roles(declarations(), focus="Q1", explicit=explicit)
    binding = updated.roles
    assert binding is not None
    assert binding.focus == "Q1"
    assert dict(binding.explicit) == explicit
    assert dict(binding.names) == expected
    assert updated.target == original.target
    assert updated.value("Q1.rate") == 4300.0
    previous = original.roles
    assert previous is not None
    assert dict(previous.names) == {"capture_driver": "Q1", "capture_sensor": "R1"}
    assert dict(previous.explicit) == {}
    for role in declarations():
        if role not in expected:
            with pytest.raises(RoleResolutionError) as error:
                updated.resolve_path(f"${role}.rate")
            assert error.value.role == role
    restored = updated.bind_roles(declarations(), focus=binding.focus, explicit={})
    assert restored.resolve_path("$capture_driver.rate") == "Q1.rate"
    assert restored.resolve_path("$capture_sensor.rate") == "R1.rate"


def test_bind_roles_uses_only_fixed_point_and_detaches_mappings(
    source: tuple[PointDocument, LibrarySnapshot],
) -> None:
    context = capture(source)
    declared = declarations()
    bound = context.bind_roles(declared, focus="Q1", explicit={})
    declared.clear()
    source[0].components.clear()
    binding = bound.roles
    assert binding == RoleBinding(
        "Q1", declarations(), {"capture_driver": "Q1", "capture_sensor": "R1"}
    )
    assert binding is not None
    if isinstance(binding.names, MutableMapping):
        binding.names["capture_driver"] = "Q2"
    if isinstance(binding.declarations, MutableMapping):
        binding.declarations.clear()
    rebound = bound.bind_roles(
        declarations(), focus="Q1", explicit={"capture_driver": "Q2"}
    )
    assert bound.value("$capture_driver.rate") == 4300.0
    assert rebound.value("$capture_driver.rate") == 5100.0
    assert rebound.value("$capture_sensor.rate") == 6500.0
    assert bound.target == context.target
    assert rebound.target == context.target
    assert context.roles is None


@pytest.mark.parametrize("fault_document", [None, "point", "module_cfg"])
def test_empty_role_binding_does_not_consume_components_or_sources(
    source: tuple[PointDocument, LibrarySnapshot],
    tmp_path: Path,
    fault_document: Literal["point", "module_cfg"] | None,
) -> None:
    faults = (
        ()
        if fault_document is None
        else (
            ContextSourceFault(
                fault_document,
                tmp_path / f"{fault_document}.yaml",
                "OSError",
                "unreadable",
            ),
        )
    )
    context = ParameterContext(
        point=ParameterContent({"components": ["not a component mapping"]}),
        library=source[1],
        target=None,
        roles=None,
        source_faults=faults,
    )

    bound = context.bind_roles({}, focus=None, explicit={})

    assert bound.roles == RoleBinding(None, {}, {})
    assert bound.target is None
    assert bound.source_faults == faults
    with pytest.raises(RoleResolutionError, match="Role is not declared"):
        context.bind_roles({}, focus=None, explicit={"extra": "Q1"})
    if fault_document == "point":
        with pytest.raises(ContextSourceUnavailableError) as error:
            bound.document()
        assert error.value.faults == faults
    elif fault_document == "module_cfg":
        with pytest.raises(ContextSourceUnavailableError) as error:
            _ = bound.library
        assert error.value.faults == faults


def test_bind_roles_returns_partial_aliases_after_ambiguous_inference(
    source: tuple[PointDocument, LibrarySnapshot],
) -> None:
    context = capture(source)
    partial = context.bind_roles(declarations(), focus=None, explicit={})
    assert partial.roles is not None
    assert partial.roles.names == {}
    assert partial.roles.declarations == declarations()
    assert partial.value("Q1.rate") == 4300.0
    assert partial.target == context.target
    for role in declarations():
        with pytest.raises(RoleResolutionError, match=f"{role}.*not bound"):
            partial.value(f"${role}.rate")


@pytest.mark.parametrize("same_name_component", [False, True])
def test_partial_capture_reports_unbound_role_only_when_read(
    source: tuple[PointDocument, LibrarySnapshot], same_name_component: bool
) -> None:
    point, library = source
    if same_name_component:
        point.components["capture_driver"] = point.components["Q2"].model_copy(
            deep=True
        )
    context = ParameterContext(
        point=point,
        library=library,
        target=SavedPoint(entry_name="entry", point="working"),
        roles=RoleBinding(None, declarations(), {"capture_sensor": "R1"}),
    )
    assert context.value("$capture_sensor.rate") == 6500.0
    assert context.value("Q1.rate") == 4300.0
    if same_name_component:
        assert context.value("capture_driver.rate") == 5100.0
    assert context.roles is not None
    assert "capture_driver" not in context.roles.names
    with pytest.raises(RoleResolutionError, match="capture_driver.*not bound"):
        context.value("$capture_driver.rate")
    with pytest.raises(RoleResolutionError, match="capture_driver.*not bound"):
        context.library_key("$capture_driver.module.x180")


@pytest.mark.parametrize(
    "explicit",
    [{"extra": "Q1"}, {"capture_driver": "missing"}, {"capture_driver": "R1"}],
)
def test_bad_role_capture_preserves_entry_error(
    source: tuple[PointDocument, LibrarySnapshot], explicit: dict[str, str]
) -> None:
    with pytest.raises(RoleResolutionError):
        capture(source).bind_roles(declarations(), focus="Q1", explicit=explicit)


def test_captured_names_are_not_reinferred_and_targets_are_checked(
    source: tuple[PointDocument, LibrarySnapshot],
) -> None:
    roles = RoleBinding(
        "Q1", declarations(), {"capture_driver": "Q2", "capture_sensor": "R1"}
    )
    context = capture(source, roles=roles)
    assert context.value("$capture_driver.rate") == 5100.0
    assert context.value("$capture_sensor.rate") == 6500.0
    bad = RoleBinding(
        "Q1", declarations(), {"capture_driver": "R1", "capture_sensor": "R2"}
    )
    with pytest.raises(RoleResolutionError):
        capture(source, roles=bad)


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("general.ext.nullable", DocumentValue(True, None)),
        ("general.ext.absent", DocumentValue(False, None)),
        ("Q1.rate", DocumentValue(True, 4300.0)),
        ("Q1.absent", DocumentValue(False, None)),
        ("general.ext.nested.number", DocumentValue(True, 2.0)),
    ],
)
def test_point_lookup_preserves_absent_null_and_values(
    source: tuple[PointDocument, LibrarySnapshot], path: str, expected: DocumentValue
) -> None:
    context = capture(source)
    assert context.lookup_point(path) == expected
    assert context.lookup(path) == expected
    if expected.present:
        assert context.value(path) == expected.value
    else:
        with pytest.raises(AttributeError):
            context.value(path)


@pytest.mark.parametrize(
    "path",
    [
        "Q1.pulses",
        "Q1.pulses.pi.gain",
        "Q1.readout",
        "Q1.reset.r",
        "Q1.waveforms.base.length",
        "Q1.module",
        "Q1.module.x180.gain",
    ],
)
def test_point_query_rejects_library_and_slot_namespaces(
    source: tuple[PointDocument, LibrarySnapshot], path: str
) -> None:
    with pytest.raises(InvalidInputError) as error:
        capture(source).lookup_point(path)
    assert error.value.category == ExpectedErrorCategory.INVALID_INPUT
    assert error.value.reason_code == "parameter_path_not_readable"
    assert path in str(error.value)


@pytest.mark.parametrize(
    "path", ["", "Q1..rate", "missing.rate", "capture_driver.rate"]
)
def test_invalid_or_unknown_paths_do_not_become_missing(
    source: tuple[PointDocument, LibrarySnapshot], path: str
) -> None:
    with pytest.raises((ValueError, RoleResolutionError)):
        capture(source).lookup_point(path)


def test_library_and_slot_reads_use_same_fixed_role_binding(
    source: tuple[PointDocument, LibrarySnapshot],
) -> None:
    context = capture(source).bind_roles(declarations(), focus="Q1", explicit={})
    assert context.library_key("$capture_driver.pulses.pi") == "Q1.pulses.pi"
    assert context.library_key("$capture_driver.module.x180") == "Q1.pulses.pi"
    assert context.value("$capture_driver.module.x180") == "$capture_driver.pulses.pi"
    assert context.value("$capture_driver.module.x180.gain") == 0.25
    assert context.lookup("Q1.pulses.missing.gain") == DocumentValue(False, None)
    assert context.lookup("Q1.pulses.pi.mixer_freq") == DocumentValue(True, None)
    assert context.value("Q1.pulses.pi.waveform.length") == 0.05
    for path in (
        "Q1.module.empty.gain",
        "Q1.module.missing.gain",
        "Q1.module.dangling.gain",
    ):
        with pytest.raises(AttributeError):
            context.lookup(path)


def test_general_fields_named_like_library_partitions_remain_point_reads(
    source: tuple[PointDocument, LibrarySnapshot],
) -> None:
    source[0].general.ext["pulses"] = {"pi": 3.0}
    assert capture(source).lookup_point("general.ext.pulses.pi") == DocumentValue(
        True, 3.0
    )


def test_direct_library_roots_do_not_require_an_existing_entry(
    source: tuple[PointDocument, LibrarySnapshot],
) -> None:
    assert capture(source).library_key("Q1.pulses.future") == "Q1.pulses.future"


def test_nested_library_failure_is_not_reported_as_missing(
    source: tuple[PointDocument, LibrarySnapshot],
) -> None:
    point, library = source
    broken = library.replaced_module(
        "Q1.pulses.pi",
        PulseSpec(ch=0, nqz=1, freq=100.0, gain=0.25, waveform="Q1.waveforms.missing"),
    )
    with pytest.raises((KeyError, ValueError)):
        capture((point, broken)).lookup("Q1.pulses.pi.gain")


def test_point_consumers_ignore_unused_library_source_fault(
    source: tuple[PointDocument, LibrarySnapshot], tmp_path: Path
) -> None:
    fault = ContextSourceFault(
        "module_cfg", tmp_path / "module_cfg.yaml", "OSError", "unreadable"
    )
    context = capture(source, faults=(fault,))

    assert context.value("Q1.rate") == 4300.0
    assert context.lookup_point("Q1.rate") == DocumentValue(True, 4300.0)
    assert context.component("Q1") == source[0].components["Q1"]
    assert context.document() == source[0]
    assert context.whole_document() == source[0]
    bound = context.bind_roles(declarations(), focus="Q1", explicit={})
    assert bound.value("$capture_driver.rate") == 4300.0
    assert bound.source_faults == (fault,)
    aliased = context.bind_aliases((AliasRow("driver", "Q1"),), {})
    assert aliased.value("$driver.rate") == 4300.0
    assert aliased.roles == RoleBinding(None, {}, {"driver": "Q1"})
    with pytest.raises(ContextSourceUnavailableError) as error:
        _ = context.library
    assert error.value.faults == (fault,)


@pytest.mark.parametrize("method", ["lookup", "library_key", "has_library_key"])
def test_library_consumers_report_their_own_source_fault(
    source: tuple[PointDocument, LibrarySnapshot], tmp_path: Path, method: str
) -> None:
    fault = ContextSourceFault(
        "module_cfg", tmp_path / "module_cfg.yaml", "OSError", "unreadable"
    )
    context = capture(source, faults=(fault,))

    path = "Q1.pulses.pi.freq" if method == "lookup" else "Q1.pulses.pi"
    with pytest.raises(ContextSourceUnavailableError) as error:
        getattr(context, method)(path)
    assert error.value.faults == (fault,)
    assert context.value("Q1.rate") == 4300.0


def test_parameter_capture_rejects_setup_diagnostics(
    source: tuple[PointDocument, LibrarySnapshot],
    tmp_path: Path,
) -> None:
    fault = ContextSourceFault(
        "setup", tmp_path / "setup.yaml", "OSError", "unreadable"
    )
    with pytest.raises(ValueError, match="point/module_cfg"):
        capture(source, faults=(fault,))


def read_projection(context: ParameterContext, method: str) -> None:
    """Exercise one public getter without reaching inside the captured source."""
    if method == "document":
        context.document()
    elif method == "whole_document":
        context.whole_document()
    elif method == "component":
        context.component("Q1")
    elif method == "bind_roles":
        context.bind_roles(declarations(), focus="Q1", explicit={})
    else:
        getattr(context, method)("Q1.rate")


@pytest.mark.parametrize("document", ["point", "module_cfg"])
def test_library_capture_checks_only_its_source_fault(
    source: tuple[PointDocument, LibrarySnapshot],
    tmp_path: Path,
    document: Literal["point", "module_cfg"],
) -> None:
    fault = ContextSourceFault(
        document, tmp_path / f"{document}.yaml", "OSError", "unreadable"
    )
    broken = capture(source, faults=(fault,))
    assert broken.source_faults == (fault,)
    if document == "module_cfg":
        with pytest.raises(ContextSourceUnavailableError) as error:
            _ = broken.library
        assert error.value.faults == (fault,)
    else:
        assert (
            broken.library.entry_value("Q1.pulses.pi", kind="module")["freq"] == 100.0
        )


@pytest.mark.parametrize(
    "method",
    [
        "document",
        "whole_document",
        "component",
        "bind_roles",
        "resolve_path",
        "library_key",
        "lookup",
        "lookup_point",
        "value",
    ],
)
def test_fault_capture_refuses_data_but_keeps_diagnostics(
    source: tuple[PointDocument, LibrarySnapshot], tmp_path: Path, method: str
) -> None:
    fault = ContextSourceFault(
        "point", tmp_path / "point.yaml", "OSError", "unreadable"
    )
    healthy = capture(source)
    broken = capture(source, faults=(fault,))
    assert broken.target == healthy.target
    assert broken.roles is None
    assert broken.source_faults == (fault,)
    with pytest.raises(ContextSourceUnavailableError) as error:
        read_projection(broken, method)
    assert error.value.reason_code == "source_unavailable"
    assert error.value.faults == (fault,)
    assert healthy.value("Q1.rate") == 4300.0


def test_authored_alias_reads_keep_the_fixed_point_and_distinct_concrete_namespace(
    source: tuple[PointDocument, LibrarySnapshot],
) -> None:
    point, _library = source
    point.components["capture_driver"] = point.components["Q2"].model_copy(deep=True)
    table = AliasTable(
        (
            AliasRow("capture_driver", "Q1"),
            AliasRow("capture_sensor", "$capture_driver.sense"),
        )
    )
    context = capture(source).bind_aliases(table.rows, declarations())
    assert context.value("$capture_driver.rate") == 4300.0
    assert context.resolve_path("$capture_sensor.rate") == "R1.rate"
    assert context.value("capture_driver.rate") == 5100.0
    assert context.roles is not None
    assert context.roles.names == {"capture_driver": "Q1", "capture_sensor": "R1"}
    table.replace((AliasRow("capture_driver", "Q2"), *table.rows[1:]))
    point.components["Q1"] = point.components["Q2"].model_copy(deep=True)
    assert context.value("$capture_driver.rate") == 4300.0
    assert context.value("$capture_sensor.rate") == 6500.0
    assert context.aliases[0].value == "Q1"


def test_authored_aliases_can_remain_unbound_until_read(
    source: tuple[PointDocument, LibrarySnapshot],
) -> None:
    point, _library = source
    point.components["capture_driver"] = point.components["Q1"].model_copy(deep=True)
    context = capture(source).bind_aliases(
        (
            AliasRow("capture_driver", ""),
            AliasRow("capture_sensor", "$capture_driver.sense"),
        ),
        declarations(),
    )
    assert context.roles is not None
    assert context.roles.names == {}
    assert context.value("capture_driver.rate") == 4300.0
    assert context.value("general.ext.nested.number") == 2.0
    with pytest.raises(RoleResolutionError) as caught:
        context.value("$capture_sensor.rate")
    assert caught.value.role == "capture_sensor"


def test_authored_aliases_route_library_and_point_provenance_from_one_capture(
    source: tuple[PointDocument, LibrarySnapshot],
) -> None:
    point, _library = source
    point.components["Q1"] = CaptureDrive(
        kind="capture/drive",
        rate=4300.0,
        sense="R1",
        module={"x180": "$capture_driver.pulses.pi"},
    )
    context = capture(source).bind_aliases(
        (AliasRow("capture_driver", "Q1"),), declarations()
    )
    assert context.library_key("$capture_driver.module.x180") == "Q1.pulses.pi"
    assert context.value("$capture_driver.module.x180.gain") == 0.25
    assert context.value("$capture_driver.pulses.pi.waveform.length") == 0.05
    assert context.lookup_point("$capture_driver.rate") == DocumentValue(True, 4300.0)


def test_wrong_kind_authored_alias_is_reported_only_when_consumed(
    source: tuple[PointDocument, LibrarySnapshot],
) -> None:
    context = capture(source).bind_aliases(
        (AliasRow("capture_driver", "R1"),), declarations()
    )
    assert context.value("Q1.rate") == 4300.0
    with pytest.raises(RoleResolutionError) as caught:
        context.value("$capture_driver.rate")
    assert caught.value.role == "capture_driver"
    assert caught.value.required_kind == "capture/drive"
    with pytest.raises(RoleResolutionError):
        _ = context.roles


def test_alias_projection_is_fixed_and_ignores_consumer_kind_constraints(
    source: tuple[PointDocument, LibrarySnapshot],
) -> None:
    point, _ = source
    rows = (
        AliasRow("capture_driver", "R1"),
        AliasRow("drive", "Q1"),
        AliasRow("sensor", "$drive.sense"),
        AliasRow("unbound", ""),
        AliasRow("missing", "Absent"),
    )
    context = capture(source).bind_aliases(rows, declarations())
    # Mutating the supplied document cannot change this capture's resolution.
    point.components["Q1"] = point.components["Q1"].model_copy(update={"sense": "R2"})
    projection = context.resolved_aliases()
    assert [(row.alias, row.value, row.resolved) for row in projection] == [
        ("capture_driver", "R1", "R1"),
        ("drive", "Q1", "Q1"),
        ("sensor", "$drive.sense", "R1"),
        ("unbound", "", None),
        ("missing", "Absent", "Absent"),
    ]
    assert [row.error for row in projection[:3]] == [None, None, None]
    assert "$unbound" in str(projection[3].error)
    assert "not bound" in str(projection[3].error)
    assert projection[4].error is None
    assert context.aliases == rows


def test_alias_projection_source_fault_fails_the_whole_read(
    source: tuple[PointDocument, LibrarySnapshot],
    tmp_path: Path,
) -> None:
    fault = ContextSourceFault(
        "point", tmp_path / "point.yaml", "OSError", "source fault"
    )
    context = capture(
        source,
        roles=RoleBinding(
            focus=None, declarations={}, names={}, aliases=(AliasRow("drive", "Q1"),)
        ),
        faults=(fault,),
    )
    with pytest.raises(ContextSourceUnavailableError):
        context.resolved_aliases()


@pytest.mark.parametrize("authored", [True, False], ids=["run", "load"])
@pytest.mark.parametrize("role_target", ["coupler", "Q1"])
def test_library_key_keeps_bare_component_distinct_from_same_named_role(
    source: tuple[PointDocument, LibrarySnapshot], authored: bool, role_target: str
) -> None:
    point, library = source
    point.components["coupler"] = point.components["Q2"].model_copy(deep=True)
    context = capture(
        (point, library),
        roles=RoleBinding(
            "Q1",
            {"coupler": RoleSpec("capture/drive")},
            {"coupler": role_target},
            aliases=(AliasRow("coupler", role_target),) if authored else None,
        ),
    )
    point.components.clear()
    assert context.library_key("coupler.pulses.cz_f2") == "coupler.pulses.cz_f2"
    assert context.resolve_path("coupler.rate") == "coupler.rate"
    assert context.value("coupler.rate") == 5100.0
    assert context.library_key("$coupler.pulses.cz_f2") == f"{role_target}.pulses.cz_f2"
    assert context.value("$coupler.rate") == (
        5100.0 if role_target == "coupler" else 4300.0
    )
