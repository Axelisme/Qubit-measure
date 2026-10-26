"""Fixed parameter projection, role capture and read capability contracts."""

from collections.abc import Generator, MutableMapping
from pathlib import Path

import pytest
from zcu_tools.cfg_model import ConstWaveformCfg, PulseSpec
from zcu_tools.gui.expected_error import ExpectedErrorCategory, InvalidInputError
from zcu_tools.gui.session.aliases import AliasRow, AliasTable
from zcu_tools.gui.session.context_types import (
    ContextSourceFault,
    ContextSourceUnavailableError,
    SavedPoint,
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
from zcu_tools.resources.entry.schema import PointDocument, PointGeneral


class CaptureDrive(ComponentSchema):
    rate: float
    sense: str
    module: dict[str, str | None]


class CaptureSensor(ComponentSchema):
    rate: float


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
                entry_id="00000000-0000-0000-0000-000000000001",
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
        target=SavedPoint(entry_id="entry", point="working"),
        roles=roles,
        source_faults=faults,
    )


def declarations() -> dict[str, RoleSpec]:
    return {
        "capture_driver": RoleSpec("capture/drive"),
        "capture_sensor": RoleSpec("capture/sensor", "capture_driver.sense"),
    }


def test_point_capture_is_detached_and_preserves_target(
    source: tuple[PointDocument, LibrarySnapshot],
) -> None:
    context = capture(source)
    source[0].general.ext.clear()
    copied = context.document()
    copied.general.ext.clear()
    assert context.value("Q1.rate") == 4300.0
    assert context.value("general.ext.nested.number") == 2.0
    assert context.target == SavedPoint(entry_id="entry", point="working")
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
        target=SavedPoint(entry_id="entry", point="working"),
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
    if method == "library":
        _ = context.library
    elif method == "document":
        context.document()
    elif method == "bind_roles":
        context.bind_roles({}, focus=None, explicit={})
    else:
        getattr(context, method)("Q1.rate")


@pytest.mark.parametrize(
    "method",
    [
        "document",
        "library",
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
