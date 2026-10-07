"""Fixed parameter projection, role capture and read capability contracts."""

from collections.abc import Generator, MutableMapping
from pathlib import Path

import pytest
from zcu_tools.cfg_model import ConstWaveformCfg, PulseSpec
from zcu_tools.gui.expected_error import ExpectedErrorCategory, InvalidInputError
from zcu_tools.gui.session.context_types import (
    ContextSourceFault,
    ContextSourceUnavailableError,
    EntryDraft,
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
    ResultEntry,
    RoleResolutionError,
    RoleSpec,
    component_registry,
    role_registry,
    snapshot,
    source_path,
)
from zcu_tools.resources.entry.schema import PointDocument


class CaptureDrive(ComponentSchema):
    rate: float
    sense: str
    module: dict[str, str | None]


class CaptureSensor(ComponentSchema):
    rate: float


@pytest.fixture
def source(tmp_path: Path) -> Generator[tuple[PointDocument, LibrarySnapshot]]:
    before = component_registry.kinds()
    component_registry.register("capture/drive", CaptureDrive)
    component_registry.register("capture/sensor", CaptureSensor)
    role_registry.register("capture_driver", RoleSpec("capture/drive"))
    role_registry.register(
        "capture_sensor", RoleSpec("capture/sensor", "capture_driver.sense")
    )
    try:
        entry = ResultEntry.create(
            "projection",
            result_root=tmp_path / "results",
            database_root=tmp_path / "database",
        )
        for name, rate, sense in (("Q1", 4300.0, "R1"), ("Q2", 5100.0, "R2")):
            entry.setup.add_component(
                name,
                kind="capture/drive",
                rate=rate,
                sense=sense,
                module={
                    "x180": "capture_driver.pulses.pi",
                    "empty": None,
                    "dangling": "Q1.pulses.missing",
                },
            )
        for name in ("R1", "R2"):
            entry.setup.add_component(name, kind="capture/sensor", rate=6500.0)
        point = entry.new_point("working")
        document = snapshot(point)
        document.general.ext = {"nullable": None, "nested": {"number": 2.0}}
        library = LibrarySnapshot(
            ModuleLibraryDocument(
                format="zcu.module-library",
                format_version="1.0",
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
        # All following reads must use the capture, not reopen the point file.
        source_path(point).unlink()
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
    assert bound.value("capture_driver.rate") == 4300.0
    assert rebound.value("capture_driver.rate") == 5100.0
    assert rebound.value("capture_sensor.rate") == 6500.0
    assert bound.target == context.target
    assert rebound.target == context.target
    assert context.roles is None


def test_draft_projection_keeps_its_entry_target(
    source: tuple[PointDocument, LibrarySnapshot],
) -> None:
    point, library = source
    context = ParameterContext(
        point=point, library=library, target=EntryDraft(entry_id="entry"), roles=None
    )
    assert context.target == EntryDraft(entry_id="entry")
    assert context.value("Q1.rate") == 4300.0


def test_metadata_routes_to_real_point_or_library_source(
    source: tuple[PointDocument, LibrarySnapshot],
) -> None:
    point, library = source
    point.provenance["Q1.rate"] = {
        "source": "manual",
        "kind": None,
        "run_id": None,
        "at": "2026-10-04T00:00:00Z",
        "stderr": 0.5,
    }
    library = library.replaced_module(
        "Q1.pulses.pi", library.module_spec("Q1.pulses.pi")
    )
    context = capture((point, library)).bind_roles(
        declarations(), focus="Q1", explicit={}
    )
    actual = context.meta("capture_driver.rate")
    assert actual is not None
    assert actual.stderr == 0.5
    assert context.meta("capture_driver.rate") == actual
    library_source = context.meta("capture_driver.module.x180.gain")
    assert library_source == library.meta("Q1.pulses.pi.gain")
    assert library_source is not None
    assert library_source.source == "manual"
    slot_source = context.meta("Q1.module.x180")
    assert slot_source is not None
    assert slot_source.at == point.provenance["Q1.module.x180"]["at"]
    assert context.meta("general.ext.nullable") is None
    assert context.meta("Q1.absent") is None


def test_empty_roles_do_not_use_notebook_shorthand(
    source: tuple[PointDocument, LibrarySnapshot],
) -> None:
    bound = capture(source).bind_roles({}, focus="Q1", explicit={})
    assert bound.roles == RoleBinding("Q1", {}, {})
    with pytest.raises(RoleResolutionError):
        bound.value("capture_driver.rate")


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
    assert context.value("capture_driver.rate") == 5100.0
    assert context.value("capture_sensor.rate") == 6500.0
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
    assert context.library_key("capture_driver.pulses.pi") == "Q1.pulses.pi"
    assert context.library_key("capture_driver.module.x180") == "Q1.pulses.pi"
    assert context.value("capture_driver.module.x180") == "capture_driver.pulses.pi"
    assert context.value("capture_driver.module.x180.gain") == 0.25
    assert context.lookup("Q1.pulses.missing.gain") == DocumentValue(False, None)
    assert context.lookup("Q1.pulses.pi.mixer_freq") == DocumentValue(True, None)
    assert context.value("Q1.pulses.pi.waveform.length") == 0.05
    assert context.meta("Q1.pulses.pi.gain") is None
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
        "meta",
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
