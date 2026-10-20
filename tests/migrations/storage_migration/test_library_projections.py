"""Default-profile library projection through build_mapping and migrate_storage.

These migration contracts retire with the offline migrations package.
"""

import json
from collections.abc import Generator
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import Field
from zcu_tools.datafile import JsonObject
from zcu_tools.resources.document_store import YamlMap
from zcu_tools.resources.entry import (
    ComponentSchema,
    ResultEntry,
    RoleSpec,
    component_registry,
    document_value,
    role_registry,
    snapshot,
    source_path,
)

from migrations.storage_mapping import build_mapping
from migrations.storage_migration import (
    MigrationMapping,
    MigrationReport,
    MigrationRequest,
    migrate_storage,
)
from tests.migrations.storage_migration.fakes import (
    assert_difference,
    noop_validation,
    validate_cfg,
    write_meta,
)
from tests.resources.entry.fakes import registry_state, restore_registry


class ProjectionComponent(ComponentSchema):
    """Synthetic fixed-profile component with independent slots and raw extensions."""

    resonator: str | None = None
    module: dict[str, str] = Field(default_factory=dict)
    ext: YamlMap = Field(default_factory=dict)


@pytest.fixture(scope="module", autouse=True)
def profile_registry() -> Generator[None]:
    before = registry_state()
    try:
        for kind in component_registry.kinds():
            component_registry.unregister(kind)
        for role in role_registry.names():
            role_registry.unregister(role)
        role_registry.configure(shorthand=(), focus_kinds=())
        for kind in ("resonator", "qubit/fluxonium", "amplifier/jpa"):
            component_registry.register(kind, ProjectionComponent)
        role_registry.register("qubit", RoleSpec("qubit/*"))
        role_registry.register("resonator", RoleSpec("resonator"))
        expected = registry_state()
        yield
        assert registry_state() == expected, "projection module polluted registries"
    finally:
        restore_registry(before)
        assert registry_state() == before


@pytest.fixture(autouse=True)
def projection_registry_guard(request: pytest.FixtureRequest) -> Generator[None]:
    before = registry_state()
    yield
    assert registry_state() == before, f"registry polluter: {request.node.nodeid}"


@pytest.fixture
def default_mapping() -> MigrationMapping:
    return build_mapping(qubit_kind="qubit/fluxonium", data_schemas={}, native_tags={})


def _convert(request: MigrationRequest, mapping: MigrationMapping) -> MigrationReport:
    return migrate_storage(
        request,
        mapping=mapping,
        validate_cfg=validate_cfg,
        validate_native=noop_validation,
    )


def _entry(request: MigrationRequest) -> ResultEntry:
    return ResultEntry.open(
        request.name,
        result_root=request.results_root,
        database_root=request.database_root,
    )


def _pulse(gain: float, waveform: JsonObject | str) -> JsonObject:
    return {
        "type": "pulse",
        "ch": 0,
        "nqz": 1,
        "freq": 100.0,
        "gain": gain,
        "waveform": waveform,
    }


def _payload() -> JsonObject:
    return {
        "modules": {
            "pi_amp": _pulse(0.125, "qub_cos"),
            "pi_len": _pulse(0.9, {"style": "const", "length": 0.35}),
            "pi2_amp": _pulse(0.25, "ro_waveform"),
            "pi2_len": _pulse(0.8, {"style": "const", "length": 0.45}),
        },
        "waveforms": {
            "mist_waveform": {"style": "const", "length": 1.0, "unknown.note": "keep"},
            "qub_cos": {"style": "cosine", "length": 2.0},
            "qub_flat": {
                "style": "flat_top",
                "length": 3.0,
                "raise_waveform": "qub_cos",
            },
            "ro_waveform": {"style": "const", "length": 4.0},
        },
    }


def _library(meta: Path, payload: JsonObject) -> Path:
    path = meta.with_name("module_cfg.yaml")
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


_SCALARS: JsonObject = {
    "pi_gain": 0.125,
    "pi_len": 0.35,
    "pi2_gain": 0.25,
    "pi2_len": 0.45,
}
_LEAVES = {
    "pi_gain": "Q1.pulses.pi_amp.gain",
    "pi_len": "Q1.pulses.pi_len.waveform.length",
    "pi2_gain": "Q1.pulses.pi2_amp.gain",
    "pi2_len": "Q1.pulses.pi2_len.waveform.length",
}


@pytest.mark.parametrize("label", ["", "exps/context"])
def test_default_waveforms_publish_with_canonical_keys_and_only_unknown_raw(
    request_data: MigrationRequest, default_mapping: MigrationMapping, label: str
) -> None:
    request = replace(request_data, part="parameters")
    source = _library(write_meta(request, label, {}), _payload())
    dry = _convert(replace(request, dry_run=True), default_mapping)
    planned = {
        row.new_path
        for row in dry.key_mappings
        if row.old_file == source
        and row.library_kind == "waveform"
        and row.action == "module"
    }
    assert planned == {
        "Q1.waveforms.mist_waveform",
        "Q1.waveforms.qub_cos",
        "Q1.waveforms.qub_flat",
        "R1.waveforms.ro_waveform",
    }
    assert not request.results_root.exists()
    report = _convert(request, default_mapping)
    entry = _entry(request)
    point = "context" if label else None
    library = entry.module_library(point).snapshot()
    assert set(library.waveform_keys()) == planned
    for key, length in (
        ("Q1.waveforms.mist_waveform", 1.0),
        ("Q1.waveforms.qub_cos", 2.0),
        ("Q1.waveforms.qub_flat", 3.0),
        ("R1.waveforms.ro_waveform", 4.0),
    ):
        assert library.get_waveform(key).length == length
    view = entry.setup if point is None else entry.use_point(point)
    assert document_value(snapshot(view), "Q1.module") == {
        "x180": "Q1.pulses.pi_amp",
        "x90": "Q1.pulses.pi2_amp",
    }
    assert document_value(snapshot(view), "general.ext.legacy_module_cfg") == {
        "waveforms": {"mist_waveform": {"unknown.note": "keep"}}
    }
    preserved = [
        row
        for row in report.key_mappings
        if row.old_file == source and row.action == "preserved"
    ]
    assert len(preserved) == 1
    assert preserved[0].source_keys == ("waveforms", "mist_waveform", "unknown.note")
    assert preserved[0].destination_keys == (
        "general",
        "ext",
        "legacy_module_cfg",
        "waveforms",
        "mist_waveform",
        "unknown.note",
    )
    assert not report.pending


@pytest.mark.parametrize("label", ["", "exps/context"])
def test_pi_scalars_deduplicate_only_their_explicit_candidate_and_report_actual_leaf(
    request_data: MigrationRequest, default_mapping: MigrationMapping, label: str
) -> None:
    request = replace(request_data, part="all")
    source = write_meta(request, label, _SCALARS)
    _library(source, _payload())
    dry = _convert(replace(request, dry_run=True), default_mapping)
    report = _convert(request, default_mapping)
    entry = _entry(request)
    point = "context" if label else None
    library = entry.module_library(point)
    for result in (dry, report):
        assert not [item for item in result.pending if item.source == source]
        rows = {
            row.old_key: row for row in result.key_mappings if row.old_file == source
        }
        for key, leaf in _LEAVES.items():
            row = rows[key]
            assert row.action == "remove"
            assert row.new_file == library.source_path
            assert row.new_path == leaf
            assert row.destination_keys == tuple(leaf.split("."))
            assert row.library_kind == "module"
            assert row.source_keys == (key,)
            assert library.snapshot().value(leaf) == _SCALARS[key]
    view = entry.setup if point is None else entry.use_point(point)
    assert document_value(snapshot(view), "Q1.ext") == {"readout_cal": {}}
    # The exact owner sidecar protects reads too; only its empty bytes are stable.
    sidecar = library.source_path.with_suffix(".yaml.lock")
    assert sidecar.read_bytes() == b""
    files = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in entry.result_path.rglob("*")
        if path.is_file() and path != sidecar
    }
    rerun = _convert(request, default_mapping)
    assert rerun.skipped
    assert sidecar.read_bytes() == b""
    assert files == {
        path: (path.read_bytes(), path.stat().st_mtime_ns) for path in files
    }
    with library.edit() as draft:
        pulse = library.snapshot().module_spec("Q1.pulses.pi_len")
        assert pulse.type == "pulse"
        draft.set_module("Q1.pulses.pi_len", pulse.model_copy(update={"gain": 0.7}))
    assert_difference(
        lambda: _convert(request, default_mapping), request, [library.source_path]
    )


@pytest.mark.parametrize(
    "mode", ["different", "failed", "missing", "no-library", "setup-only"]
)
def test_unmatched_pi_scalars_keep_original_values_in_their_context(
    request_data: MigrationRequest, default_mapping: MigrationMapping, mode: str
) -> None:
    request = replace(request_data, part="parameters")
    payload = _payload()
    modules = payload["modules"]
    assert isinstance(modules, dict)
    values = dict(_SCALARS)
    if mode == "different":
        # These equal unrelated leaves, not the declared candidates.
        values = {"pi_gain": 0.9, "pi_len": 2.0, "pi2_gain": 0.8, "pi2_len": 4.0}
    elif mode in ("failed", "missing"):
        for name in ("pi_amp", "pi_len", "pi2_amp", "pi2_len"):
            if mode == "failed":
                modules[name] = {"type": "pulse", "gain": 0.125}
            else:
                del modules[name]
        # Another candidate in the same slot must not swallow the source values.
        modules["pi_len"] = _pulse(0.8, {"style": "const", "length": 9.0})
        modules["pi2_len"] = _pulse(0.9, {"style": "const", "length": 8.0})
    if mode == "setup-only":
        _library(write_meta(request, "", {}), _payload())
    source = write_meta(request, "exps/context", values)
    if mode not in ("no-library", "setup-only"):
        _library(source, payload)
    dry = _convert(replace(request, dry_run=True), default_mapping)
    report = _convert(request, default_mapping)
    view = _entry(request).use_point("context")
    ext = document_value(snapshot(view), "Q1.ext")
    assert ext == {"readout_cal": {}, **values}
    for result in (dry, report):
        assert not [item for item in result.pending if item.source == source]
        rows = {
            row.old_key: row for row in result.key_mappings if row.old_file == source
        }
        assert set(rows) == set(values)
        for key in values:
            assert rows[key].action == "value"
            assert rows[key].new_path == f"Q1.ext.{key}"
            assert rows[key].new_file == source_path(view)
