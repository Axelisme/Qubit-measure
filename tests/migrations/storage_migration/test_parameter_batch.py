"""Caller-owned whole-document conversion before current-schema validation."""

from dataclasses import replace

import pytest
from pydantic import field_validator
from zcu_tools.datafile import JsonObject
from zcu_tools.resources.entry import component_registry, document_value, snapshot
from zcu_tools.resources.parameter_upgrade import ParameterDocument

from migrations.storage_migration import KeyRule, MigrationMapping, MigrationRequest
from tests.migrations.storage_migration.fakes import SyntheticProbe, write_meta
from tests.migrations.storage_migration.params_support import migrate, open_entry


@pytest.mark.parametrize("dry_run", [False, True])
def test_seed_batch_converts_before_current_schema_validation(
    request_data: MigrationRequest, mapping: MigrationMapping, dry_run: bool
) -> None:
    def transform(document: ParameterDocument) -> ParameterDocument:
        components = document.data["components"]
        if not isinstance(components, dict):
            raise ValueError("components must be a mapping")
        probe = components["C1"]
        if not isinstance(probe, dict):
            raise ValueError("C1 must be a mapping")
        probe["frequency"] = probe.pop("historical_frequency")
        return document

    mapping = replace(
        mapping,
        components={
            "C1": {"kind": "migration_test_probe", "historical_frequency": 3.0}
        },
        parameter_transform=transform,
    )
    request_data = replace(request_data, dry_run=dry_run)
    write_meta(request_data, "", {})
    migrate(request_data, mapping)
    if dry_run:
        assert not (request_data.results_root / request_data.name).exists()
    else:
        setup = snapshot(open_entry(request_data).setup)
        assert document_value(setup, "C1.frequency") == 3.0


@pytest.mark.parametrize("dry_run", [False, True])
def test_value_batch_converts_whole_group_before_current_validation(
    request_data: MigrationRequest, mapping: MigrationMapping, dry_run: bool
) -> None:
    seen: list[float] = []

    def transform(document: ParameterDocument) -> ParameterDocument:
        components = document.data["components"]
        if not isinstance(components, dict):
            raise ValueError("components must be a mapping")
        probe = components["C1"]
        if not isinstance(probe, dict):
            raise ValueError("C1 must be a mapping")
        if "historical_frequency" in probe:
            value = probe.pop("historical_frequency")
            if not isinstance(value, float):
                raise ValueError("historical_frequency must be numeric")
            second = probe.pop("historical_offset")
            if not isinstance(second, float):
                raise ValueError("historical_offset must be numeric")
            seen.append(value + second)
            probe["frequency"] = value + second
        return document

    mapping = replace(
        mapping,
        parameter_transform=transform,
        rules=(
            KeyRule(
                old_key="flux_unit",
                target_path="general.ext.unit",
                action="value",
                reason="Synthetic unit",
            ),
            KeyRule(
                old_key="value",
                target_path="C1.historical_frequency",
                action="value",
                requires_keys=("flux_unit",),
                reason="Synthetic historical field",
            ),
            KeyRule(
                old_key="offset",
                target_path="C1.historical_offset",
                action="value",
                requires_keys=("flux_unit",),
                reason="Second field must be present in the same complete batch",
            ),
        ),
    )
    request_data = replace(request_data, dry_run=dry_run)
    write_meta(request_data, "", {"flux_unit": "V", "value": 4.0, "offset": 5.0})
    report = migrate(request_data, mapping)
    assert seen == [9.0]
    assert report.pending == ()
    if dry_run:
        assert not (request_data.results_root / request_data.name).exists()
    else:
        setup = snapshot(open_entry(request_data).setup)
        assert document_value(setup, "C1.frequency") == 9.0
        assert document_value(setup, "general.ext.unit") == "V"


@pytest.mark.parametrize("dry_run", [False, True])
def test_conversion_notes_reach_report_for_seed_setup_and_point_batches(
    request_data: MigrationRequest, mapping: MigrationMapping, dry_run: bool
) -> None:
    def transform(document: ParameterDocument) -> ParameterDocument:
        return replace(document, notes=("C1.frequency: check the declared source",))

    mapping = replace(
        mapping,
        parameter_transform=transform,
        rules=(
            KeyRule(
                old_key="flux_unit",
                target_path="general.ext.unit",
                action="value",
                reason="Synthetic unit",
            ),
            KeyRule(
                old_key="value",
                target_path="C1.frequency",
                action="value",
                requires_keys=("flux_unit",),
                reason="Synthetic coordinate",
            ),
        ),
    )
    request_data = replace(request_data, dry_run=dry_run)
    write_meta(request_data, "", {"flux_unit": "V", "value": 4.0})
    write_meta(request_data, "exps/point-a", {"flux_unit": "V", "value": 5.0})
    report = migrate(request_data, mapping)
    destination = request_data.results_root / request_data.name
    assert set(report.notes) == {
        f"{destination / 'setup.yaml'}: C1.frequency: check the declared source",
        f"{destination / 'points/point-a/point.yaml'}: C1.frequency: check the declared source",
    }
    assert report.pending == ()


class OptionalSettingProbe(SyntheticProbe):
    """Synthetic schema allowing omitted settings but rejecting explicit null."""

    kind: str = "migration_optional_probe"
    optional_setting: int | None = None

    @field_validator("optional_setting")
    @classmethod
    def reject_explicit_null(cls, value: int | None) -> int:
        """Keep an unknown setting omitted rather than accepting a supplied null."""
        if value is None:
            raise ValueError("a supplied optional_setting cannot be null")
        return value


@pytest.mark.parametrize("explicit_null", [False, True])
def test_batch_preserves_omitted_setting_without_dropping_explicit_null(
    request_data: MigrationRequest,
    mapping: MigrationMapping,
    explicit_null: bool,
) -> None:
    def transform(document: ParameterDocument) -> ParameterDocument:
        components = document.data["components"]
        if not isinstance(components, dict):
            raise ValueError("components must be a mapping")
        probe = components["C1"]
        if not isinstance(probe, dict):
            raise ValueError("C1 must be a mapping")
        if "historical_frequency" in probe:
            probe["frequency"] = probe.pop("historical_frequency")
        return document

    mapping = replace(
        mapping,
        components={"C1": {"kind": "migration_optional_probe", "frequency": 1.0}},
        parameter_transform=transform,
        rules=(
            KeyRule(
                old_key="flux_unit",
                target_path="general.ext.unit",
                action="value",
                reason="Synthetic group coordinate",
            ),
            KeyRule(
                old_key="value",
                target_path="C1.historical_frequency",
                action="value",
                requires_keys=("flux_unit",),
                reason="Convert the whole historical group before validation",
            ),
            KeyRule(
                old_key="setting",
                target_path="C1.optional_setting",
                action="value",
                requires_keys=("flux_unit",),
                reason="An explicit null must reach the schema unchanged",
            ),
        ),
    )
    values: JsonObject = {"flux_unit": "V", "value": 9.0}
    if explicit_null:
        values["setting"] = None
    write_meta(request_data, "", values)
    component_registry.register("migration_optional_probe", OptionalSettingProbe)
    try:
        report = migrate(request_data, mapping)
        setup = snapshot(open_entry(request_data).setup)
        if explicit_null:
            assert report.pending
            assert "optional_setting cannot be null" in report.pending[0].reason
            assert document_value(setup, "C1.frequency") == 1.0
            assert document_value(setup, "general.ext.legacy_meta_info.setting") is None
        else:
            assert report.pending == (), "\n".join(
                item.reason for item in report.pending
            )
            assert document_value(setup, "C1.frequency") == 9.0
            assert document_value(setup, "general.ext.unit") == "V"
    finally:
        component_registry.unregister("migration_optional_probe")
