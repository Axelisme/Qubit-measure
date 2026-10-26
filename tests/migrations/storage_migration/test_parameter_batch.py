"""Caller-owned whole-document conversion before current-schema validation."""

from dataclasses import replace

import pytest
from zcu_tools.resources.entry import document_value, snapshot
from zcu_tools.resources.parameter_upgrade import ParameterDocument

from migrations.storage_migration import KeyRule, MigrationMapping, MigrationRequest
from tests.migrations.storage_migration.fakes import write_meta
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
