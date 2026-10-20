"""Global, point-local and independent JPA unit evidence through storage migration."""

from dataclasses import replace

import pytest
from pydantic import JsonValue, TypeAdapter
from zcu_tools.resources.entry import document_value, snapshot

from migrations.storage_migration import (
    KeyRule,
    MigrationInputError,
    MigrationMapping,
    MigrationReport,
    MigrationRequest,
)
from tests.migrations.storage_migration.fakes import write_meta
from tests.migrations.storage_migration.params_support import (
    backup_path,
    declare_request,
    migrate,
    open_entry,
    write_params,
)

pytestmark = pytest.mark.usefixtures("analysis_kind")


def test_missing_units_drive_alias_and_module_scalars_remain_observable(
    request_data: MigrationRequest, metadata_mapping: MigrationMapping
) -> None:
    adapter = TypeAdapter(MigrationRequest)
    request = adapter.validate_python(
        {**adapter.dump_python(request_data), "assume_flux_unit": "mA"}
    )
    write_meta(request, "", {"qub_ch": 3, "pi_gain": 0.2, "flx_half": 0.5})
    write_meta(request, "bias", {"qub_ch": 4, "qub_0_1_ch": 2, "pi2_len": 0.03})
    report = migrate(request, metadata_mapping)
    setup = snapshot(open_entry(request).setup)
    assert document_value(setup, "Q1.wiring") == {"ch": 3}
    assert document_value(setup, "Q1.ext.pi_gain") == 0.2
    assert document_value(setup, "Q1.flux_half") == 0.5
    assert document_value(setup, "general.ext.flux_unit") == "mA"
    point = snapshot(open_entry(request).use_point("bias"))
    assert document_value(point, "Q1.wiring.ch") == 2
    assert document_value(point, "Q1.ext.pi2_len") == 0.03
    assert any("假設單位" in row.reason for row in report.key_mappings)


def test_unbound_pi_scalars_are_saved_in_component_extensions(
    request_data: MigrationRequest, metadata_mapping: MigrationMapping
) -> None:
    write_meta(request_data, "", {"pi_gain": 0.2, "pi_len": 0.05})
    migrate(request_data, metadata_mapping)
    setup = snapshot(open_entry(request_data).setup)
    assert document_value(setup, "Q1.ext") == {"pi_gain": 0.2, "pi_len": 0.05}


def test_assumption_supplies_missing_metadata_evidence_without_changing_values(
    request_data: MigrationRequest, metadata_mapping: MigrationMapping
) -> None:
    adapter = TypeAdapter(MigrationRequest)
    request = adapter.validate_python(
        {**adapter.dump_python(request_data), "assume_flux_unit": "mA"}
    )
    write_meta(request, "", {"flx_half": 0.5, "cur_A": 0.7})
    report = migrate(request, metadata_mapping)
    setup = snapshot(open_entry(request).setup)
    assert document_value(setup, "general.ext") == {
        "flux_unit": "mA",
        "flux_value": 0.7,
    }
    assert document_value(setup, "Q1.flux_half") == 0.5
    assert document_value(setup, "general.ext.flux_value") == 0.7
    assert any("假設單位" in row.reason for row in report.key_mappings)


@pytest.mark.parametrize("unit_source", ["meta_info.json", "params.json"])
def test_entry_explicit_unit_beats_assumption_and_reports_its_file(
    request_data: MigrationRequest,
    metadata_mapping: MigrationMapping,
    unit_source: str,
) -> None:
    adapter = TypeAdapter(MigrationRequest)
    request = adapter.validate_python(
        {**adapter.dump_python(request_data), "assume_flux_unit": "mA"}
    )
    root = request.result_root / "chip/qubit"
    (root / unit_source).write_text('{"flux_unit": "V"}', encoding="utf-8")
    write_meta(request, "bias", {"flx_half": 0.3})
    report = migrate(request, metadata_mapping)
    point = snapshot(open_entry(request).use_point("bias"))
    assert document_value(point, "general.ext.flux_unit") == "V"
    assert document_value(point, "Q1.flux_half") == 0.3
    assert any(
        row.old_key == "flx_half" and str(root / unit_source) in row.reason
        for row in report.key_mappings
    )


def test_point_unit_override_clears_only_unreplaced_inherited_group_values(
    request_data: MigrationRequest, metadata_mapping: MigrationMapping
) -> None:
    metadata_mapping = replace(
        metadata_mapping,
        rules=(
            *metadata_mapping.rules,
            KeyRule(
                old_key="period_error",
                target_path="Q1.flux_period",
                action="stderr",
                reason="Synthetic uncertainty in the value working unit",
            ),
        ),
    )
    write_meta(
        request_data,
        "",
        {
            "flux_unit": "V",
            "local_flux_unit": "A",
            "flx_half": 0.2,
            "flx_period": 0.4,
            "period_error": 0.02,
            "flx_int": 0.1,
            "cur_A": 0.5,
            "flux_bias": 0.6,
            "jpa_flux_unit": "V",
            "best_jpa_flux": 1.0,
        },
    )
    point_meta = write_meta(
        request_data,
        "bias",
        {"flux_unit": "A", "flx_half": 0.3, "local_flux_unit": "V"},
    )
    report = migrate(request_data, metadata_mapping)
    point = snapshot(open_entry(request_data).use_point("bias"))
    assert document_value(point, "Q1.flux_half") == 0.3
    assert document_value(point, "Q1.flux_period") is None
    assert open_entry(request_data).setup.meta("Q1.flux_period") == {"stderr": 0.02}
    assert open_entry(request_data).use_point("bias").meta("Q1.flux_period") == {}
    assert document_value(point, "Q1.flux_int") is None
    assert document_value(point, "general.ext") == {"flux_unit": "A"}
    assert document_value(point, "Q1.ext") == {}
    assert document_value(point, "Q1.flux_unit") == "V"
    assert document_value(point, "J1.flux") == 1.0
    assert document_value(point, "J1.flux_unit") == "V"
    assert not any(item.source == point_meta for item in report.pending)


def migrate_with_unit_pending(
    request: MigrationRequest, mapping: MigrationMapping
) -> MigrationReport:
    """Unit incompatibility is located pending, not an entry-wide schema abort."""
    try:
        return migrate(request, mapping)
    except ValueError as exc:
        pytest.fail(f"Unit incompatibility must become located pending: {exc}")


def test_jpa_does_not_borrow_params_unit_and_rejects_unsupported_owner_unit(
    request_data: MigrationRequest, metadata_mapping: MigrationMapping
) -> None:
    write_params(request_data, {"flux_unit": "A"})
    source = write_meta(
        request_data,
        "",
        {
            "best_jpa_flux": 0.2,
            "local_flux_unit": "mA",
            "flx_half": 0.3,
        },
    )
    report = migrate_with_unit_pending(request_data, metadata_mapping)
    setup = snapshot(open_entry(request_data).setup)
    assert any(
        item.source == source and item.location == "best_jpa_flux"
        for item in report.pending
    )
    assert any(
        item.source == source and item.location == "local_flux_unit"
        for item in report.pending
    )
    assert document_value(setup, "Q1.flux_half") == 0.3
    assert getattr(open_entry(request_data).setup.J1, "flux", None) is None


def test_invalid_jpa_unit_keeps_whole_group_pending_without_conversion(
    request_data: MigrationRequest, metadata_mapping: MigrationMapping
) -> None:
    source = write_meta(
        request_data,
        "",
        {
            "jpa_flux_unit": "mA",
            "best_jpa_flux": 2.0,
        },
    )
    report = migrate_with_unit_pending(request_data, metadata_mapping)
    assert {item.location for item in report.pending if item.source == source} >= {
        "jpa_flux_unit",
        "best_jpa_flux",
    }
    assert getattr(open_entry(request_data).setup.J1, "flux", None) is None


def test_conflicting_entry_units_block_values_instead_of_using_assumption(
    request_data: MigrationRequest, metadata_mapping: MigrationMapping
) -> None:
    adapter = TypeAdapter(MigrationRequest)
    request = adapter.validate_python(
        {**adapter.dump_python(request_data), "assume_flux_unit": "mA"}
    )
    write_params(request, {"flux_unit": "A"})
    source = write_meta(request, "", {"flux_unit": "V", "flx_half": 0.2})
    report = migrate(request, metadata_mapping)
    setup = snapshot(open_entry(request).setup)
    assert document_value(setup, "general.ext") == {
        "legacy_meta_info": {"flux_unit": "V", "flx_half": 0.2}
    }
    assert any(
        item.source == source
        and item.location == "flx_half"
        and "params.json" in item.reason
        for item in report.pending
    )


def test_jpa_never_borrows_global_assumption(
    request_data: MigrationRequest, metadata_mapping: MigrationMapping
) -> None:
    request = declare_request(request_data, assume_flux_unit="A")
    source = write_meta(request, "", {"cur_A": 0.4, "best_jpa_flux": 0.2})
    report = migrate(request, metadata_mapping)
    assert any(
        item.source == source and item.location == "best_jpa_flux"
        for item in report.pending
    )
    assert getattr(open_entry(request).setup.J1, "flux", None) is None
    assert document_value(snapshot(open_entry(request).setup), "general.ext") == {
        "flux_unit": "A",
        "flux_value": 0.4,
        "legacy_meta_info": {"best_jpa_flux": 0.2},
    }


def test_independent_jpa_assumption_publishes_v_without_changing_global_ma(
    request_data: MigrationRequest, metadata_mapping: MigrationMapping
) -> None:
    request = declare_request(
        request_data, assume_flux_unit="mA", assume_jpa_flux_unit="V"
    )
    write_meta(request, "", {"cur_A": 0.4, "best_jpa_flux": 0.2})
    report = migrate(request, metadata_mapping)
    assert getattr(open_entry(request).setup.J1, "flux", None) == 0.2
    setup = snapshot(open_entry(request).setup)
    assert document_value(setup, "J1.flux_unit") == "V"
    assert document_value(setup, "general.ext") == {
        "flux_unit": "mA",
        "flux_value": 0.4,
    }
    assert any(
        row.old_key == "jpa_flux_unit" and "假設單位" in row.reason
        for row in report.key_mappings
    )


@pytest.mark.parametrize("explicit", ["A", "mA", None])
def test_jpa_assumption_never_replaces_explicit_or_invalid_evidence(
    request_data: MigrationRequest,
    metadata_mapping: MigrationMapping,
    explicit: str | None,
) -> None:
    request = declare_request(request_data, assume_jpa_flux_unit="V")
    source = write_meta(request, "", {"jpa_flux_unit": explicit, "best_jpa_flux": 0.2})
    report = migrate_with_unit_pending(request, metadata_mapping)
    if explicit == "A":
        assert (
            document_value(snapshot(open_entry(request).setup), "J1.flux_unit") == "A"
        )
        assert not any(item.source == source for item in report.pending)
    else:
        assert getattr(open_entry(request).setup.J1, "flux", None) is None
        assert any(
            item.source == source and item.location == "best_jpa_flux"
            for item in report.pending
        )


def test_resume_rejects_changed_jpa_assumption_before_writes(
    request_data: MigrationRequest, metadata_mapping: MigrationMapping
) -> None:
    request = declare_request(request_data, assume_jpa_flux_unit="V")
    write_meta(request, "", {"best_jpa_flux": 0.2})
    migrate(request, metadata_mapping)
    before = (request.results_root / request.name / "setup.yaml").read_bytes()
    changed = declare_request(request, assume_jpa_flux_unit="A", resume=True)
    with pytest.raises(MigrationInputError, match="declarations"):
        migrate(changed, metadata_mapping)
    assert (request.results_root / request.name / "setup.yaml").read_bytes() == before


@pytest.mark.parametrize("unit", ["mA", "unknown", "", True])
def test_invalid_jpa_assumption_is_rejected_before_any_entry_or_backup(
    request_data: MigrationRequest, metadata_mapping: MigrationMapping, unit: JsonValue
) -> None:
    with pytest.raises(ValueError, match="assume_jpa_flux_unit"):
        migrate(
            declare_request(request_data, assume_jpa_flux_unit=unit), metadata_mapping
        )
    assert not (request_data.results_root / request_data.name).exists()
    assert not backup_path(request_data).exists()
