from __future__ import annotations

import json
from pathlib import Path

import pytest
import zcu_tools.gui.result_scope as result_scope_mod
from zcu_tools.gui.result_scope import (
    ResultScopeError,
    ResultScopeManager,
    migrate_params_v0_to_v1_project_info,
    read_params_identity,
)


def test_result_scope_scans_canonical_params(tmp_path) -> None:
    params_path = tmp_path / "result" / "ChipA" / "Q1" / "params.json"
    params_path.parent.mkdir(parents=True)
    params_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "project": {"chip_name": "ChipA", "qubit_name": "Q1"},
                "name": "ChipA/Q1",
            }
        ),
        encoding="utf8",
    )

    scopes = ResultScopeManager(tmp_path).list_scopes()

    assert len(scopes) == 1
    assert scopes[0].chip_name == "ChipA"
    assert scopes[0].qub_name == "Q1"
    assert scopes[0].result_dir == str(params_path.parent.resolve())


def test_list_scopes_uses_snapshot_cache_until_refresh(tmp_path, monkeypatch) -> None:
    params_path = tmp_path / "result" / "ChipA" / "Q1" / "params.json"
    params_path.parent.mkdir(parents=True)
    params_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "project": {"chip_name": "ChipA", "qubit_name": "Q1"},
            }
        ),
        encoding="utf8",
    )
    calls: list[str | Path] = []
    original = result_scope_mod.migrate_params_v0_to_v1_project_info

    def counting_migrate(
        params_path: str | Path, *, result_root: str | Path
    ) -> tuple[str, str]:
        calls.append(params_path)
        return original(params_path, result_root=result_root)

    monkeypatch.setattr(
        result_scope_mod, "migrate_params_v0_to_v1_project_info", counting_migrate
    )
    manager = ResultScopeManager(tmp_path)

    first = manager.list_scopes()
    second = manager.list_scopes()
    refreshed = manager.list_scopes(refresh=True)

    assert first == second == refreshed
    assert len(calls) == 2


def test_ensure_scope_with_cached_scope_does_not_rescan(tmp_path, monkeypatch) -> None:
    params_path = tmp_path / "result" / "ChipA" / "Q1" / "params.json"
    params_path.parent.mkdir(parents=True)
    params_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "project": {"chip_name": "ChipA", "qubit_name": "Q1"},
            }
        ),
        encoding="utf8",
    )
    manager = ResultScopeManager(tmp_path)
    cached_scope = manager.list_scopes()[0]

    def fail_migrate(
        _params_path: str | Path, *, result_root: str | Path
    ) -> tuple[str, str]:
        del result_root
        raise AssertionError("cached scope apply should not rescan result root")

    monkeypatch.setattr(
        result_scope_mod, "migrate_params_v0_to_v1_project_info", fail_migrate
    )

    scope = manager.ensure_scope(
        chip_name="ChipA", qub_name="Q1", scope_id=cached_scope.scope_id
    )

    assert scope == cached_scope


def test_generated_scope_invalidates_empty_snapshot_cache(tmp_path) -> None:
    manager = ResultScopeManager(tmp_path)
    assert manager.list_scopes() == ()

    manager.ensure_scope(chip_name="ChipA", qub_name="Q1")

    scopes = manager.list_scopes()
    assert len(scopes) == 1
    assert scopes[0].chip_name == "ChipA"
    assert scopes[0].qub_name == "Q1"


def test_result_scope_reads_legacy_name_in_one_place(tmp_path) -> None:
    params_path = tmp_path / "result" / "legacy" / "params.json"
    params_path.parent.mkdir(parents=True)
    params_path.write_text(json.dumps({"name": "LegacyChip/Q2"}), encoding="utf8")

    assert read_params_identity(params_path) == ("LegacyChip", "Q2")


def test_list_scopes_migrates_two_level_params_from_path(tmp_path) -> None:
    params_path = tmp_path / "result" / "Q3_2D" / "Q1" / "params.json"
    params_path.parent.mkdir(parents=True)
    params_path.write_text(json.dumps({"name": "stale/wrong"}), encoding="utf8")

    scopes = ResultScopeManager(tmp_path).list_scopes()

    assert len(scopes) == 1
    assert scopes[0].chip_name == "Q3_2D"
    assert scopes[0].qub_name == "Q1"
    raw = json.loads(params_path.read_text(encoding="utf8"))
    assert raw["schema_version"] == 1
    assert raw["project"] == {
        "chip_name": "Q3_2D",
        "qubit_name": "Q1",
        "resonator_name": "unknown",
    }
    assert raw["name"] == "Q3_2D/Q1"


def test_list_scopes_migrates_single_level_params_to_chip_equals_qubit(
    tmp_path,
) -> None:
    params_path = tmp_path / "result" / "Si001" / "params.json"
    params_path.parent.mkdir(parents=True)
    params_path.write_text(json.dumps({"name": "Si001"}), encoding="utf8")

    scopes = ResultScopeManager(tmp_path).list_scopes()

    assert len(scopes) == 1
    assert scopes[0].chip_name == "Si001"
    assert scopes[0].qub_name == "Si001"
    raw = json.loads(params_path.read_text(encoding="utf8"))
    assert raw["project"] == {
        "chip_name": "Si001",
        "qubit_name": "Si001",
        "resonator_name": "unknown",
    }
    assert raw["name"] == "Si001/Si001"


def test_migrate_params_uses_unknown_for_non_scope_path(tmp_path) -> None:
    params_path = tmp_path / "result" / "Chip" / "Q1" / "extra" / "params.json"
    params_path.parent.mkdir(parents=True)
    params_path.write_text(json.dumps({"name": "anything"}), encoding="utf8")

    identity = migrate_params_v0_to_v1_project_info(
        params_path, result_root=tmp_path / "result"
    )

    assert identity == ("unknown", "unknown")
    raw = json.loads(params_path.read_text(encoding="utf8"))
    assert raw["project"] == {
        "chip_name": "unknown",
        "qubit_name": "unknown",
        "resonator_name": "unknown",
    }


def test_ensure_scope_migrates_params_without_losing_sections(tmp_path) -> None:
    params_path = tmp_path / "result" / "LegacyChip" / "Q2" / "params.json"
    params_path.parent.mkdir(parents=True)
    params_path.write_text(
        json.dumps({"name": "LegacyChip/Q2", "fluxdep_fit": {"params": {}}}),
        encoding="utf8",
    )

    scope = ResultScopeManager(tmp_path).ensure_scope(
        chip_name="LegacyChip", qub_name="Q2"
    )

    assert scope.params_path == str(params_path.resolve())
    raw = json.loads(params_path.read_text(encoding="utf8"))
    assert raw["project"] == {
        "chip_name": "LegacyChip",
        "qubit_name": "Q2",
        "resonator_name": "unknown",
    }
    assert raw["fluxdep_fit"] == {"params": {}}


def test_ensure_scope_rejects_identity_mismatch(tmp_path) -> None:
    params_path = tmp_path / "result" / "ChipA" / "Q1" / "params.json"
    params_path.parent.mkdir(parents=True)
    params_path.write_text(
        json.dumps({"project": {"chip_name": "Other", "qubit_name": "Q1"}}),
        encoding="utf8",
    )

    with pytest.raises(ResultScopeError, match="Other/Q1"):
        ResultScopeManager(tmp_path).ensure_scope(chip_name="ChipA", qub_name="Q1")
