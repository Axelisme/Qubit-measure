"""Offline CLI rerun/diagnostic contracts with synthetic composition only."""

import json
from collections.abc import Generator
from pathlib import Path

import pytest
from pydantic import ConfigDict, Field
from zcu_tools.resources.document_store import YamlMap
from zcu_tools.resources.entry import ComponentSchema, component_registry

from migrations.migrate_storage import main
from tests.resources.entry.fakes import registry_state


class CliComponent(ComponentSchema):
    """Synthetic storage-only component accepting the fixed mapping's seeds."""

    model_config = ConfigDict(extra="allow")
    ext: YamlMap = Field(default_factory=dict)


@pytest.fixture(scope="module", autouse=True)
def cli_kinds() -> Generator[None]:
    before = registry_state()
    kinds = ("qubit/fluxonium", "qubit/transmon", "resonator", "amplifier/jpa")
    for kind in kinds:
        component_registry.register(kind, CliComponent)
    expected = registry_state()
    try:
        yield
        assert registry_state() == expected, "CLI module polluted registries"
    finally:
        for kind in reversed(kinds):
            component_registry.unregister(kind)
        assert registry_state() == before


@pytest.fixture(autouse=True)
def registry_guard(request: pytest.FixtureRequest, cli_kinds: None) -> Generator[None]:
    before = registry_state()
    yield
    assert registry_state() == before, f"registry polluter: {request.node.nodeid}"


@pytest.fixture
def arguments(tmp_path: Path) -> list[str]:
    repo = tmp_path / "repo"
    for root in ("result", "Database"):
        (repo / root / "chip/qubit").mkdir(parents=True)
    (repo / "result/chip/qubit/image").mkdir()
    (repo / "result/chip/qubit/image/fit.png").write_bytes(b"synthetic figure")
    return ["--repo-root", str(repo), "--backup-parent", str(tmp_path),
            "--result-root", str(repo / "result"), "--database-root", str(repo / "Database"),
            "--entry-result-root", str(repo / "result"), "--source-chip", "chip",
            "--source-qubit", "qubit", "--name", "converted", "--part", "all",
            "--qubit-kind", "qubit/fluxonium"]


def tree(root: Path) -> dict[Path, tuple[bytes, int]]:
    return {path: (path.read_bytes(), path.stat().st_mtime_ns)
            for path in root.rglob("*") if path.is_file()}


def call(arguments: list[str]) -> int:
    return main(arguments, register_components=lambda _registry: None, experiments=())


@pytest.mark.parametrize("resume", [False, True])
@pytest.mark.parametrize("dry_run", [False, True])
def test_cli_identical_rerun_prints_skipped_and_keeps_all_files(
    arguments: list[str], tmp_path: Path, capsys: pytest.CaptureFixture[str],
    resume: bool, dry_run: bool
) -> None:
    assert call(arguments) == 0
    first = json.loads(capsys.readouterr().out)
    before = tree(tmp_path)
    again = arguments + (["--resume"] if resume else []) + (["--dry-run"] if dry_run else [])
    assert call(again) == 0, "Owned CLI rerun must automatically verify and skip"
    report = json.loads(capsys.readouterr().out)
    assert report["entry_id"] == first["entry_id"]
    assert any(row["destination"].endswith("/setup.yaml") for row in report["skipped"])
    assert any(row["destination"].endswith("/figures/image/fit.png") for row in report["skipped"])
    assert tree(tmp_path) == before


@pytest.mark.parametrize("conflict", ["legal-edit", "missing-baseline"])
@pytest.mark.parametrize("dry_run", [False, True])
def test_cli_conflict_exits_two_with_located_diagnostic_without_writing(
    arguments: list[str], tmp_path: Path, capsys: pytest.CaptureFixture[str],
    conflict: str, dry_run: bool
) -> None:
    assert call(arguments) == 0
    capsys.readouterr()
    root = tmp_path / "repo/result/converted"
    target = root / ("setup.yaml" if conflict == "legal-edit" else "records/migration-state.json")
    if conflict == "legal-edit":
        from zcu_tools.resources.entry import ResultEntry
        entry = ResultEntry.open("converted", result_root=root.parent, database_root=tmp_path / "repo/Database")
        entry.setup.description = "Legal user edit"
    else:
        raw = json.loads(target.read_text())
        raw.pop("documents", None)
        target.write_text(json.dumps(raw))
    before = tree(tmp_path)
    assert call(arguments + ["--resume"] + (["--dry-run"] if dry_run else [])) == 2
    output = capsys.readouterr()
    assert str(target) in output.err
    assert "document-baseline" in output.err if conflict == "missing-baseline" else "setup" in output.err
    assert tree(tmp_path) == before
