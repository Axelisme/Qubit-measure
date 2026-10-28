"""Synthetic upgrades through the public pure runner and offline CLI seams."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from shlex import split

import pytest
from ruamel.yaml import YAML
from zcu_tools.format_version import YamlMap, YamlValue
from zcu_tools.resources.parameter_schema import ParameterSchemaPolicy
from zcu_tools.resources.parameter_upgrade import (
    ParameterDocument,
    UpgradePlan,
    UpgradeSource,
    UpgradeStep,
    main,
    upgrade_document,
    upgrade_files,
)


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    path = tmp_path / "project" / "result"
    path.mkdir(parents=True)
    return path


def test_schema_hint_previews_and_applies_source_or_caller_target() -> None:
    policy = ParameterSchemaPolicy(
        2, ("uv", "run", "--no-sync", "--", "python", "scripts/upgrade_parameters.py")
    )
    source = Path("/data/entry with spaces/setup.yaml")
    with pytest.raises(ValueError, match="schema_version missing") as error:
        policy.require_current({}, source=source)
    lines = str(error.value).splitlines()
    assert lines[0] == f"{source}: schema_version missing; current 2."
    assert lines[1].startswith("Preview: ")
    assert lines[2].startswith("Apply: ")
    assert split(lines[1].removeprefix("Preview: ")) == [
        *policy.upgrade_command,
        str(source),
    ]
    assert split(lines[2].removeprefix("Apply: ")) == [
        *policy.upgrade_command,
        str(source),
        "--apply",
    ]
    assert lines[2].endswith("--apply")

    with pytest.raises(ValueError, match="schema_version missing") as error:
        policy.require_current({}, source=source, upgrade_target=source.parent)
    lines = str(error.value).splitlines()
    assert split(lines[1].removeprefix("Preview: ")) == [
        *policy.upgrade_command,
        str(source.parent),
    ]
    assert split(lines[2].removeprefix("Apply: ")) == [
        *policy.upgrade_command,
        str(source.parent),
        "--apply",
    ]


def test_runner_orders_steps_and_preserves_caller_input() -> None:
    calls: list[int] = []

    def validate(document: ParameterDocument, version: int) -> None:
        if document.data.get("schema_version") != version:
            raise ValueError("version does not match shape")

    def step_one(document: ParameterDocument) -> ParameterDocument:
        calls.append(1)
        document.data["middle"] = document.data.pop("old")
        document.data["schema_version"] = 2
        return document

    def step_two(document: ParameterDocument) -> ParameterDocument:
        calls.append(2)
        document.data["new"] = document.data.pop("middle")
        document.data["schema_version"] = 3
        return document

    plan = UpgradePlan(
        ParameterSchemaPolicy(3, ("upgrade",)),
        (UpgradeStep(1, step_one), UpgradeStep(2, step_two)),
        validate,
        lambda _: False,
    )
    original = ParameterDocument({"schema_version": 1, "old": 7, "keep": [None, "x"]})
    before = deepcopy(original)
    result = upgrade_document(original, plan=plan)
    assert result.data == {"schema_version": 3, "new": 7, "keep": [None, "x"]}
    assert original == before
    assert calls == [1, 2]
    assert upgrade_document(result, plan=plan) == result
    assert calls == [1, 2]


def test_cli_reports_conversion_notes_outside_yaml_payload(
    data_dir: Path, rename_plan: UpgradePlan, capsys: pytest.CaptureFixture[str]
) -> None:
    def transform(document: ParameterDocument) -> ParameterDocument:
        converted = rename_plan.steps[0].transform(document)
        return replace(converted, notes=("new: rename the declared placeholder",))

    plan = replace(rename_plan, steps=(UpgradeStep(1, transform),))
    path = data_dir / "point.yaml"
    path.write_text("schema_version: 1\nold: 7\n", encoding="utf-8")
    assert main([str(path)], plan=plan) == 0
    assert "new: rename the declared placeholder" in capsys.readouterr().out
    assert upgrade_document(
        ParameterDocument({"schema_version": 1, "old": 7}), plan=plan
    ).data == {
        "schema_version": 2,
        "new": 7,
    }


def test_unversioned_document_requires_complete_caller_classification() -> None:
    def classify(document: ParameterDocument) -> bool:
        return document.data == {"old": 7}

    def validate(document: ParameterDocument, version: int) -> None:
        if document.data != {"old": 7, "schema_version": 1}:
            raise ValueError("incomplete v1 document")

    plan = UpgradePlan(ParameterSchemaPolicy(1, ("upgrade",)), (), validate, classify)
    original = ParameterDocument({"old": 7})
    assert upgrade_document(original, plan=plan).data == {"old": 7, "schema_version": 1}
    assert original.data == {"old": 7}
    corrupt_documents: tuple[YamlMap, ...] = (
        {"old": None},
        {"old": 7, "new": 7},
        {"schema_version": None},
    )
    for corrupt in corrupt_documents:
        with pytest.raises(ValueError, match="classification|unsupported"):
            upgrade_document(ParameterDocument(corrupt), plan=plan)


@pytest.mark.parametrize("sources", [(2, 1), (1, 1), (1, 3), (0, 1)])
def test_chain_declarations_must_be_ordered_consecutive_and_end_at_current(
    sources: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError, match="consecutive"):
        UpgradePlan(
            ParameterSchemaPolicy(3, ("upgrade",)),
            tuple(
                UpgradeStep(version, lambda document: document) for version in sources
            ),
            lambda _document, _version: None,
            lambda _: False,
        )


@pytest.mark.parametrize("next_version", [1, 3, 2.0, "2", None, True])
def test_step_must_return_exact_next_integer_version(next_version: YamlValue) -> None:
    def malformed_step(document: ParameterDocument) -> ParameterDocument:
        document.data["schema_version"] = next_version
        return document

    plan = UpgradePlan(
        ParameterSchemaPolicy(2, ("upgrade",)),
        (UpgradeStep(1, malformed_step),),
        lambda _document, _version: None,
        lambda _: False,
    )
    original = ParameterDocument({"schema_version": 1})
    with pytest.raises(ValueError, match="must produce"):
        upgrade_document(original, plan=plan)
    assert original.data == {"schema_version": 1}


@pytest.fixture
def rename_plan() -> UpgradePlan:
    def validate(document: ParameterDocument, version: int) -> None:
        expected = "old" if version == 1 else "new"
        forbidden = "new" if version == 1 else "old"
        if document.data.get(expected) != 7 or forbidden in document.data:
            raise ValueError(f"v{version}: require {expected}=7 and no {forbidden}")

    def transform(document: ParameterDocument) -> ParameterDocument:
        document.data["new"] = document.data.pop("old")
        document.data["schema_version"] = 2
        return document

    def classify(document: ParameterDocument) -> bool:
        validate(document, 1)
        return True

    return UpgradePlan(
        ParameterSchemaPolicy(2, ("upgrade",)),
        (UpgradeStep(1, transform),),
        validate,
        classify,
    )


def test_file_dry_run_reports_changes_and_apply_backs_up_before_idempotent_rerun(
    data_dir: Path, rename_plan: UpgradePlan
) -> None:
    path = data_dir / "point.yaml"
    original = b"schema_version: 1\nold: 7\nkeep: [null, x]\n"
    path.write_bytes(original)
    source = UpgradeSource(path, point="bias-a")
    (dry,) = upgrade_files((source,), plan=rename_plan)
    assert dry.source == source
    assert dry.status == "dry-run"
    assert {change.path for change in dry.changes} == {
        ("old",),
        ("new",),
        ("schema_version",),
    }
    assert path.read_bytes() == original
    assert list(data_dir.iterdir()) == [path]
    (applied,) = upgrade_files((source,), plan=rename_plan, apply=True)
    assert applied.status == "applied"
    assert not list(path.parent.glob("*.lock"))
    assert applied.backup_path is not None
    assert applied.backup_path.read_bytes() == original
    assert (
        applied.backup_path.parent
        == data_dir.parent.with_name("project-backup") / "result"
    )
    assert set(data_dir.iterdir()) == {path}
    assert YAML(typ="safe").load(path) == {
        "schema_version": 2,
        "new": 7,
        "keep": [None, "x"],
    }
    after = path.read_bytes()
    (skipped,) = upgrade_files((source,), plan=rename_plan, apply=True)
    assert skipped.status == "skipped"
    assert skipped.backup_path is None
    assert path.read_bytes() == after


def test_cli_isolates_corrupt_and_partial_documents_and_defaults_to_dry_run(
    data_dir: Path, rename_plan: UpgradePlan, capsys: pytest.CaptureFixture[str]
) -> None:
    good = data_dir / "bias-a.yaml"
    partial = data_dir / "bias-b.yaml"
    corrupt = data_dir / "bias-c.yaml"
    malformed = data_dir / "syntax.yaml"
    current_bad = data_dir / "current-bad.yaml"
    inputs = {
        good: b"old: 7\nkeep: untouched\n",
        partial: b"schema_version: 1\nold: 7\nnew: 7\n",
        corrupt: b"schema_version: 1\nold: null\n",
        malformed: b"broken: [\n",
        current_bad: b"schema_version: 2\nold: 7\n",
    }
    for path, content in inputs.items():
        path.write_bytes(content)
    argv = [str(path) for path in inputs]
    assert main(argv, plan=rename_plan) == 1
    output = capsys.readouterr().out
    assert "dry-run" in output
    assert "old" in output and "new" in output
    assert all(str(path) in output for path in inputs)
    assert "v1: require old=7 and no new" in output
    assert "v2: require new=7 and no old" in output
    assert set(data_dir.iterdir()) == set(inputs)
    assert all(path.read_bytes() == content for path, content in inputs.items())
    assert main(["--apply", *argv], plan=rename_plan) == 1
    output = capsys.readouterr().out
    assert "backup:" in output
    assert "applied" in output
    assert YAML(typ="safe").load(good) == {
        "schema_version": 2,
        "new": 7,
        "keep": "untouched",
    }
    assert all(
        path.read_bytes() == content for path, content in inputs.items() if path != good
    )
    assert main([str(good), "--apply"], plan=rename_plan) == 0
    assert "skipped" in capsys.readouterr().out


def test_component_cli_expands_entry_for_preview_and_apply(
    data_dir: Path, rename_plan: UpgradePlan, capsys: pytest.CaptureFixture[str]
) -> None:
    root = data_dir / "entry with spaces"
    sources = (
        root / "setup.yaml",
        root / "points" / "bias-a" / "point.yaml",
        root / "points" / "bias-b" / "point.yaml",
    )
    original = b"old: 7\nkeep: untouched\n"
    for path in sources:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(original)
    unrelated = root / "points" / "bias-a" / "module_cfg.yaml"
    unrelated.write_bytes(b"library: untouched\n")
    before = set(root.rglob("*"))

    assert main([str(root)], plan=rename_plan) == 0
    preview = capsys.readouterr().out
    assert [line for line in preview.splitlines() if line.endswith(": dry-run")] == [
        f"{path}: dry-run" for path in sources
    ]
    assert set(root.rglob("*")) == before
    assert all(path.read_bytes() == original for path in sources)

    assert main([str(root), "--apply"], plan=rename_plan) == 0
    output = capsys.readouterr().out
    for path in sources:
        assert f"{path}: applied" in output
        assert YAML(typ="safe").load(path) == {
            "schema_version": 2,
            "new": 7,
            "keep": "untouched",
        }
        backups = list(
            (
                data_dir.parent.with_name("project-backup")
                / path.parent.relative_to(data_dir.parent)
            ).glob(f"{path.name}.*.bak")
        )
        assert len(backups) == 1
        assert backups[0].read_bytes() == original
        assert str(backups[0]) in output
    assert unrelated.read_bytes() == b"library: untouched\n"


def test_component_cli_rejects_non_entry_directory(
    data_dir: Path, rename_plan: UpgradePlan, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as error:
        main([str(data_dir)], plan=rename_plan)
    assert error.value.code == 2
    diagnostic = capsys.readouterr().err
    assert str(data_dir) in diagnostic
    assert "not an entry directory" in diagnostic
    assert "setup.yaml" in diagnostic
    assert list(data_dir.iterdir()) == []


def test_component_cli_prints_caller_assumptions_before_file_results(
    data_dir: Path, rename_plan: UpgradePlan, capsys: pytest.CaptureFixture[str]
) -> None:
    path = data_dir / "point.yaml"
    path.write_text("old: 7\n", encoding="utf-8")
    plan = replace(rename_plan, notes=("Reference points are assumed; recalibrate.",))
    assert main([str(path)], plan=plan) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "Reference points are assumed; recalibrate."
    assert f"{path}: dry-run" in lines


def test_file_runner_carries_source_context_across_fresh_step_outputs(
    data_dir: Path,
) -> None:
    source = data_dir / "setup.yaml"
    source.write_text("schema_version: 1\n", encoding="utf-8")
    seen: list[tuple[Path | None, int]] = []

    def validate(document: ParameterDocument, version: int) -> None:
        seen.append((document.source_path, version))

    plan = UpgradePlan(
        ParameterSchemaPolicy(2, ("upgrade",)),
        (UpgradeStep(1, lambda _: ParameterDocument({"schema_version": 2})),),
        validate,
        lambda _: False,
    )
    (result,) = upgrade_files((UpgradeSource(source),), plan=plan)
    assert result.status == "dry-run"
    assert seen == [(source, 1), (source, 2)]


@pytest.mark.parametrize("fail_replace", [False, True])
def test_apply_finishes_backup_before_replace_and_retains_it_on_failure(
    data_dir: Path,
    rename_plan: UpgradePlan,
    monkeypatch: pytest.MonkeyPatch,
    fail_replace: bool,
) -> None:
    source = data_dir / "point.yaml"
    original = b"schema_version: 1\nold: 7\n"
    source.write_bytes(original)
    replace = Path.replace
    backups: list[Path] = []

    def observed_replace(temporary: Path, destination: Path) -> Path:
        backups.extend(
            (data_dir.parent.with_name("project-backup") / "result").glob(
                "point.yaml.*.bak"
            )
        )
        assert len(backups) == 1
        assert backups[0].read_bytes() == original
        assert destination.read_bytes() == original
        if fail_replace:
            raise OSError("publication unavailable")
        return replace(temporary, destination)

    monkeypatch.setattr(Path, "replace", observed_replace)
    (result,) = upgrade_files((UpgradeSource(source),), plan=rename_plan, apply=True)
    assert result.backup_path == backups[0]
    if fail_replace:
        assert result.status == "rejected"
        assert result.error == "publication unavailable"
        assert source.read_bytes() == original
    else:
        assert result.status == "applied"
    assert set(data_dir.iterdir()) == {source}


def test_cli_backup_override_mirrors_project_path(
    data_dir: Path, rename_plan: UpgradePlan
) -> None:
    source = data_dir / "point.yaml"
    original = b"old: 7\n"
    source.write_bytes(original)
    backup_root = data_dir.parent / "archive"
    assert (
        main(
            [str(source), "--apply", "--backup-root", str(backup_root)],
            plan=rename_plan,
        )
        == 0
    )
    (backup,) = (backup_root / "result").glob("point.yaml.*.bak")
    assert backup.read_bytes() == original
    assert list(data_dir.iterdir()) == [source]


def test_standalone_apply_requires_override_and_can_restore_original(
    tmp_path: Path, rename_plan: UpgradePlan
) -> None:
    source_dir = tmp_path / "standalone"
    source_dir.mkdir()
    source = source_dir / "point.yaml"
    original = b"old: 7\n"
    source.write_bytes(original)
    sources = (UpgradeSource(source),)
    (preview,) = upgrade_files(sources, plan=rename_plan)
    assert preview.status == "dry-run"
    (rejected,) = upgrade_files(sources, plan=rename_plan, apply=True)
    assert rejected.status == "rejected"
    assert "--backup-root" in (rejected.error or "")
    assert set(tmp_path.rglob("*")) == {source_dir, source}
    assert source.read_bytes() == original
    backup_root = tmp_path / "archive"
    (applied,) = upgrade_files(
        sources, plan=rename_plan, apply=True, backup_root=backup_root
    )
    assert applied.status == "applied"
    assert applied.backup_path is not None
    assert applied.backup_path.parent == backup_root
    source.write_bytes(applied.backup_path.read_bytes())
    assert source.read_bytes() == original


@pytest.mark.parametrize(
    "unsafe", ["result", "Database", "symlink", "mirror-symlink", "standalone"]
)
def test_unsafe_backup_destination_rejects_before_writes(
    tmp_path: Path, rename_plan: UpgradePlan, unsafe: str
) -> None:
    data_dir = tmp_path / "project" / "result"
    data_dir.mkdir(parents=True)
    source = data_dir / "point.yaml"
    backup_root = tmp_path / "archive"
    if unsafe == "standalone":
        source = tmp_path / "point.yaml"
        backup_root = tmp_path / "nested"
    elif unsafe == "symlink":
        backup_root.symlink_to(data_dir, target_is_directory=True)
    elif unsafe == "mirror-symlink":
        backup_root.mkdir()
        (backup_root / "result").symlink_to(data_dir, target_is_directory=True)
    else:
        backup_root = tmp_path / "other" / unsafe / "backups"
    source.write_bytes(b"old: 7\n")
    before = set(tmp_path.rglob("*"))
    (result,) = upgrade_files(
        (UpgradeSource(source),), plan=rename_plan, apply=True, backup_root=backup_root
    )
    assert result.status == "rejected"
    assert "backup" in (result.error or "")
    assert result.backup_path is None
    assert source.read_bytes() == b"old: 7\n"
    assert set(tmp_path.rglob("*")) == before


@pytest.mark.parametrize("data_tree", ["result", "Database"])
@pytest.mark.parametrize("symlink_override", [False, True])
def test_project_root_backup_override_rejects_mirror_into_source_tree_before_writes(
    tmp_path: Path,
    rename_plan: UpgradePlan,
    data_tree: str,
    symlink_override: bool,
) -> None:
    project = tmp_path / "project"
    source = project / data_tree / "entry" / "point.yaml"
    source.parent.mkdir(parents=True)
    original = b"old: 7\n"
    source.write_bytes(original)
    backup_root = project
    if symlink_override:
        backup_root = tmp_path / "backup-link"
        backup_root.symlink_to(project, target_is_directory=True)
    before = set(tmp_path.rglob("*"))

    (result,) = upgrade_files(
        (UpgradeSource(source),), plan=rename_plan, apply=True, backup_root=backup_root
    )

    assert result.status == "rejected"
    assert "backup" in (result.error or "")
    assert result.backup_path is None
    assert source.read_bytes() == original
    assert set(tmp_path.rglob("*")) == before


@pytest.fixture
def legacy_source(data_dir: Path) -> Path:
    source = data_dir / "point.yaml"
    source.write_bytes(b"old: 7\n")
    return source


@pytest.mark.parametrize("data_tree", ["result", "Database"])
@pytest.mark.parametrize("symlink_override", [False, True])
def test_other_project_root_override_rejects_before_writes(
    tmp_path: Path,
    legacy_source: Path,
    rename_plan: UpgradePlan,
    data_tree: str,
    symlink_override: bool,
) -> None:
    other_project = tmp_path / "other-project"
    other_data = other_project / data_tree
    other_data.mkdir(parents=True)
    existing = other_data / "existing.dat"
    existing.write_bytes(b"other project data\n")
    backup_root = other_project
    if symlink_override:
        backup_root = tmp_path / "archive-backup"
        backup_root.symlink_to(other_project, target_is_directory=True)
    before = set(tmp_path.rglob("*"))

    (result,) = upgrade_files(
        (UpgradeSource(legacy_source),),
        plan=rename_plan,
        apply=True,
        backup_root=backup_root,
    )

    assert result.status == "rejected"
    assert "*-backup" in (result.error or "")
    assert "empty directory" in (result.error or "")
    assert str(other_project) in (result.error or "")
    assert result.backup_path is None
    assert legacy_source.read_bytes() == b"old: 7\n"
    assert existing.read_bytes() == b"other project data\n"
    assert set(tmp_path.rglob("*")) == before


@pytest.mark.parametrize("root_choice", ["default", "override", "symlink"])
def test_named_backup_root_resumes_existing_mirrors(
    tmp_path: Path,
    legacy_source: Path,
    rename_plan: UpgradePlan,
    root_choice: str,
) -> None:
    root = tmp_path / (
        "project-backup" if root_choice == "default" else "archive-backup"
    )
    existing_files = (
        root / "result" / "migration-original.bak",
        root / "Database" / "migration-original.bak",
    )
    for path in existing_files:
        path.parent.mkdir(parents=True)
        path.write_bytes(b"migration original\n")
    override = None if root_choice == "default" else root
    if root_choice == "symlink":
        override = tmp_path / "archive-link"
        override.symlink_to(root, target_is_directory=True)
    backups: list[Path] = []

    for _ in range(2):
        legacy_source.write_bytes(b"old: 7\n")
        (result,) = upgrade_files(
            (UpgradeSource(legacy_source),),
            plan=rename_plan,
            apply=True,
            backup_root=override,
        )
        assert result.status == "applied"
        assert result.backup_path is not None
        assert result.backup_path.parent == root / "result"
        assert result.backup_path.read_bytes() == b"old: 7\n"
        assert YAML(typ="safe").load(legacy_source) == {"schema_version": 2, "new": 7}
        backups.append(result.backup_path)

    assert len(set(backups)) == 2
    assert set((root / "result").glob("point.yaml.*.bak")) == set(backups)
    assert all(path.read_bytes() == b"old: 7\n" for path in backups)
    assert all(path.read_bytes() == b"migration original\n" for path in existing_files)


@pytest.mark.parametrize("initial_root", ["missing", "empty", "unrelated-directory"])
def test_unnamed_backup_root_allows_first_mirror_but_rejects_continuation(
    tmp_path: Path,
    legacy_source: Path,
    rename_plan: UpgradePlan,
    initial_root: str,
) -> None:
    root = legacy_source.parent.parent / "archive"
    if initial_root != "missing":
        root.mkdir()
    if initial_root == "unrelated-directory":
        (root / "notes").mkdir()
    (first,) = upgrade_files(
        (UpgradeSource(legacy_source),),
        plan=rename_plan,
        apply=True,
        backup_root=root,
    )
    assert first.status == "applied"
    assert first.backup_path is not None
    assert first.backup_path.parent == root / "result"
    assert first.backup_path.read_bytes() == b"old: 7\n"
    assert YAML(typ="safe").load(legacy_source) == {"schema_version": 2, "new": 7}
    legacy_source.write_bytes(b"old: 7\n")
    before = set(tmp_path.rglob("*"))

    (second,) = upgrade_files(
        (UpgradeSource(legacy_source),),
        plan=rename_plan,
        apply=True,
        backup_root=root,
    )

    assert second.status == "rejected"
    assert "*-backup" in (second.error or "")
    assert second.backup_path is None
    assert legacy_source.read_bytes() == b"old: 7\n"
    assert first.backup_path.read_bytes() == b"old: 7\n"
    assert set(tmp_path.rglob("*")) == before


@pytest.mark.parametrize("destination_tree", ["result", "Database"])
@pytest.mark.parametrize("redirect_entry", [False, True])
def test_mirror_symlink_into_another_project_rejects_before_writes(
    tmp_path: Path,
    rename_plan: UpgradePlan,
    destination_tree: str,
    redirect_entry: bool,
) -> None:
    source = tmp_path / "project-a" / "result" / "entry" / "point.yaml"
    source.parent.mkdir(parents=True)
    original = b"old: 7\n"
    source.write_bytes(original)
    backup_root = tmp_path / "archive-backup"
    other_tree = backup_root / "project-b" / destination_tree
    other_tree.mkdir(parents=True)
    mirror = backup_root / "result"
    if redirect_entry:
        mirror.mkdir()
        mirror = mirror / "entry"
    mirror.symlink_to(other_tree, target_is_directory=True)
    before = set(tmp_path.rglob("*"))

    (result,) = upgrade_files(
        (UpgradeSource(source),), plan=rename_plan, apply=True, backup_root=backup_root
    )

    assert result.status == "rejected"
    assert "backup" in (result.error or "")
    assert result.backup_path is None
    assert source.read_bytes() == original
    assert set(tmp_path.rglob("*")) == before


def test_database_source_uses_project_sibling_backup(
    tmp_path: Path, rename_plan: UpgradePlan
) -> None:
    source = tmp_path / "project" / "Database" / "entry" / "point.yaml"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"old: 7\n")
    (result,) = upgrade_files((UpgradeSource(source),), plan=rename_plan, apply=True)
    assert result.status == "applied"
    assert result.backup_path is not None
    assert (
        result.backup_path.parent == tmp_path / "project-backup" / "Database" / "entry"
    )
    assert result.backup_path.read_bytes() == b"old: 7\n"


def test_no_policy_default_or_missing_upgrade_step_is_inferred() -> None:
    from zcu_tools.resources.entry.registry import ComponentRegistry

    registry = ComponentRegistry()
    with pytest.raises(RuntimeError, match="not configured"):
        registry.require_schema_policy()
    plan = UpgradePlan(
        ParameterSchemaPolicy(3, ("upgrade",)),
        (),
        lambda _document, _version: None,
        lambda _: False,
    )
    with pytest.raises(ValueError, match="missing upgrade step v1"):
        upgrade_document(ParameterDocument({"schema_version": 1}), plan=plan)
    with pytest.raises(ValueError, match="unsupported schema_version 4"):
        upgrade_document(ParameterDocument({"schema_version": 4}), plan=plan)
