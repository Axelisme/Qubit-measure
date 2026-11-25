"""Registry key naming contracts using synthetic definition modules."""

import json
import sys
from collections.abc import Callable
from types import ModuleType, SimpleNamespace

import pytest
from check_registry_key_paths import (
    KeyDefinitionModule,
    KeyPathViolation,
    check_registered_key_paths,
    check_registry_key_paths,
    main,
)
from zcu_tools.gui.app.measure.registry import Registry

from tests.gui.app.measure._synthetic import SampleAdapter


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("zcu_lab.v2.family.echo.gui", ()),
        (
            "zcu_lab.v2.family.ramsey.gui",
            (KeyPathViolation("family/echo", "family/ramsey", "family/ramsey"),),
        ),
    ],
    ids=["declared-directory", "wrong-directory"],
)
def test_registered_key_paths_follow_definition_source_not_run_module(
    source: str, expected: tuple[KeyPathViolation, ...]
) -> None:
    registry = Registry()

    def register(
        target: Registry, record_source: Callable[[str, ModuleType], None]
    ) -> None:
        target.register("family/echo", SampleAdapter)
        record_source("family/echo", ModuleType(source))

    # The synthetic adapter's Run is outside the declared Lab directory.
    result = check_registered_key_paths(registry, register)
    assert result.paths_compared
    assert result.source_issues == ()
    assert result.path_mismatches == expected
    assert registry.list_names() == ["family/echo"]


@pytest.mark.parametrize(
    ("sources", "expected"),
    [
        (
            (),
            KeyPathViolation(
                "family/echo", "0 definition sources", "exactly 1 definition source"
            ),
        ),
        (
            (KeyDefinitionModule("family/echo", "zcu_lab.v2.family.echo.gui"),) * 2,
            KeyPathViolation(
                "family/echo", "2 definition sources", "exactly 1 definition source"
            ),
        ),
        (
            (
                KeyDefinitionModule("family/echo", "zcu_lab.v2.family.echo.gui"),
                KeyDefinitionModule("family/ramsey", "zcu_lab.v2.family.ramsey.gui"),
            ),
            KeyPathViolation(
                "family/ramsey", "source for an unregistered key", "a registered key"
            ),
        ),
        (
            (KeyDefinitionModule("family/echo", "zcu_lab.v2.family.ec-ho.gui"),),
            KeyPathViolation(
                "family/echo",
                "zcu_lab.v2.family.ec-ho.gui",
                "zcu_lab.v2.<directory>.<module>",
            ),
        ),
    ],
    ids=["missing", "duplicate", "extra", "invalid-module"],
)
def test_registered_key_paths_skip_comparison_without_one_valid_source_per_key(
    sources: tuple[KeyDefinitionModule, ...], expected: KeyPathViolation
) -> None:
    def register(
        target: Registry, record_source: Callable[[str, ModuleType], None]
    ) -> None:
        target.register("family/echo", SampleAdapter)
        target.register("family/other", SampleAdapter)
        record_source("family/other", ModuleType("zcu_lab.v2.family.echo.gui"))
        for source in sources:
            record_source(source.key, ModuleType(source.module))

    result = check_registered_key_paths(Registry(), register)
    assert not result.paths_compared
    assert result.source_issues == (expected,)
    assert result.path_mismatches == ()


def _install_synthetic_assembly(
    monkeypatch: pytest.MonkeyPatch,
    register: Callable[[Registry, Callable[[str, ModuleType], None]], None],
) -> None:
    # The CLI has no assembly injection; replace its import without loading Lab.
    assembly = ModuleType("zcu_lab.assembly")
    monkeypatch.setattr(
        assembly,
        "lab_definitions",
        lambda: SimpleNamespace(templates={}, register_experiments=register),
        raising=False,
    )
    monkeypatch.setitem(sys.modules, assembly.__name__, assembly)


@pytest.mark.parametrize(
    ("source", "actual", "expected", "summary"),
    [
        (
            "zcu_lab.v2.family.ramsey.gui",
            "family/ramsey",
            "family/ramsey",
            "suggested key",
        ),
        (
            None,
            "0 definition sources",
            "exactly 1 definition source",
            "comparison skipped",
        ),
    ],
    ids=["path-mismatch", "missing-source"],
)
def test_main_reports_findings_without_failing_registration(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    source: str | None,
    actual: str,
    expected: str,
    summary: str,
) -> None:
    def register(
        target: Registry, record_source: Callable[[str, ModuleType], None]
    ) -> None:
        target.register("family/echo", SampleAdapter)
        if source is not None:
            record_source("family/echo", ModuleType(source))

    _install_synthetic_assembly(monkeypatch, register)

    assert main([]) == 0
    output = capsys.readouterr()
    assert json.loads(output.out) == {
        "checked_count": 1,
        "violation_count": 1,
        "violations": [{"key": "family/echo", "actual": actual, "expected": expected}],
    }
    assert output.err.count("family/echo") == 1
    assert actual in output.err and expected in output.err
    assert "warning:" in output.err and summary in output.err


def test_main_propagates_duplicate_registry_key_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def register(
        target: Registry, _record_source: Callable[[str, ModuleType], None]
    ) -> None:
        target.register("family/echo", SampleAdapter)
        target.register("family/echo", SampleAdapter)

    _install_synthetic_assembly(monkeypatch, register)

    with pytest.raises(ValueError, match="already registered"):
        main([])
    assert capsys.readouterr().out == ""


def test_single_entry_key_includes_the_entire_definition_directory() -> None:
    violations = check_registry_key_paths(
        [
            KeyDefinitionModule(
                "family/time_domain/echo", "zcu_lab.v2.family.time_domain.echo.gui"
            ),
            KeyDefinitionModule(
                "family/ramsey", "zcu_lab.v2.family.time_domain.ramsey.gui"
            ),
        ]
    )

    assert violations == (
        KeyPathViolation(
            "family/ramsey", "family/time_domain/ramsey", "family/time_domain/ramsey"
        ),
    )


def test_multiple_entries_require_one_variant_segment_in_the_definition_directory() -> (
    None
):
    violations = check_registry_key_paths(
        [
            KeyDefinitionModule(
                "family/sweep/gain", "zcu_lab.v2.family.sweep.gain_gui"
            ),
            KeyDefinitionModule(
                "family/sweep/freq", "zcu_lab.v2.family.sweep.freq_gui"
            ),
            KeyDefinitionModule("family/sweep", "zcu_lab.v2.family.sweep.gui"),
            KeyDefinitionModule("family/sweep/", "zcu_lab.v2.family.sweep.gui"),
            KeyDefinitionModule(
                "family/sweep/gain/fast", "zcu_lab.v2.family.sweep.gui"
            ),
            KeyDefinitionModule("elsewhere/sweep/fast", "zcu_lab.v2.family.sweep.gui"),
            KeyDefinitionModule("family/echo/slow", "zcu_lab.v2.family.echo.gui"),
        ]
    )

    assert violations == (
        KeyPathViolation("family/sweep", "family/sweep", "family/sweep/<variant>"),
        KeyPathViolation("family/sweep/", "family/sweep", "family/sweep/<variant>"),
        KeyPathViolation(
            "family/sweep/gain/fast", "family/sweep", "family/sweep/<variant>"
        ),
        KeyPathViolation(
            "elsewhere/sweep/fast", "family/sweep", "family/sweep/<variant>"
        ),
        KeyPathViolation("family/echo/slow", "family/echo", "family/echo"),
    )


def test_module_without_an_experiment_directory_does_not_pass_by_matching_its_key() -> (
    None
):
    violations = check_registry_key_paths(
        [
            KeyDefinitionModule("another/family/echo", "another.family.echo.gui"),
            KeyDefinitionModule("root_form", "zcu_lab.v2.root_form"),
        ]
    )

    assert violations == (
        KeyPathViolation(
            "another/family/echo",
            "another.family.echo.gui",
            "zcu_lab.v2.<directory>.<module>",
        ),
        KeyPathViolation(
            "root_form", "zcu_lab.v2.root_form", "zcu_lab.v2.<directory>.<module>"
        ),
    )
