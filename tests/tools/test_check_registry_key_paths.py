"""Registry key naming contracts using synthetic definition modules."""

from collections.abc import Callable
from types import ModuleType

import pytest
from check_registry_key_paths import (
    KeyDefinitionModule,
    KeyPathViolation,
    check_registered_key_paths,
    check_registry_key_paths,
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
    assert check_registered_key_paths(registry, register) == expected
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
def test_registered_key_paths_require_one_valid_source_per_registered_key(
    sources: tuple[KeyDefinitionModule, ...], expected: KeyPathViolation
) -> None:
    def register(
        target: Registry, record_source: Callable[[str, ModuleType], None]
    ) -> None:
        target.register("family/echo", SampleAdapter)
        for source in sources:
            record_source(source.key, ModuleType(source.module))

    assert check_registered_key_paths(Registry(), register) == (expected,)


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
