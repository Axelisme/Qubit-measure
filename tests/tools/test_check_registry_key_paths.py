"""Registry key naming contracts using synthetic definition modules."""

from check_registry_key_paths import KeyPathViolation, check_registry_key_paths


def test_single_entry_key_includes_the_entire_definition_directory() -> None:
    violations = check_registry_key_paths(
        [
            ("family/time_domain/echo", "zcu_lab.v2.family.time_domain.echo.gui"),
            ("family/ramsey", "zcu_lab.v2.family.time_domain.ramsey.gui"),
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
            ("family/sweep/gain", "zcu_lab.v2.family.sweep.gain_gui"),
            ("family/sweep/freq", "zcu_lab.v2.family.sweep.freq_gui"),
            ("family/sweep", "zcu_lab.v2.family.sweep.gui"),
            ("family/sweep/", "zcu_lab.v2.family.sweep.gui"),
            ("family/sweep/gain/fast", "zcu_lab.v2.family.sweep.gui"),
            ("elsewhere/sweep/fast", "zcu_lab.v2.family.sweep.gui"),
            ("family/echo/slow", "zcu_lab.v2.family.echo.gui"),
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
            ("another/family/echo", "another.family.echo.gui"),
            ("root_form", "zcu_lab.v2.root_form"),
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
