from __future__ import annotations

from pathlib import Path

import pytest
from check_ratchet import suppression_regressions
from suppression_comparison import compare_ignores


@pytest.mark.parametrize("blanket", ["type: ignore", "type: ignore[attr-defined]"])
def test_blanket_to_explicit_diagnostics_keeps_the_same_escape_site(
    blanket: str,
) -> None:
    before = f"import missing  # {blanket}\n"
    after = "import missing  # pyright: ignore[reportMissingImports]\n"

    assert compare_ignores(before, after) == ()


def test_comment_deletion_and_import_formatting_preserve_statement_identity() -> None:
    before = (
        "# old explanatory text\n"
        "try:\n"
        "    from missing import Component  # type: ignore\n"
        "except ImportError:\n"
        "    Component = None\n"
    )
    after = (
        "try:\n"
        "    from missing import (  # pyright: ignore[reportMissingImports]\n"
        "        Component,\n"
        "    )\n"
        "except ImportError:\n"
        "    Component = None\n"
    )

    assert compare_ignores(before, after) == ()


def test_inserting_an_unrelated_statement_does_not_move_the_logical_site() -> None:
    before = "import missing  # type: ignore\n"
    after = "value = 1\nimport missing  # pyright: ignore[reportMissingImports]\n"

    assert compare_ignores(before, after) == ()


@pytest.mark.parametrize(
    "kind", ["type: ignore", "pyright: ignore[reportMissingImports]"]
)
def test_deletion_elsewhere_does_not_pay_for_a_new_position(kind: str) -> None:
    before = f"import first  # {kind}\nimport second\n"
    after = f"import first\nimport second  # {kind}\n"

    found = compare_ignores(before, after)

    assert len(found) == 1
    assert found[0].line == 2
    assert found[0].reason == "new-position"


def test_another_family_cannot_pay_for_a_new_pyright_position() -> None:
    before = "import first  # type: ignore\nimport second\n"
    after = "import first\nimport second  # pyright: ignore[reportMissingImports]\n"

    found = compare_ignores(before, after)

    assert [(item.kind, item.reason) for item in found] == [
        ("pyright-ignore", "new-position")
    ]


@pytest.mark.parametrize(
    "after_directive",
    [
        "pyright: ignore[reportMissingImports, reportArgumentType]",
        "pyright: ignore[reportArgumentType]",
        "pyright: ignore",
        "type: ignore",
    ],
)
def test_a_narrow_scope_cannot_grow_or_be_replaced_with_blanket(
    after_directive: str,
) -> None:
    before = "import missing  # pyright: ignore[reportMissingImports]\n"
    after = f"import missing  # {after_directive}\n"

    found = compare_ignores(before, after)

    assert len(found) == 1
    assert found[0].reason in {"scope-expanded", "not-narrow"}


@pytest.mark.parametrize(
    "codes",
    [
        "reportMissingImports",
        "reportArgumentType",
        "reportArgumentType, reportMissingImports",
    ],
)
def test_preserving_or_reducing_the_existing_diagnostic_set_is_allowed(
    codes: str,
) -> None:
    before = (
        "import missing  # pyright: ignore[reportMissingImports, reportArgumentType]\n"
    )
    after = f"import missing  # pyright: ignore[{codes}]\n"

    assert compare_ignores(before, after) == ()


@pytest.mark.parametrize(
    "directive",
    [
        "pyright: ignore",
        "pyright: ignore[]",
        "pyright: ignore[reportMissingImports",
        "pyright: ignore[notAReport]",
    ],
)
def test_a_non_narrow_migration_is_not_credited(directive: str) -> None:
    before = "import missing  # type: ignore\n"
    after = f"import missing  # {directive}\n"

    found = compare_ignores(before, after)

    assert len(found) == 1
    assert found[0].reason == "not-narrow"


@pytest.mark.parametrize(
    ("before", "after"),
    [
        (
            "def first():\n    import missing  # type: ignore\n"
            "def second():\n    import missing\n",
            "def first():\n    import missing\n"
            "def second():\n    import missing  # pyright: ignore[reportMissingImports]\n",
        ),
        (
            "if enabled:\n    import missing  # type: ignore\n"
            "else:\n    import missing\n",
            "if enabled:\n    import missing\n"
            "else:\n    import missing  # pyright: ignore[reportMissingImports]\n",
        ),
    ],
)
def test_identical_statements_in_different_owners_cannot_trade_ignores(
    before: str, after: str
) -> None:
    found = compare_ignores(before, after)

    assert len(found) == 1
    assert found[0].reason == "new-position"


@pytest.mark.parametrize(
    "directive",
    ["type: ignoreNotADirective", "pyright: ignore[", "pyright: ignore[notAReport]"],
)
def test_an_unproven_previous_directive_cannot_authorize_a_new_ignore(
    directive: str,
) -> None:
    before = f"import missing  # {directive}\n"
    after = "import missing  # pyright: ignore[reportMissingImports]\n"

    found = compare_ignores(before, after)

    assert len(found) == 1
    assert found[0].reason == "unproven-position"


def test_repeated_statements_in_the_same_owner_are_not_guessed_as_migrations() -> None:
    before = "import missing  # type: ignore\nimport missing\n"
    after = "import missing  # pyright: ignore[reportMissingImports]\nimport missing\n"

    found = compare_ignores(before, after)

    assert len(found) == 1
    assert found[0].reason == "unproven-position"


@pytest.mark.parametrize(
    ("before", "after", "line", "reason"),
    [
        (
            "if flag:\n    import missing  # type: ignore\n"
            "flag = not flag\nif flag:\n    pass\n",
            "if flag:\n    pass\nflag = not flag\n"
            "if flag:\n    import missing  # {directive}\n",
            5,
            "unproven-position",
        ),
        (
            "try:\n    import missing  # type: ignore\n"
            "except ImportError:\n    pass\n"
            "try:\n    pass\nexcept ImportError:\n    pass\n",
            "try:\n    pass\nexcept ImportError:\n    pass\n"
            "try:\n    import missing  # {directive}\n"
            "except ImportError:\n    pass\n",
            6,
            "unproven-position",
        ),
        (
            "try:\n    run()\nexcept ValueError:\n"
            "    import missing  # type: ignore\nexcept ValueError:\n    pass\n",
            "try:\n    run()\nexcept ValueError:\n    pass\n"
            "except ValueError:\n    import missing  # {directive}\n",
            6,
            "unproven-position",
        ),
        (
            "if flag:\n    import missing  # type: ignore\nif flag:\n    pass\n",
            "if flag:\n    import missing  # {directive}\n",
            2,
            "new-position",
        ),
        (
            "if flag:\n    import missing  # type: ignore\n",
            "if flag:\n    pass\nif flag:\n    import missing  # {directive}\n",
            4,
            "unproven-position",
        ),
        (
            "if flag:\n    if nested:\n        import missing  # type: ignore\n"
            "if flag:\n    pass\n",
            "if flag:\n    pass\nif flag:\n    if nested:\n"
            "        import missing  # {directive}\n",
            5,
            "unproven-position",
        ),
    ],
    ids=["if-owners", "try-owners", "handlers", "before-only", "after-only", "deep"],
)
@pytest.mark.parametrize(
    "directive", ["pyright: ignore[reportMissingImports]", "type: ignore"]
)
def test_ambiguous_ancestry_cannot_authorize_an_escape_site(
    before: str, after: str, line: int, reason: str, directive: str
) -> None:
    found = compare_ignores(before, after.format(directive=directive))

    assert len(found) == 1
    assert found[0].line == line
    assert found[0].reason == reason


def test_exact_unchanged_input_preserves_ambiguous_existing_debt() -> None:
    source = "import missing  # type: ignore\nimport missing\n"

    assert compare_ignores(source, source) == ()


def test_syntax_failure_cannot_establish_a_previous_escape_site() -> None:
    before = "import missing  # type: ignore\nunfinished = (\n"
    after = "import missing  # pyright: ignore[reportMissingImports]\n"

    found = compare_ignores(before, after)

    assert len(found) == 1
    assert found[0].reason == "unproven-position"


def test_a_malformed_candidate_does_not_hide_an_observed_marker() -> None:
    before = "import missing  # type: ignore\n"
    after = "import missing  # pyright: ignore[reportMissingImports]\nunfinished = (\n"

    found = compare_ignores(before, after)

    assert len(found) == 1
    assert found[0].reason == "unproven-position"


def test_new_source_has_no_inherited_escape_positions() -> None:
    found = compare_ignores(
        "", "import missing  # pyright: ignore[reportMissingImports]\n"
    )

    assert [(item.kind, item.reason) for item in found] == [
        ("pyright-ignore", "new-position")
    ]


def test_strings_and_comment_prose_do_not_create_escape_sites() -> None:
    source = (
        'example = "# pyright: ignore[reportMissingImports]"\n'
        "# see also # type: ignore\n"
    )

    assert compare_ignores("", source) == ()


def test_removing_an_escape_site_is_allowed() -> None:
    before = "import missing  # pyright: ignore[reportMissingImports]\n"

    assert compare_ignores(before, "import missing\n") == ()


@pytest.fixture
def source_roots(tmp_path: Path) -> tuple[Path, Path]:
    roots = (tmp_path / "base", tmp_path / "candidate")
    for root in roots:
        (root / "lib").mkdir(parents=True)
    return roots


def test_git_recognized_rename_inherits_only_the_original_source_positions(
    source_roots: tuple[Path, Path],
) -> None:
    base, candidate = source_roots
    (base / "lib/old.py").write_text("import missing  # type: ignore\n")
    (candidate / "lib/new.py").write_text(
        "import missing  # pyright: ignore[reportMissingImports]\n"
    )

    assert (
        suppression_regressions(
            base, candidate, ("lib/new.py",), {"lib/old.py": "lib/new.py"}
        )
        == ()
    )


def test_an_unrecognized_new_file_does_not_inherit_another_files_ignores(
    source_roots: tuple[Path, Path],
) -> None:
    base, candidate = source_roots
    (base / "lib/old.py").write_text("import missing  # type: ignore\n")
    (candidate / "lib/new.py").write_text(
        "import missing  # pyright: ignore[reportMissingImports]\n"
    )

    found = suppression_regressions(base, candidate, ("lib/new.py",), {})

    assert [(item.path, item.rule, item.before, item.after) for item in found] == [
        ("lib/new.py", "suppression:pyright-ignore:new-position@1", 0, 1)
    ]


def test_scope_expansion_is_reported_even_when_raw_marker_count_does_not_rise(
    source_roots: tuple[Path, Path],
) -> None:
    base, candidate = source_roots
    (base / "lib/source.py").write_text(
        "import missing  # pyright: ignore[reportMissingImports]\n"
    )
    (candidate / "lib/source.py").write_text(
        "import missing  # pyright: ignore[reportMissingImports, reportArgumentType]\n"
    )

    found = suppression_regressions(base, candidate, ("lib/source.py",), {})

    assert [(item.path, item.rule, item.before, item.after) for item in found] == [
        ("lib/source.py", "suppression:pyright-ignore:scope-expanded@1", 0, 1)
    ]
