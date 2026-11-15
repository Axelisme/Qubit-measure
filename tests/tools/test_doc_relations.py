"""Document-relation behavior through synthetic source mappings and one real Git diff."""

import subprocess
from pathlib import Path

import pytest
from doc_relations import (
    Hit,
    Related,
    Term,
    Violation,
    check_references,
    find_hits,
    main,
    related_documents,
    search_terms,
)

pytestmark = pytest.mark.requires_subprocess

_KERNEL = "lib/pkg/engine/kernels/t6.py"
_SOURCES = {
    "lib/pkg/engine/README.md": "Engine notes, governed by ADR-0078.\n",
    "lib/pkg/engine/kernels/README.md": "Kernel notes.\n",
    "lib/pkg/README.md": "Package notes.\n",
    "lib/pkg/sim/README.md": "The simulator borrows the plan, see ADR-0078.\n",
    "lib/pkg/gui/README.md": "GUI notes, see ADR-0067.\n",
    "docs/adr/0078-engine-boundary.md": "# 0078\n",
    "docs/adr/0067-gui.md": "# 0067\n",
    "docs/discussions/2026-01-01-engine.md": "Snapshot citing ADR-0078.\n",
    "docs/adr/retired/0011-old.md": "Retired text citing ADR-0078.\n",
}


def test_related_documents_follow_readme_chain_and_shared_records() -> None:
    found = related_documents([_KERNEL], _SOURCES)

    assert found == (
        Related(
            "docs/adr/0078-engine-boundary.md",
            "ADR-0078 cited by lib/pkg/engine/README.md",
        ),
        Related("lib/pkg/README.md", f"README chain of {_KERNEL}"),
        Related("lib/pkg/engine/README.md", f"README chain of {_KERNEL}"),
        Related("lib/pkg/engine/kernels/README.md", f"README chain of {_KERNEL}"),
        Related("lib/pkg/sim/README.md", "cites ADR-0078"),
    )


def test_related_documents_include_records_cited_by_changed_code() -> None:
    sources = {**_SOURCES, "lib/pkg/gui/view.py": "# layout rule: ADR-0067\n"}

    found = related_documents(["lib/pkg/gui/view.py"], sources)

    assert (
        Related("docs/adr/0067-gui.md", "ADR-0067 cited by lib/pkg/gui/README.md")
        in found
    )


def test_related_documents_list_a_widely_cited_record_without_its_citers() -> None:
    sources = {
        "lib/pkg/README.md": "Placement follows ADR-0072.\n",
        "docs/adr/0072-boundaries.md": "# 0072\n",
        **{f"lib/other{n}/README.md": "See ADR-0072.\n" for n in range(9)},
    }

    found = related_documents(["lib/pkg/mod.py"], sources)

    assert found == (
        Related(
            "docs/adr/0072-boundaries.md",
            "ADR-0072 cited by lib/pkg/README.md; its 9 other citers are not listed",
        ),
        Related("lib/pkg/README.md", "README chain of lib/pkg/mod.py"),
    )


def test_search_terms_name_removed_and_changed_public_definitions() -> None:
    before = {
        _KERNEL: "LIMIT = 3\n\ndef advance(x):\n    return x\n\ndef gone():\n    pass\n\n"
        "class Plan:\n    def close(self):\n        return 1\n\ndef _private():\n    pass\n"
    }
    after = {
        _KERNEL: "LIMIT = 4\n\ndef advance(x):\n    return x\n\n"
        "class Plan:\n    def close(self):\n        return 2\n\ndef _private():\n    return 1\n"
    }

    assert search_terms(before, after) == (
        Term("LIMIT", "changed name"),
        Term("Plan", "changed name"),
        Term("gone", "removed name"),
        Term("kernels/t6.py", "changed path"),
    )


def test_search_terms_treat_a_deleted_file_as_removed_names() -> None:
    before = {"tools/old.py": "def helper():\n    pass\n"}

    assert search_terms(before, {}) == (
        Term("helper", "removed name"),
        Term("tools/old.py", "changed path"),
    )


def test_find_hits_match_names_as_whole_words_in_live_documents() -> None:
    documents = {
        "lib/pkg/sim/README.md": "The `Plan` owns caches.\nPlanner is unrelated.\n",
        "docs/discussions/2026-01-01-engine.md": "Plan as it was.\n",
        "docs/guide.md": "See [t6](../lib/pkg/engine/kernels/t6.py).\n",
    }
    terms = [Term("Plan", "changed name"), Term("kernels/t6.py", "changed path")]

    assert find_hits(terms, documents) == (
        Hit("Plan", "lib/pkg/sim/README.md", 1, "The `Plan` owns caches."),
        Hit(
            "kernels/t6.py",
            "docs/guide.md",
            1,
            "See [t6](../lib/pkg/engine/kernels/t6.py).",
        ),
    )


def test_check_references_reports_missing_targets_line_anchors_and_records() -> None:
    documents = {
        "lib/pkg/README.md": (
            "[ok](engine/README.md) [gone](engine/missing.md)\n"
            "[pinned](engine/kernels/t6.py#L12) [web](https://example.org) [top](#notes)\n"
            "Governed by ADR-0078 and ADR-0999.\n"
            "[directory](engine/) [section](engine/README.md#notes) "
            "[mail](mailto:reader@example.org)\n"
        ),
        "docs/discussions/2026-01-01-engine.md": "[stale](../gone.md)\n",
        "docs/adr/retired/0011-old.md": "[stale](../gone.md) ADR-0999\n",
    }
    tracked = frozenset(
        {
            "lib/pkg/README.md",
            "lib/pkg/engine/README.md",
            "lib/pkg/engine/kernels/t6.py",
            "docs/adr/0078-engine-boundary.md",
        }
    )

    assert check_references(documents, tracked) == (
        Violation(
            "lib/pkg/README.md", 1, "link target is not tracked: engine/missing.md"
        ),
        Violation(
            "lib/pkg/README.md",
            2,
            "link pins a line number: engine/kernels/t6.py#L12",
        ),
        Violation("lib/pkg/README.md", 3, "record does not exist: ADR-0999"),
    )


def test_check_references_skips_fenced_links_but_keeps_prose_lines_and_adrs() -> None:
    documents = {
        "guide.md": (
            "[before](first.md)\n"
            "```markdown\n"
            "[example](missing.md) [pinned](module.py#L12)\n"
            "ADR-0999\n"
            "```\n"
            "[after](last.md)\n"
        )
    }

    assert check_references(documents, frozenset()) == (
        Violation("guide.md", 1, "link target is not tracked: first.md"),
        Violation("guide.md", 4, "record does not exist: ADR-0999"),
        Violation("guide.md", 6, "link target is not tracked: last.md"),
    )


@pytest.mark.parametrize(
    ("opening", "non_closers", "closing"),
    [
        (
            "   ````markdown",
            ("```", "~~~~", "```` trailing", "    ````"),
            " ````` \t",
        ),
        (
            " ~~~markdown `info`",
            ("~~", "```", "~~~ trailing", "    ~~~"),
            "  ~~~~ \t",
        ),
    ],
    ids=["backticks", "tildes"],
)
def test_check_references_requires_a_matching_fence_of_sufficient_length(
    opening: str, non_closers: tuple[str, ...], closing: str
) -> None:
    lines = [opening, "[example](hidden.md)"]
    for line in non_closers:
        lines.extend([line, "[still example](hidden.md)"])
    lines.extend([closing, "[prose](missing.md)"])

    assert check_references({"guide.md": "\n".join(lines)}, frozenset()) == (
        Violation("guide.md", 12, "link target is not tracked: missing.md"),
    )


@pytest.mark.parametrize(
    "opening", ["``markdown", "    ```markdown", "```markdown `info`"]
)
def test_check_references_does_not_treat_invalid_openers_as_fenced_code(
    opening: str,
) -> None:
    documents = {"guide.md": f"{opening}\n[prose](missing.md)\n"}

    assert check_references(documents, frozenset()) == (
        Violation("guide.md", 2, "link target is not tracked: missing.md"),
    )


def test_check_references_ends_unclosed_fences_at_the_document_boundary() -> None:
    documents = {
        "b.md": "[prose](missing.md)\n",
        "a.md": "~~~markdown [info](example.md)\n[example](hidden.md)\n",
    }

    assert check_references(documents, frozenset()) == (
        Violation("b.md", 1, "link target is not tracked: missing.md"),
    )


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@example.com",
            *args,
        ],
        check=True,
        capture_output=True,
    )


def test_main_lists_related_documents_and_hits_for_a_diff(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "docs").mkdir()
    (tmp_path / "pkg" / "README.md").write_text("Notes.\n", encoding="utf-8")
    (tmp_path / "pkg" / "plan.py").write_text(
        "class Plan:\n    size = 1\n", encoding="utf-8"
    )
    (tmp_path / "docs" / "guide.md").write_text(
        "The `Plan` is small.\n", encoding="utf-8"
    )
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "base")
    (tmp_path / "pkg" / "plan.py").write_text(
        "class Plan:\n    size = 2\n", encoding="utf-8"
    )

    assert main(["HEAD", "--root", str(tmp_path), "--term", "small"]) == 0

    lines = capsys.readouterr().out.splitlines()
    assert "related: pkg/README.md  (README chain of pkg/plan.py)" in lines
    assert "hit: docs/guide.md:1: Plan (changed name): The `Plan` is small." in lines
    assert "hit: docs/guide.md:1: small (given): The `Plan` is small." in lines


def test_main_replaces_the_hits_of_a_term_over_the_limit_with_a_count(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "notes.md").write_text("alpha\nalpha\nalpha\n", encoding="utf-8")
    (tmp_path / "a.py").write_text("X = 1\n", encoding="utf-8")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "base")

    assert (
        main(["HEAD", "--root", str(tmp_path), "--term", "alpha", "--max-hits", "2"])
        == 0
    )

    out = capsys.readouterr().out
    assert "too many: alpha (given) has 3 hits (over 2); use a narrower term" in out
    assert "hit: notes.md" not in out


@pytest.mark.parametrize("broken", [False, True])
def test_main_check_reports_broken_references_and_exit_code(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], *, broken: bool
) -> None:
    target = "missing.md" if broken else "other.md"
    (tmp_path / "guide.md").write_text(
        f"[x]({target})\n```markdown\n[example](local.png)\n```\n", encoding="utf-8"
    )
    (tmp_path / "other.md").write_text("ok\n", encoding="utf-8")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", "-A")

    assert main(["--check", "--root", str(tmp_path)]) == (1 if broken else 0)

    out = capsys.readouterr().out
    assert out == (
        "guide.md:1: link target is not tracked: missing.md\n" if broken else ""
    )
