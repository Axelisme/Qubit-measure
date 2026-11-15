"""List the tracked documents a code change can make stale, and check document references."""

from __future__ import annotations

import argparse
import ast
import posixpath
import re
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final

_RECORD: Final = re.compile(r"ADR-(\d{4})|adr/(?:retired/)?(\d{4})-")
_RECORD_PATH: Final = re.compile(r"^docs/adr/(?:retired/)?(\d{4})-[^/]+\.md$")
_LINK: Final = re.compile(r"\]\(([^)\s]+)\)")
_FENCE: Final = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
_LINE_ANCHOR: Final = re.compile(r"\.py#L\d+")
_DEFAULT_MAX_HITS: Final = 40
# A record cited by more documents than this governs the whole repository; its citers are
# not candidates for one change.
_MAX_CITERS: Final = 8


@dataclass(frozen=True)
class Related:
    """A document tied to a change through the reference graph.

    path is the document's repo-relative POSIX path. reason names the changed
    path or the decision record that ties it to the change.
    """

    path: str
    reason: str


@dataclass(frozen=True)
class Term:
    """A literal to search documents for.

    text is the literal. source is "removed name", "changed name",
    "changed path" or "given".
    """

    text: str
    source: str


@dataclass(frozen=True)
class Hit:
    """One document line containing a term.

    term is the term text, path the document's repo-relative POSIX path, line
    the 1-based line number and text the stripped line.
    """

    term: str
    path: str
    line: int
    text: str


@dataclass(frozen=True)
class Violation:
    """A broken reference.

    path is the document's repo-relative POSIX path, line the 1-based line
    number and reason a human-readable description naming the reference.
    """

    path: str
    line: int
    reason: str


def _is_snapshot(path: str) -> bool:
    """Discussion snapshots and retired ADRs are frozen and never updated."""
    return path.startswith("docs/adr/retired/") or (
        path.startswith("docs/discussions/") and path != "docs/discussions/INDEX.md"
    )


def _records(text: str) -> set[str]:
    return {first or second for first, second in _RECORD.findall(text)}


def _readme_chain(changed: Sequence[str], sources: Mapping[str, str]) -> dict[str, str]:
    """README.md of every ancestor directory of a changed path, with its reason."""
    found: dict[str, str] = {}
    for path in changed:
        for parent in PurePosixPath(path).parents:
            readme = "README.md" if str(parent) == "." else f"{parent}/README.md"
            if readme in sources:
                found.setdefault(readme, f"README chain of {path}")
    return found


def _cited_records(
    changed: Sequence[str], readmes: Sequence[str], sources: Mapping[str, str]
) -> dict[str, str]:
    """Record number -> the document citing it; a README wins over a changed file."""
    cited: dict[str, str] = {}
    for path in (*changed, *readmes):
        for number in sorted(_records(sources.get(path, ""))):
            if path in readmes or number not in cited:
                cited[number] = path
    return cited


def _other_citers(
    cited: Mapping[str, str], skip: Mapping[str, str], sources: Mapping[str, str]
) -> dict[str, list[str]]:
    """Record number -> live documents citing it, apart from records and skip."""
    citers: dict[str, list[str]] = {number: [] for number in cited}
    for path, text in sorted(sources.items()):
        if not path.endswith(".md") or _is_snapshot(path) or path in skip:
            continue
        if _RECORD_PATH.match(path) is None:
            for number in _records(text) & cited.keys():
                citers[number].append(path)
    return citers


def related_documents(
    changed: Sequence[str], sources: Mapping[str, str]
) -> tuple[Related, ...]:
    """List live Markdown documents tied to changed paths by the reference graph.

    changed holds repo-relative POSIX paths. sources maps repo-relative POSIX
    paths to text and must contain every tracked Markdown document; a changed
    file present in sources also contributes the records it cites.

    Three kinds of document are returned, one entry per path, sorted by path:
    every README.md in a changed path's ancestor directories; every
    docs/adr/NNNN-*.md whose number those READMEs or the changed files cite as
    ADR-NNNN or by an adr/NNNN- link; and every other live document citing one
    of those numbers. A record with more than eight other citers is listed
    without them, and its reason says how many were left out. Discussion
    snapshots and retired ADRs are skipped.
    """
    found = _readme_chain(changed, sources)
    cited = _cited_records(changed, sorted(found), sources)
    citers = _other_citers(cited, found, sources)
    for path in sorted(sources):
        record = _RECORD_PATH.match(path)
        if record is None or _is_snapshot(path) or record[1] not in cited:
            continue
        reason = f"ADR-{record[1]} cited by {cited[record[1]]}"
        if len(citers[record[1]]) > _MAX_CITERS:
            reason += f"; its {len(citers[record[1]])} other citers are not listed"
        found[path] = reason
    for number in sorted(cited):
        for path in citers[number] if len(citers[number]) <= _MAX_CITERS else ():
            previous = found.get(path)
            found[path] = (
                f"{previous}, ADR-{number}" if previous else f"cites ADR-{number}"
            )
    return tuple(Related(path, reason) for path, reason in sorted(found.items()))


def _definitions(source: str) -> dict[str, str]:
    """Public top-level functions, classes and upper-case constants, each as a dump."""
    found: dict[str, str] = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            if not node.name.startswith("_"):
                found[node.name] = ast.dump(node)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    found[target.id] = ast.dump(node.value)
    return found


def search_terms(
    before: Mapping[str, str], after: Mapping[str, str]
) -> tuple[Term, ...]:
    """Derive search terms from changed Python sources.

    before and after map repo-relative POSIX paths to source text; a path
    missing from one side was added or deleted. For each path the public
    top-level functions, classes and upper-case constants are compared: a name
    only in before is a "removed name", a name whose definition differs is a
    "changed name". Each path also yields its last two components as a
    "changed path". Method names are left out because they match unrelated
    prose. Terms are deduplicated and sorted by text.

    Raise SyntaxError when a source does not parse.
    """
    terms: dict[str, str] = {}
    for path in sorted(before.keys() | after.keys()):
        old = _definitions(before[path]) if path in before else {}
        new = _definitions(after[path]) if path in after else {}
        for name in old.keys() - new.keys():
            terms[name] = "removed name"
        for name in old.keys() & new.keys():
            if old[name] != new[name]:
                terms.setdefault(name, "changed name")
        terms["/".join(PurePosixPath(path).parts[-2:])] = "changed path"
    return tuple(Term(text, source) for text, source in sorted(terms.items()))


def find_hits(terms: Sequence[Term], documents: Mapping[str, str]) -> tuple[Hit, ...]:
    """Find the lines of live documents that contain a term.

    documents maps repo-relative POSIX paths to text. A "changed path" term
    matches as a substring; every other term matches only as a whole word.
    Discussion snapshots and retired ADRs are skipped. Hits are ordered by
    term in the given order, then by path and line.
    """
    hits: list[Hit] = []
    for term in terms:
        if term.source == "changed path":
            pattern = re.compile(re.escape(term.text))
        else:
            pattern = re.compile(rf"(?<!\w){re.escape(term.text)}(?!\w)")
        for path, text in sorted(documents.items()):
            if _is_snapshot(path):
                continue
            for number, line in enumerate(text.splitlines(), 1):
                if pattern.search(line):
                    hits.append(Hit(term.text, path, number, line.strip()))
    return tuple(hits)


def _unfenced_links(text: str) -> dict[int, list[str]]:
    """Extract link targets outside fenced code, keyed by original line number."""
    links: dict[int, list[str]] = {}
    fence = ""
    for number, line in enumerate(text.splitlines(), 1):
        marker = _FENCE.match(line)
        if fence:
            if marker and marker[1].startswith(fence) and not marker[2].strip():
                fence = ""
            continue
        if marker and (marker[1].startswith("~") or "`" not in marker[2]):
            fence = marker[1]
            continue
        links[number] = _LINK.findall(line)
    return links


def check_references(
    documents: Mapping[str, str], tracked: frozenset[str]
) -> tuple[Violation, ...]:
    """Check relative links and ADR citations in live documents.

    documents maps repo-relative POSIX paths to Markdown text; tracked holds
    every tracked repo-relative POSIX path. A relative link must resolve to a
    tracked file or to a directory containing one, and must not pin a line
    with a .py#L<n> anchor. Links in fenced code blocks, with a URL scheme or
    with in-page anchors are skipped. Every ADR-NNNN, including in fenced code,
    must match a tracked docs/adr/NNNN-*.md, retired
    or not. Discussion snapshots and retired ADRs are skipped. Violations are
    ordered by path, line and position in the line.
    """
    numbers = {match[1] for match in map(_RECORD_PATH.match, tracked) if match}
    directories = {
        str(parent) for path in tracked for parent in PurePosixPath(path).parents
    }
    found: list[Violation] = []
    for path, text in sorted(documents.items()):
        if _is_snapshot(path):
            continue
        links = _unfenced_links(text)
        for number, line in enumerate(text.splitlines(), 1):
            for target in links.get(number, ()):
                if "://" in target or target.startswith(("#", "mailto:")):
                    continue
                if _LINE_ANCHOR.search(target):
                    found.append(
                        Violation(path, number, f"link pins a line number: {target}")
                    )
                    continue
                resolved = posixpath.normpath(
                    posixpath.join(posixpath.dirname(path), target.split("#", 1)[0])
                )
                if resolved not in tracked and resolved not in directories:
                    found.append(
                        Violation(path, number, f"link target is not tracked: {target}")
                    )
            for record in re.findall(r"ADR-(\d{4})", line):
                if record not in numbers:
                    found.append(
                        Violation(path, number, f"record does not exist: ADR-{record}")
                    )
    return tuple(found)


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, check=True
    ).stdout.decode("utf-8")


def _tracked(root: Path) -> tuple[str, ...]:
    return tuple(path for path in _git(root, "ls-files", "-z").split("\0") if path)


def _read(root: Path, paths: Sequence[str]) -> dict[str, str]:
    return {
        path: (root / path).read_text(encoding="utf-8")
        for path in paths
        if (root / path).is_file()
    }


def main(argv: list[str] | None = None) -> int:
    """Print the documents and lines a change touches, or check all references.

    `doc_relations.py <base>` compares commit <base> with the working tree
    under --root (default: this checkout). It prints one `related:` line per
    document from related_documents, then one `hit:` line per line found by
    find_hits for the terms of search_terms plus each --term, as
    `hit: <path>:<line>: <term> (<source>): <text>`. A term with more than
    --max-hits hits prints one `too many:` line instead. Returns 0.

    `doc_relations.py --check` runs check_references over every tracked
    Markdown document, prints `<path>:<line>: <reason>` per violation and
    returns 1 when there is any, else 0.

    argv=None reads sys.argv. Git and file-read errors propagate.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "base", nargs="?", help="commit to compare with the working tree"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="check tracked Markdown references instead of a diff",
    )
    parser.add_argument(
        "--term",
        action="append",
        default=[],
        help="additional literal search term; repeatable",
    )
    parser.add_argument(
        "--max-hits",
        type=int,
        default=_DEFAULT_MAX_HITS,
        help="summarize terms exceeding this hit count (default: 40)",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="checkout to inspect (default: this checkout)",
    )
    options = parser.parse_args(argv)
    root: Path = options.root
    tracked = _tracked(root)
    documents = _read(root, [path for path in tracked if path.endswith(".md")])
    if options.check:
        violations = check_references(documents, frozenset(tracked))
        for item in violations:
            print(f"{item.path}:{item.line}: {item.reason}")
        return 1 if violations else 0
    if options.base is None:
        parser.error("give a base commit, or --check")

    changed: list[str] = []
    before: dict[str, str] = {}
    for line in _git(root, "diff", "--name-status", "-M", options.base).splitlines():
        status, old, *renamed = line.split("\t")
        changed.extend([old, *renamed])
        if old.endswith(".py") and not status.startswith("A"):
            before[old] = _git(root, "show", f"{options.base}:{old}")
    after = _read(root, [path for path in changed if path.endswith(".py")])
    for item in related_documents(changed, {**documents, **_read(root, changed)}):
        print(f"related: {item.path}  ({item.reason})")
    terms = (
        *search_terms(before, after),
        *(Term(text, "given") for text in options.term),
    )
    sources = {term.text: term.source for term in terms}
    hits = find_hits(terms, documents)
    for term in terms:
        own = [hit for hit in hits if hit.term == term.text]
        if len(own) > options.max_hits:
            print(
                f"too many: {term.text} ({term.source}) has {len(own)} hits"
                f" (over {options.max_hits}); use a narrower term"
            )
            continue
        for hit in own:
            print(
                f"hit: {hit.path}:{hit.line}: {hit.term} ({sources[hit.term]}): {hit.text}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
