"""Check that lab registry keys match their experiment definition directories."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class KeyPathViolation:
    """One key that does not match its definition directory.

    key is the registered name. actual is the definition directory, or the
    dotted module if no directory can be determined. expected is the required
    key or pattern, or the required module shape when the module is invalid.
    """

    key: str
    actual: str
    expected: str


def check_registry_key_paths(
    entries: Iterable[tuple[str, str]],
) -> tuple[KeyPathViolation, ...]:
    """Return violations in input order for (registry key, definition module) pairs.

    Supply the complete registry as dotted Python modules. A single entry in a
    directory uses the full directory relative to zcu_lab/v2; multiple entries
    in one directory each append exactly one nonempty variant segment. A module
    with no definition directory under zcu_lab/v2 is itself a violation.
    """
    definitions = [
        (key, module, _definition_directory(module)) for key, module in entries
    ]
    counts = Counter(
        directory for _, _, directory in definitions if directory is not None
    )
    violations: list[KeyPathViolation] = []
    for key, module, directory in definitions:
        if directory is None:
            violations.append(
                KeyPathViolation(key, module, "zcu_lab.v2.<directory>.<module>")
            )
        elif counts[directory] == 1:
            if key != directory:
                violations.append(KeyPathViolation(key, directory, directory))
        else:
            parent, _, variant = key.rpartition("/")
            if parent != directory or not variant:
                violations.append(
                    KeyPathViolation(key, directory, f"{directory}/<variant>")
                )
    return tuple(violations)


def _definition_directory(module: str) -> str | None:
    parts = module.split(".")
    if parts[:2] != ["zcu_lab", "v2"] or len(parts) < 4 or not all(parts):
        return None
    return "/".join(parts[2:-1])


def main(argv: Sequence[str] | None = None) -> int:
    """Check this checkout's complete lab registry without opening a GUI or form.

    argv takes no options except --help; None reads sys.argv. Print a JSON
    report with registered key and violation counts plus all violations, and
    return 1 for violations or 0 otherwise. Lab assembly errors propagate.
    """
    argparse.ArgumentParser(description=__doc__).parse_args(argv)

    from zcu_tools.gui.app.measure.registry import Registry

    # ADR-0072: read lab declarations only through its assembly entry point.
    from zcu_lab.assembly import lab_definitions

    lab = lab_definitions()
    registry = Registry(templates=lab.templates)
    lab.register_experiments(registry)

    # The lab defines run in each adapter's own leaf, even for factory adapters
    # whose make_form is bound with functools.partial. Do not invoke either.
    names = registry.list_names()
    violations = check_registry_key_paths(
        (name, registry.create(name).run.__module__) for name in names
    )
    print(
        json.dumps(
            {
                "checked_count": len(names),
                "violation_count": len(violations),
                "violations": [asdict(violation) for violation in violations],
            }
        )
    )
    for violation in violations:
        print(
            f"{violation.key!r}: actual {violation.actual!r}, expected {violation.expected!r}",
            file=sys.stderr,
        )
    return int(bool(violations))


if __name__ == "__main__":
    raise SystemExit(main())
