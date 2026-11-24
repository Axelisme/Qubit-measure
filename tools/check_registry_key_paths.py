"""Check that lab registry keys match their experiment definition directories."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zcu_tools.gui.app.measure.registry import Registry


_MODULE_SHAPE = "zcu_lab.v2.<directory>.<module>"


@dataclass(frozen=True)
class KeyDefinitionModule:
    """A registry key paired with the dotted name of its direct declaration module.

    key is the adapter's registered name. module is the Python module that
    declares the registered Adapter value or its factory, not the Run hook.
    """

    key: str
    module: str


@dataclass(frozen=True)
class KeyPathViolation:
    """One key with an invalid source record or a mismatched directory.

    key is the registry name being checked. actual and expected describe the
    observed and required source count, module name, or directory/key pair.
    """

    key: str
    actual: str
    expected: str


def check_registry_key_paths(
    entries: Iterable[KeyDefinitionModule],
) -> tuple[KeyPathViolation, ...]:
    """Return violations in input order for named key and definition module pairs.

    Supply the complete registry as dotted Python modules. A single entry in a
    directory uses the full directory relative to zcu_lab/v2; multiple entries
    in one directory each append exactly one nonempty variant segment. A module
    with no definition directory under zcu_lab/v2 is itself a violation.
    """
    definitions = [(entry, _definition_directory(entry.module)) for entry in entries]
    counts = Counter(directory for _, directory in definitions if directory is not None)
    violations: list[KeyPathViolation] = []
    # ADR-0074: the key carries the full directory and, only for shared leaves,
    # one variant segment.
    for entry, directory in definitions:
        key = entry.key
        if directory is None:
            violations.append(KeyPathViolation(key, entry.module, _MODULE_SHAPE))
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
    if parts[:2] != ["zcu_lab", "v2"] or len(parts) < 4:
        return None
    if not all(part.isidentifier() for part in parts[2:]):
        return None
    return "/".join(parts[2:-1])


def check_registered_key_paths(
    registry: Registry,
    register_experiments: Callable[[Registry, Callable[[str, ModuleType], None]], None],
) -> tuple[KeyPathViolation, ...]:
    """Fill the supplied empty Registry and check the declared definition directories.

    The registrar receives a sink to call with the key and declaration module
    after each registration. Return missing, duplicate, extra or invalid source
    violations first: variant detection needs a complete set of valid sources.
    Otherwise return directory violations. Registration errors propagate; no
    adapter hooks run.
    """
    sources: dict[str, list[ModuleType]] = {}

    def record_source(key: str, source: ModuleType) -> None:
        sources.setdefault(key, []).append(source)

    register_experiments(registry, record_source)
    entries: list[KeyDefinitionModule] = []
    violations: list[KeyPathViolation] = []
    for key in registry.list_names():
        records = sources.pop(key, [])
        if len(records) != 1:
            violations.append(
                KeyPathViolation(
                    key,
                    f"{len(records)} definition sources",
                    "exactly 1 definition source",
                )
            )
            continue
        module = _source_module_name(records[0])
        if _definition_directory(module) is None:
            violations.append(KeyPathViolation(key, module, _MODULE_SHAPE))
        else:
            entries.append(KeyDefinitionModule(key, module))
    violations.extend(
        KeyPathViolation(key, "source for an unregistered key", "a registered key")
        for key in sources
    )
    return tuple(violations) if violations else check_registry_key_paths(entries)


def _source_module_name(source: object) -> str:
    if not isinstance(source, ModuleType):
        return f"{type(source).__name__} instead of a module"
    name: object = getattr(source, "__name__", None)
    return name if isinstance(name, str) else "module without a string __name__"


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
    violations = check_registered_key_paths(registry, lab.register_experiments)
    print(
        json.dumps(
            {
                "checked_count": len(registry.list_names()),
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
