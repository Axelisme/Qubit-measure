"""Materialize an entry template, then open the ordinary entry in the GUI."""

import argparse
from pathlib import Path

from pydantic import TypeAdapter
from zcu_tools.format_version import YamlMap
from zcu_tools.resources.entry import (
    ResultEntry,
    load_entry_template,
    validate_path_segment,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def materialize_entry(
    template: Path,
    name: str,
    *,
    project_root: Path,
    point_label: str = "initial",
) -> ResultEntry:
    """Create setup, aliases and the first point under project_root.

    template is a zcu.entry-template file. The caller must bootstrap the shared
    component registry. name and point_label are safe single path segments.
    Roots are project_root/result and project_root/Database, as in measure GUI.
    Template validation precedes creation. Either existing entry destination
    raises DestinationExistsError without overwriting it. Later I/O/commit
    failures propagate and may leave a partial new entry; no multi-file
    transaction or rollback is promised. Return the created ResultEntry.
    """
    loaded = load_entry_template(template)
    validate_path_segment(point_label, field="point_label")
    entry = ResultEntry.create(
        name,
        result_root=project_root / "result",
        database_root=project_root / "Database",
    )
    for component_name, component in loaded.components.items():
        # Omitted optional wiring channels must stay absent, not explicit null.
        fields = TypeAdapter(YamlMap).validate_python(
            component.model_dump(exclude_unset=True)
        )
        fields.pop("kind")
        entry.setup.add_component(component_name, kind=component.kind, **fields)
    entry.save_aliases(loaded.aliases)
    entry.new_point(point_label)
    return entry


def main(argv: list[str] | None = None) -> int:
    """Run the entry CLI, registering lab component definitions at composition.

    argv supplies template and entry name plus optional --project-root and
    --point-label. None reads sys.argv. Errors propagate without overwriting an
    existing entry. Successful creation prints its result path and returns 0.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("template", type=Path)
    parser.add_argument("name")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--point-label", default="initial")
    args = parser.parse_args(argv)

    from zcu_tools.resources.entry import component_registry

    from zcu_lab.components import register_all

    register_all(component_registry)
    entry = materialize_entry(
        args.template,
        args.name,
        project_root=args.project_root,
        point_label=args.point_label,
    )
    print(entry.result_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
