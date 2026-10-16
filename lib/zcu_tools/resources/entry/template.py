"""Portable component and alias documents, without entry identity or roots."""

from dataclasses import dataclass
from pathlib import Path

from pydantic import TypeAdapter
from ruamel.yaml import YAML

from zcu_tools.format_version import FormatVersion, YamlMap, validate_header

from .aliases import EntryAliases, parse_entry_aliases
from .schema import ComponentSchema, PointDocument


@dataclass(frozen=True)
class EntryTemplate:
    """Validated startup data, not a saved setup or point.

    components maps public names to models from the caller-bootstrapped shared
    component_registry, in authored order. Models include their schema defaults.
    aliases holds strict ordered raw strings; GUI owns grammar and resolution.
    Neither field includes entry identity, timestamps, roots or module assets.
    """

    components: dict[str, ComponentSchema]
    aliases: EntryAliases


def load_entry_template(source: Path) -> EntryTemplate:
    """Read a zcu.entry-template 1.x YAML with components and an alias document.

    The caller must register component models before calling. Components use
    the same names, kind lookup and original model validation as point files.
    aliases is a nested complete zcu.aliases 1.x document, not resolved names.
    Reject bad headers, missing sections, kinds, fields and non-string alias
    bindings at load; syntax/backward links belong to the GUI's AliasTable.
    I/O and YAML errors propagate. No entry is created and no file is written.
    """
    with source.open(encoding="utf-8") as stream:
        raw = YAML(typ="safe").load(stream)
    document = TypeAdapter(YamlMap).validate_python(raw, strict=True)
    validate_header(
        document,
        expected_format="zcu.entry-template",
        supported_version=FormatVersion(1, 0),
        source=source,
    )
    for section in ("components", "aliases"):
        if section not in document:
            raise ValueError(f"{source}: missing {section} section")

    class TemplateComponents(PointDocument):
        _source = source

    components = TemplateComponents.validate_components(document["components"])
    aliases = parse_entry_aliases(document["aliases"], source=source)
    return EntryTemplate(components=components, aliases=aliases)
