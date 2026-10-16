"""Strict authored alias data and entry-owned YAML persistence."""

from pathlib import Path
from tempfile import NamedTemporaryFile

from pydantic import BaseModel, ConfigDict, TypeAdapter
from ruamel.yaml import YAML

from zcu_tools.format_version import FormatVersion, YamlMap, validate_header


class EntryAliases(BaseModel):
    """An entry's authored insertion-ordered alias table, not resolved names.

    aliases maps string alias names to raw string bindings in authored order.
    An empty string is unbound; $ links keep their original text. An empty
    mapping is an explicitly saved empty table, distinct from a missing file.
    Non-string keys/values raise ValidationError, never coerce null or numbers.
    Syntax, backward links and component resolution belong to the caller's
    alias table owner, not this storage representation.
    """

    model_config = ConfigDict(strict=True, extra="forbid")
    aliases: dict[str, str]


def load_entry_aliases(source: Path) -> EntryAliases | None:
    """Read source as zcu.aliases major 1, preserving authored mapping order.

    Return None only when source does not exist. Accept forward 1.x minors.
    Bad headers raise FormatError/VersionError; malformed structure or string
    mappings raise ValidationError/ValueError or a YAML parser error. Other I/O
    failures propagate. No fallback, component lookup or file writes occur.
    """
    try:
        stream = source.open(encoding="utf-8")
    except FileNotFoundError:
        return None
    with stream:
        raw = YAML(typ="safe").load(stream)
    return parse_entry_aliases(raw, source=source)


def parse_entry_aliases(raw: object, *, source: Path) -> EntryAliases:
    """Validate an in-memory zcu.aliases 1.x document without resolving links.

    raw is an external document; source labels diagnostics. No I/O occurs.
    Preserve authored row order and strict strings, as load_entry_aliases does.
    Invalid headers, missing rows or non-string bindings raise ValueError.
    """
    document = TypeAdapter(YamlMap).validate_python(raw, strict=True)
    validate_header(
        document,
        expected_format="zcu.aliases",
        supported_version=FormatVersion(1, 0),
        source=source,
    )
    if "aliases" not in document:
        raise ValueError(f"{source}: missing aliases mapping")
    return EntryAliases.model_validate({"aliases": document["aliases"]})


def save_entry_aliases(destination: Path, aliases: EntryAliases) -> None:
    """Replace destination with zcu.aliases 1.0 and this authored table.

    The parent must exist. Strict string data is validated before I/O. Write a
    sibling temporary, close it, then atomically replace the destination. I/O
    failures propagate and retain the old destination; cleanup removes the
    temporary. No lock, multiwriter merge or power-loss guarantee is provided.
    """
    validated = EntryAliases.model_validate(aliases.model_dump())
    document: YamlMap = {
        "format": "zcu.aliases",
        "format_version": "1.0",
        "aliases": dict(validated.aliases),
    }
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            YAML(typ="rt").dump(document, stream)
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
