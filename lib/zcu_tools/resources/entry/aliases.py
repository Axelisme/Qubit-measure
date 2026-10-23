"""Strict authored alias data and entry-owned YAML persistence."""

from hashlib import sha256
from io import StringIO
from pathlib import Path
from tempfile import NamedTemporaryFile

from pydantic import BaseModel, ConfigDict, TypeAdapter
from ruamel.yaml import YAML

from zcu_tools.format_version import FormatVersion, YamlMap, validate_header
from zcu_tools.utils import atomic_replace

from ..document_store import DocumentRevision


class EntryAliases(BaseModel):
    """A setup or point authored insertion-ordered alias table, not resolved names.

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
    aliases, _ = load_entry_aliases_with_revision(source)
    return aliases


def load_entry_aliases_with_revision(
    source: Path,
) -> tuple[EntryAliases | None, DocumentRevision]:
    """Read and validate aliases and hash exactly those bytes in one read.

    Missing source returns (None, revision=None). Other failures propagate
    without a result, as load_entry_aliases does. No file writes occur.
    """
    try:
        content = source.read_bytes()
    except FileNotFoundError:
        return None, DocumentRevision(source.resolve(), None)
    raw = YAML(typ="safe").load(content.decode("utf-8"))
    aliases = parse_entry_aliases(raw, source=source)
    return aliases, DocumentRevision(source.resolve(), sha256(content).hexdigest())


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


def save_entry_aliases(destination: Path, aliases: EntryAliases) -> DocumentRevision:
    """Replace destination with zcu.aliases 1.0 and this authored table.

    The parent must exist. Strict string data is validated before I/O. Write a
    sibling temporary, close it, then atomically replace the destination. I/O
    failures propagate and retain the old destination; cleanup removes the
    temporary. Return the version of the exact bytes written. No lock,
    multiwriter merge or power-loss guarantee is provided.
    """
    validated = EntryAliases.model_validate(aliases.model_dump())
    document: YamlMap = {
        "format": "zcu.aliases",
        "format_version": "1.0",
        "aliases": dict(validated.aliases),
    }
    rendered = StringIO()
    YAML(typ="rt").dump(document, rendered)
    content = rendered.getvalue().encode("utf-8")
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        atomic_replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return DocumentRevision(destination.resolve(), sha256(content).hexdigest())
