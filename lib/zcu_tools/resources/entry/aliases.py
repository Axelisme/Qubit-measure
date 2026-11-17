"""Strict authored alias data and entry-owned YAML persistence."""

from dataclasses import dataclass
from hashlib import sha256
from io import StringIO
from pathlib import Path
from tempfile import NamedTemporaryFile

from pydantic import BaseModel, ConfigDict, TypeAdapter
from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError

from zcu_tools.format_version import FormatVersion, YamlMap, validate_header
from zcu_tools.utils import atomic_replace

from ..document_store import DocumentRevision

type AliasValue = str | list[str]
"""An authored or resolved scalar name or ordered name list; empty means unbound."""


class EntryAliases(BaseModel):
    """A global or point authored insertion-ordered alias table, not resolved names.

    aliases maps names to raw scalar bindings or ordered lists in authored order.
    An empty string is unbound; $ links keep their original text. An empty
    mapping is an explicitly saved empty table, distinct from a missing file.
    Non-string keys or elements raise ValidationError; null/numbers never coerce.
    Syntax, backward links and component resolution belong to the caller's
    alias table owner, not this storage representation.
    """

    model_config = ConfigDict(strict=True, extra="forbid")
    aliases: dict[str, AliasValue]


@dataclass(frozen=True)
class AliasReadResult:
    """One raw-byte observation, including a deferred content failure.

    aliases is the parsed table, or an empty table for missing/invalid content.
    revision hashes the exact observed bytes, including invalid content; None
    means absent only. error is the original decoding/header/schema exception,
    or None for a valid table or missing file. Callers must raise error before
    consuming aliases. Filesystem errors other than absence are not deferred.
    """

    aliases: EntryAliases
    revision: DocumentRevision
    error: Exception | None = None

    def require_valid(self) -> EntryAliases:
        """Return this observation's table, or raise its deferred format error."""
        if self.error is not None:
            raise self.error
        return self.aliases


class AliasFile:
    """A cached observation of one global or point aliases.yaml.

    Construction reads only source, once. Missing files give an empty table and
    revision None; format errors are deferred in read().error. Other I/O errors
    raise. This handle validates file format, not session alias syntax. Callers
    with extra admission rules must open a new candidate rather than refresh an
    installed handle. required_parameters gates point saves only.
    """

    def __init__(
        self, source: Path, *, required_parameters: Path | None = None
    ) -> None:
        self._source = source
        self._required_parameters = required_parameters
        self._observation = read_entry_aliases(source)

    def read(self) -> AliasReadResult:
        """Return an isolated copy of the held observation, without I/O."""
        observed = self._observation
        return AliasReadResult(
            observed.aliases.model_copy(deep=True), observed.revision, observed.error
        )

    def save(self, aliases: EntryAliases) -> DocumentRevision:
        """Atomically replace this file, without optimistic locking or rereads.

        aliases contains authored ordered scalar/list bindings; session syntax
        belongs to the caller. Point handles require parameters.yaml to exist,
        without parsing it. Success caches an isolated input copy and the exact
        writer revision. Validation/I/O failure retains memory and the old file.
        No other document is read or changed.
        """
        if (
            self._required_parameters is not None
            and not self._required_parameters.is_file()
        ):
            raise FileNotFoundError(self._required_parameters)
        copied = aliases.model_copy(deep=True)
        revision = save_entry_aliases(self._source, copied)
        self._observation = AliasReadResult(copied, revision)
        return revision

    def refresh(self, *, strict: bool = False) -> bool:
        """Reread only this file; report ordered content or validity changes.

        Lax mode installs deferred format errors. Strict mode raises those
        errors without installation. I/O failures always retain the observation.
        Header/comment-only changes and missing/valid-empty transitions return
        False but update revision. Two invalid observations compare byte hashes.
        Session syntax validation is not performed; use a fresh candidate for
        extra admission rules rather than refreshing an installed handle.
        """
        candidate = read_entry_aliases(self._source)
        if strict and candidate.error is not None:
            raise candidate.error
        previous = self._observation
        if previous.error is not None and candidate.error is not None:
            changed = previous.revision != candidate.revision
        elif previous.error is not None or candidate.error is not None:
            changed = True
        else:
            changed = list(previous.aliases.aliases.items()) != list(
                candidate.aliases.aliases.items()
            )
        self._observation = candidate
        return changed


def read_entry_aliases(source: Path) -> AliasReadResult:
    """Observe aliases once, retaining content errors and the same bytes hash.

    Missing source yields an empty table and revision=None. Invalid UTF-8,
    YAML, header or schema yields the original error with an empty table, not a
    valid fallback. Other I/O errors propagate. No files are written.
    """
    try:
        content = source.read_bytes()
    except FileNotFoundError:
        return AliasReadResult(
            EntryAliases(aliases={}), DocumentRevision(source.resolve(), None)
        )
    revision = DocumentRevision(source.resolve(), sha256(content).hexdigest())
    try:
        raw = YAML(typ="safe").load(content.decode("utf-8"))
        aliases = parse_entry_aliases(raw, source=source)
    except (ValueError, YAMLError) as error:
        return AliasReadResult(EntryAliases(aliases={}), revision, error)
    return AliasReadResult(aliases, revision)


def load_entry_aliases(source: Path) -> EntryAliases | None:
    """Read source as zcu.aliases major 1, preserving authored mapping order.

    Return None only when source does not exist. Accept forward 1.x minors.
    Bad headers raise FormatError/VersionError; malformed structure or non-string
    elements raise ValidationError/ValueError or a YAML parser error. Other I/O
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
    result = read_entry_aliases(source)
    if result.error is not None:
        raise result.error
    return (
        None if result.revision.revision is None else result.aliases
    ), result.revision


def parse_entry_aliases(raw: object, *, source: Path) -> EntryAliases:
    """Validate an in-memory zcu.aliases 1.x document without resolving links.

    raw is an external document; source labels diagnostics. No I/O occurs.
    Preserve authored row/element order and strict strings, as load_entry_aliases does.
    Invalid headers, missing rows or non-string elements raise ValueError.
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

    The parent must exist. Strict scalar/list data is validated before I/O. Write a
    sibling temporary, close it, then atomically replace the destination. I/O
    failures propagate and retain the old destination; cleanup removes the
    temporary. Return the version of the exact bytes written. No lock,
    multiwriter merge or power-loss guarantee is provided.
    """
    validated = EntryAliases.model_validate(aliases.model_dump())
    document = TypeAdapter(YamlMap).validate_python(
        {
            "format": "zcu.aliases",
            "format_version": "1.0",
            "aliases": validated.aliases,
        },
        strict=True,
    )
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
