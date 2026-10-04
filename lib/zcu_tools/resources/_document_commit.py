"""Resources-internal prepared-file lifetime; not a public storage interface."""

from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

from pydantic import BaseModel

from zcu_tools.format_version import YamlMap

type FieldPath = tuple[str, ...]


@dataclass
class DocumentEdit[T: BaseModel]:
    """An active resources-only edit: base_document is original raw SI YAML.

    base is its validated working-unit model; draft is an independent mutable
    working copy. The owning store keeps edit ownership until context exit.
    """

    base_document: YamlMap
    base: T
    draft: T


def stage_content(source: Path, content: bytes) -> Path:
    """Write content to a sibling temp file; caller owns replace/unlink on return.

    Failure removes any created temporary and propagates the original exception.
    """
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="wb",
            dir=source.parent,
            prefix=f".{source.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        return temporary
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


@dataclass
class PreparedDocument[T: BaseModel]:
    """Resources-only preflight result; it owns a temporary until discard().

    source is the target YAML path; document is the merged SI round-trip tree.
    snapshot is the validated working-unit model, paths its changed typed paths.
    original holds exact pre-replacement disk bytes for ordinary restoration.
    temporary is a staged sibling file, or None for an empty edit. This object
    never publishes store memory or delivers observers.
    """

    source: Path
    document: YamlMap
    snapshot: T
    paths: tuple[FieldPath, ...]
    original: bytes
    temporary: Path | None

    def replace(self) -> None:
        """Atomically replace source with staged bytes, or do nothing for an empty edit.

        I/O failures propagate. Caller publishes only after success and must
        discard the temporary whether this operation succeeds or fails.
        """
        if self.temporary is not None:
            self.temporary.replace(self.source)

    def restore(self) -> None:
        """Replace source with original bytes; propagate I/O errors and clean its temp."""
        temporary = stage_content(self.source, self.original)
        try:
            temporary.replace(self.source)
        finally:
            temporary.unlink(missing_ok=True)

    def discard(self) -> None:
        """Unlink any remaining staged temp without changing source or store memory."""
        if self.temporary is not None:
            self.temporary.unlink(missing_ok=True)
