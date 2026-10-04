"""Resources-internal prepared-file lifetime; not a public storage interface."""

from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

from pydantic import BaseModel

from zcu_tools.format_version import YamlMap

type FieldPath = tuple[str, ...]


@dataclass
class DocumentEdit[T: BaseModel]:
    base_document: YamlMap
    base: T
    draft: T


def stage_content(source: Path, content: bytes) -> Path:
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
    source: Path
    document: YamlMap
    snapshot: T
    paths: tuple[FieldPath, ...]
    original: bytes
    temporary: Path | None

    def replace(self) -> None:
        if self.temporary is not None:
            self.temporary.replace(self.source)

    def restore(self) -> None:
        temporary = stage_content(self.source, self.original)
        try:
            temporary.replace(self.source)
        finally:
            temporary.unlink(missing_ok=True)

    def discard(self) -> None:
        if self.temporary is not None:
            self.temporary.unlink(missing_ok=True)
