"""Shared strict JSON I/O for the two entry analysis file owners."""

from pathlib import Path
from tempfile import NamedTemporaryFile

from pydantic import BaseModel

from zcu_tools.utils import atomic_replace


def load_file[Model: BaseModel](path: Path, model: type[Model]) -> Model:
    return model.model_validate_json(path.read_text(encoding="utf-8"))


def save_file[Model: BaseModel](
    path: Path, document: Model, model: type[Model]
) -> None:
    # Validate serialized data too: model_copy can bypass model validation.
    validated = model.model_validate(document.model_dump(exclude_unset=True))
    payload = validated.model_dump_json(indent=2, exclude_unset=True) + "\n"
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
        atomic_replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
