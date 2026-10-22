"""Preallocated, row-oriented container writing for long acquisitions."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import TracebackType

import h5py as h5
import numpy as np

from .container_models import ContainerAttrs, JsonObject, Variable, VariableSchema
from .container_nodes import (
    json_text,
    name_segment,
    validate_variable,
    write_attrs,
    write_doc,
)


class AppendWriter:
    """Own one open HDF5 file until close or context exit; not thread-safe.

    Use open_append to create it. Writes flush before returning. Context exit
    closes without finalizing; partial files remain readable. finalize only
    marks producer-declared completion, not measurement completeness. No resume,
    multi-writer or power-loss guarantee. Writes after close/finalize fail.
    """

    def __init__(self, file: h5.File) -> None:
        """Take ownership of the freshly initialized file from open_append."""
        self._file = file
        self._finalized = False

    def __enter__(self) -> AppendWriter:
        """Return the writer; context exit closes it."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the file, retaining already-flushed rows even on failure."""
        self.close()

    def _writable(self) -> None:
        if not self._file.id.valid:
            raise ValueError("append writer is closed")
        if self._finalized:
            raise ValueError("append writer is finalized")

    def write_row(self, variable: str, row: int, values: np.ndarray) -> None:
        """Replace one outermost NumPy row and flush it.

        variable must be a declared name. row must be in [0, shape[0]); values
        must have shape[1:] and a safely convertible numeric dtype. Unknown
        variables raise KeyError; invalid row, shape or dtype raises ValueError.
        Rows can be rewritten; unwritten rows keep their declared fill value.
        """
        self._writable()
        name_segment(variable)
        dataset = self._file[f"vars/{variable}/values"]
        if not isinstance(dataset, h5.Dataset):
            raise ValueError(f"{variable}: missing values dataset")
        if row < 0 or row >= dataset.shape[0]:
            raise ValueError(f"{variable}: row out of range")
        if values.shape != dataset.shape[1:]:
            raise ValueError(f"{variable}: row shape must be {dataset.shape[1:]}")
        if not np.can_cast(values.dtype, dataset.dtype, casting="safe"):
            raise ValueError(f"{variable}: incompatible row dtype")
        dataset[row] = values
        self._file.flush()

    def replace_doc(self, name: str, doc: JsonObject) -> None:
        """Replace or add one whole finite JSON object and flush; never merge keys."""
        self._writable()
        documents = self._file["docs"]
        if not isinstance(documents, h5.Group):
            raise ValueError("missing docs group")
        write_doc(documents, name, doc)
        self._file.flush()

    def finalize(self) -> None:
        """Flush and mark finalized=True; reject further writes, but leave open."""
        self._writable()
        self._file.attrs["finalized"] = True
        self._file.flush()
        self._finalized = True

    def close(self) -> None:
        """Close the owned file; repeated close is allowed and does not finalize."""
        self._file.close()


def open_append(
    path: Path,
    *,
    vars: Mapping[str, VariableSchema],
    docs: Mapping[str, JsonObject] | None = None,
    producer: str,
    kind: str | None = None,
) -> AppendWriter:
    """Create a new preallocated container at exact path and return its owner.

    All variables must have at least one dimension and complete ordered axes.
    Values initialize to each schema's fill_value without allocating full arrays
    in memory. Existing files raise FileExistsError. Invalid schema/doc input
    raises ValueError before creation. Context exit leaves finalized=False unless
    finalize was called. I/O failures propagate; partial files are not rolled back.
    """
    for name, schema in vars.items():
        name_segment(name)
        if not schema.shape or any(size < 0 for size in schema.shape):
            raise ValueError(f"{name}: append shape needs an outer dimension")
        validate_variable(
            Variable(
                np.broadcast_to(np.zeros((), dtype=schema.dtype), schema.shape),
                schema.axes,
                schema.unit,
            )
        )
        np.full((), schema.fill_value, dtype=schema.dtype)
    for name, doc in (docs or {}).items():
        name_segment(name)
        json_text(doc)
    file = h5.File(path, "x")
    try:
        write_attrs(file, ContainerAttrs("3.0", producer, kind, {"finalized": False}))
        variables = file.create_group("vars")
        documents = file.create_group("docs")
        file.create_group("groups")
        for name, schema in vars.items():
            _create_variable(variables.create_group(name), schema)
        for name, doc in (docs or {}).items():
            write_doc(documents, name, doc)
        file.flush()
    except (ValueError, TypeError, OSError):
        file.close()
        raise
    return AppendWriter(file)


def _create_variable(group: h5.Group, schema: VariableSchema) -> None:
    values = group.create_dataset(
        "values", shape=schema.shape, dtype=schema.dtype, fillvalue=schema.fill_value
    )
    values.attrs["unit"] = schema.unit
    axes = group.create_group("axes")
    for index, axis in enumerate(schema.axes):
        dataset = axes.create_dataset(axis.name, data=axis.values)
        dataset.attrs["unit"] = axis.unit
        dataset.make_scale(axis.name)
        values.dims[index].attach_scale(dataset)
