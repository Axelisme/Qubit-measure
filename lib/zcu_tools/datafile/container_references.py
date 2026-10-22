"""Retarget local HDF5 references when a same-shape dataset changes dtype."""

import h5py as h5
import numpy as np


def _has_references(dtype: np.dtype) -> bool:
    if h5.check_dtype(ref=dtype) is not None:
        return True
    if dtype.names:
        return any(_has_references(dtype[name]) for name in dtype.names)
    element = h5.check_dtype(vlen=dtype)
    return isinstance(element, np.dtype) and _has_references(element)


def _retarget_reference(
    value: h5.Reference, source: h5.Dataset, replacement: h5.Dataset
) -> tuple[h5.Reference, bool]:
    if not value:
        return value, False
    try:
        target = source.file[value]
    except KeyError:
        # Opaque content may already contain dangling references. Preserve it.
        return value, False
    if target.id != source.id:
        return value, False
    if isinstance(value, h5.RegionReference):
        name = replacement.name
        if name is None:
            raise ValueError("reference replacement must remain linked")
        selection = h5.h5r.get_region(value, source.file.id)
        return h5.h5r.create(
            replacement.file.id,
            name.encode("utf-8"),
            h5.h5r.DATASET_REGION,
            selection,
        ), True
    return replacement.ref, True


def _retarget(
    value: object, source: h5.Dataset, replacement: h5.Dataset
) -> tuple[object, bool]:
    if isinstance(value, h5.Reference):
        return _retarget_reference(value, source, replacement)
    if isinstance(value, np.void) and value.dtype.names:
        updated, changed = _retarget(np.asarray(value), source, replacement)
        if isinstance(updated, np.ndarray):
            return updated[()], changed
    if not isinstance(value, np.ndarray) or not _has_references(value.dtype):
        return value, False
    result = value.copy()
    changed = False
    if value.dtype.names:
        for name in value.dtype.names:
            updated, field_changed = _retarget(value[name], source, replacement)
            if field_changed:
                result[name] = updated
                changed = True
    else:
        for index in np.ndindex(value.shape):
            updated, item_changed = _retarget(value[index], source, replacement)
            if item_changed:
                result[index] = updated
                changed = True
    return result, changed


def retarget_references(source: h5.Dataset, replacement: h5.Dataset) -> None:
    """Point references to source at its same-file, same-shape replacement.

    Both datasets must remain linked and open during this call. Object and region
    references in local attributes/datasets retain their selections, including
    reference arrays and compound/vlen fields. Null and unrelated references are
    unchanged. External/soft link targets are not traversed. The caller unlinks
    source and moves replacement to the canonical path after this returns.
    """

    def update_node(_name: str, node: h5.Group | h5.Dataset) -> None:
        for name, value in node.attrs.items():
            updated, changed = _retarget(value, source, replacement)
            if changed:
                node.attrs.modify(name, updated)
        if isinstance(node, h5.Dataset) and _has_references(node.dtype):
            updated, changed = _retarget(node[()], source, replacement)
            if changed:
                node[()] = updated

    update_node("/", source.file)
    source.file.visititems(update_node)
