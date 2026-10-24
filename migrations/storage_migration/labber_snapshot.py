"""Read stored Labber instrument attributes without accessing any instruments."""

import math
from pathlib import Path

import h5py as h5
import numpy as np
from pydantic import JsonValue
from zcu_tools.datafile import JsonObject


class LabberSnapshotError(ValueError):
    """Stored instrument settings cannot be represented by the snapshot codec.

    The message locates the unsupported group, attribute or value. Callers may
    retain the original Labber file and omit only the snapshot. File access and
    HDF5 I/O errors are not translated into this exception.
    """


def load_labber_device_snapshot(source: Path) -> dict[str, JsonObject] | None:
    """Read Instrument config attributes from an exact Labber file, read-only.

    Return instrument group names mapped to their original attribute names and
    values. Missing Instrument config returns None; an empty group returns {}.
    No device drivers, live state, units or instrument aliases are inferred.
    Bytes decode strictly as UTF-8. NumPy scalars and arrays become JSON scalars
    and lists (empty arrays become []). Complex values become {real, imag}
    objects, recursively inside arrays. Nonfinite real values, including complex
    components, become {nonfinite: 'nan' | 'inf' | '-inf'} objects.
    Unsupported types, invalid UTF-8, nested groups or datasets raise the located
    LabberSnapshotError, not a partial snapshot. Original values remain in the
    source file. Ordinary file/HDF5 I/O errors propagate unchanged.
    """
    with h5.File(source, "r") as file:
        if "Instrument config" not in file:
            return None
        root = file["Instrument config"]
        if not isinstance(root, h5.Group):
            raise LabberSnapshotError(f"{source}/Instrument config: expected group")
        snapshot: dict[str, JsonObject] = {}
        for name, instrument in root.items():
            location = f"{source}/Instrument config/{name}"
            if not isinstance(instrument, h5.Group) or len(instrument):
                raise LabberSnapshotError(f"{location}: expected attribute-only group")
            snapshot[name] = {
                key: _json_value(value, f"{location}/{key}")
                for key, value in instrument.attrs.items()
            }
        return snapshot


def _json_value(value: object, location: str) -> JsonValue:
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError as error:
            raise LabberSnapshotError(f"{location}: invalid UTF-8") from error
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(
        value, (int, float, complex, np.integer, np.floating, np.complexfloating)
    ):
        return _json_number(value)
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist(), location)
    if isinstance(value, list):
        return [
            _json_value(item, f"{location}[{index}]")
            for index, item in enumerate(value)
        ]
    raise LabberSnapshotError(f"{location}: unsupported {type(value).__name__}")


def _json_number(
    value: int | float | complex | np.integer | np.floating | np.complexfloating,
) -> JsonValue:
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (complex, np.complexfloating)):
        return {"real": _json_number(value.real), "imag": _json_number(value.imag)}
    number = float(value)
    if math.isfinite(number):
        return number
    return {
        "nonfinite": "nan" if math.isnan(number) else "inf" if number > 0 else "-inf"
    }
