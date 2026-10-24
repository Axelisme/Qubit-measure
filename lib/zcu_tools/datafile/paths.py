"""Path helpers for experiment data files.

The dict-based ``save_data`` / ``load_data`` / ``save_local_data`` /
``load_local_data`` layer is gone (ADR-0063). This module retains the
filesystem-path helpers: extension normalization and caller path reservation.
"""

from __future__ import annotations

import os


def format_ext(filepath: str) -> str:
    """
    Format the file extension to .hdf5 if not already present.

    Args:
        filepath (str): The file path to format.

    Returns:
        str: The formatted file path with .hdf5 extension.
    """
    if filepath.endswith(".h5"):
        return filepath[: -len(".h5")] + ".hdf5"
    if filepath.endswith(".hdf5"):
        return filepath
    return filepath + ".hdf5"


def reserve_labber_filepath(filepath: str) -> str:
    """
    Return a caller-owned unique Labber filepath.

    Args:
        filepath (str): The initial file path.

    Returns:
        str: A unique file path with a numeric suffix.
    """
    filepath = os.path.abspath(filepath)

    filepath = format_ext(filepath)

    def parse_filepath(filepath: str) -> tuple[str, int, str, bool]:
        filename, ext = os.path.splitext(filepath)
        prefix, separator, suffix = filename.rpartition("_")
        if not separator or not suffix.isdigit():
            return filename, 0, ext, False

        sequence_has_root = os.path.exists(prefix + ext) or os.path.exists(
            f"{prefix}_1{ext}"
        )
        if sequence_has_root:
            return prefix, int(suffix), ext, True
        return filename, 0, ext, True

    filename, count, ext, has_numeric_suffix = parse_filepath(filepath)

    if count > 0:
        filepath = filename + f"_{count}" + ext
    elif has_numeric_suffix:
        filepath = filename + ext
    else:
        count = 1
        filepath = filename + f"_{count}" + ext
    while os.path.exists(filepath):
        count = 1 if count == 0 else count + 1
        filepath = filename + f"_{count}" + ext

    return filepath
