"""Path helpers for experiment data files.

The dict-based ``save_data`` / ``load_data`` / ``save_local_data`` /
``load_local_data`` layer is gone (ADR-0027). This module retains the
filesystem-path helpers: datafolder layout, extension normalization, and caller
path reservation.
"""

from __future__ import annotations

import os
from datetime import datetime


def get_datafolder_path(
    database_dir: str,
    name: str = "",
    now: datetime | None = None,
) -> str:
    """Return today's data-folder path without creating it."""
    database_dir = os.path.abspath(database_dir)
    timestamp = datetime.today() if now is None else now
    yy, mm, dd = timestamp.strftime("%Y-%m-%d").split("-")
    return os.path.join(database_dir, name, os.path.join(yy, mm, f"Data_{mm}{dd}"))


def create_datafolder(database_dir: str, name: str = "") -> str:
    save_dir = get_datafolder_path(database_dir, name)
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


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


def remove_ext(filepath: str) -> str:
    """
    Remove the file extension from a file path.

    Args:
        filepath (str): The file path to format.

    Returns:
        str: The file path without the extension.
    """
    if filepath.endswith(".hdf5"):
        return filepath[: -len(".hdf5")]
    if filepath.endswith(".h5"):
        return filepath[: -len(".h5")]
    return filepath


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
