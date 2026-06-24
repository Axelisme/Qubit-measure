"""Path / transport helpers for experiment data files.

The dict-based ``save_data`` / ``load_data`` / ``save_local_data`` /
``load_local_data`` layer is gone (ADR-0027): persistence is now native
``labber_io`` via ``PersistableExperiment``. This module retains the
filesystem-path helpers (datafolder layout, extension normalization, unique
filenames), the remote upload/download transport, and ``load_comment`` (a thin
read over ``labber_io``).
"""

from __future__ import annotations

import os
import warnings
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
        return filepath.replace(".h5", ".hdf5")
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
    if filepath.endswith(".h5"):
        return filepath.replace(".h5", "")
    if filepath.endswith(".hdf5"):
        return filepath.replace(".hdf5", "")
    return filepath


def safe_labber_filepath(filepath: str) -> str:
    """
    Ensure a unique file path by appending a numeric suffix if the file already exists.

    Args:
        filepath (str): The initial file path.

    Returns:
        str: A unique file path with a numeric suffix if necessary.
    """
    filepath = os.path.abspath(filepath)

    filepath = format_ext(filepath)

    def parse_filepath(filepath: str) -> tuple[str, int, str]:
        filename, ext = os.path.splitext(filepath)
        count = filename.split("_")[-1]
        if count.isdigit():
            filename = "_".join(filename.split("_")[:-1])  # remove number
            return filename, int(count), ext
        else:
            return filename, 1, ext

    filename, count, ext = parse_filepath(filepath)

    filepath = filename + f"_{count}" + ext
    while os.path.exists(filepath):
        count += 1
        filepath = filename + f"_{count}" + ext

    return filepath


def load_comment(filepath: str) -> str | None:
    """Return the file's comment string, or None if it cannot be read.

    Backed by ``labber_io.load_labber_data`` (the comment is also available as
    the ``.comment`` attribute of a loaded ``LabberData``).
    """
    from .labber_io import load_labber_data

    try:
        return load_labber_data(format_ext(filepath)).comment
    except Exception as e:
        warnings.warn(f"Failed to load comment from {filepath}: {e}")
        return None


def upload_to_server(filepath: str, server_ip: str, port: int) -> bool:
    """
    Upload a file to a remote server.

    Args:
        filepath (str): The path to the file to upload.
        server_ip (str): The IP address of the server.
        port (int): The port number of the server.

    Returns:
        bool: True if upload succeeded, False otherwise.
    """
    import requests

    filepath = os.path.abspath(filepath)
    url = f"http://{server_ip}:{port}/upload"
    try:
        with open(filepath, "rb") as file:
            files = {"file": (filepath, file)}
            response = requests.post(url, files=files)
        print(response.text)
        return response.status_code == 200
    except Exception as e:
        print(f"Upload failed: {e}")
        return False


def download_from_server(filepath: str, server_ip: str, port: int) -> None:
    """
    Download a file from a remote server.

    Args:
        filepath (str): The path where the downloaded file will be saved.
        server_ip (str): The IP address of the server.
        port (int): The port number of the server.
    """
    import requests

    url = f"http://{server_ip}:{port}/download"
    response = requests.post(url, json={"path": filepath})
    assert response.status_code == 200, f"Fail to download file: {response.text}"

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as file:
        file.write(response.content)

    print(f"Download file to {filepath}")
