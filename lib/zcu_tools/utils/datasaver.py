from __future__ import annotations

import os
import warnings
from datetime import datetime
from typing import Any, Literal, overload

import numpy as np
from numpy.typing import NDArray


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


def save_local_data(
    filepath: str,
    x_info: dict[str, Any],
    z_info: dict[str, Any],
    y_info: dict[str, Any] | None = None,
    comment: str | None = None,
    tag: str | None = None,
) -> None:
    """
    Save data locally in a Labber-compatible format.

    Backed by ``labber_io.save_labber_data`` (pure h5py/numpy, no ``Labber``
    dependency). This is a thin compatibility wrapper around the dict-based
    ``x_info``/``z_info``/``y_info`` API; for new code prefer calling
    ``labber_io.save_labber_data`` / ``LabberData.save`` directly.

    Args:
        filepath (str): The file path where the data will be saved.
        x_info (dict): Inner (x) axis, keys ``name``, ``unit``, ``values``.
        z_info (dict): Complex log channel, keys ``name``, ``unit``, ``values``.
            ``values`` is shaped ``(Nx,)`` for 1D or ``(Ny, Nx)`` for 2D
            (one trace per y value).
        y_info (Optional[dict], optional): Outer (y) axis, same keys as x_info.
            Defaults to None (1D data).
        comment (Optional[str], optional): A comment to include in the saved file. Defaults to None.
        tag (Optional[str], optional): Tags to include in the saved file. Defaults to None.
    """
    from .labber_io import save_labber_data

    zdata = np.asarray(z_info["values"], dtype=complex)
    # zdata is (Nx,) for 1D, or (Ny, Nx) for 2D (one trace per y value) --
    # exactly labber_io's z convention (inner x-axis last).
    # axes are (name, unit, values), inner axis (x) first.
    axes = [(x_info.get("name", "x"), x_info.get("unit", ""), x_info["values"])]
    if y_info is not None:
        axes.append((y_info.get("name", "y"), y_info.get("unit", ""), y_info["values"]))
    save_labber_data(
        format_ext(filepath),
        z=(z_info.get("name", "S21"), z_info.get("unit", ""), zdata),
        axes=axes,
        comment=comment or "",
        tags=tag,
    )


def load_local_data(
    filepath: str,
) -> tuple[NDArray[np.complex128], NDArray[np.float64], NDArray[np.float64] | None]:
    """
    Load data from a local HDF5 file.

    Backed by ``labber_io.load_labber_data`` (pure h5py/numpy, no ``Labber``
    dependency). Thin compatibility wrapper that preserves this function's
    historical *frequency-major* return shape; for new code prefer
    ``labber_io.load_labber_data`` / ``LabberData.load`` (which return a
    ``LabberData`` with ``.data``/``.axes`` and ``z`` in ``(Ny, Nx)`` order).

    Args:
        filepath (str): The path to the HDF5 file (``.hdf5`` / ``.h5`` optional).

    Returns:
        tuple[NDArray[np.complex128], NDArray[np.float64], Optional[NDArray[np.float64]]]:
            ``(z, x, y)``. ``z`` is ``(Nx,)`` for 1D, ``(Nx, Ny)`` for 2D, and
            ``(Nlog, Nx, Ny)`` for a file with stacked log configs; ``y`` is
            ``None`` for 1D data.
    """
    from .labber_io import load_labber_data

    d = load_labber_data(format_ext(filepath))
    x_data = np.asarray(d.x, dtype=np.float64)
    y_data = None if d.y is None else np.asarray(d.y, dtype=np.float64)

    # labber_io returns z as (Nx,) for 1D, (Ny, Nx) for 2D and (Nlog, Ny, Nx)
    # for stacked logs.  This function's historical contract is frequency-major:
    # (Nx,) for 1D, (Nx, Ny) for 2D, (Nlog, Nx, Ny) for stacked -- so transpose
    # the inner two axes back.
    z = np.asarray(d.z)
    if z.ndim == 2:  # (Ny, Nx) -> (Nx, Ny)
        z_data = z.T
    elif z.ndim == 3:  # (Nlog, Ny, Nx) -> (Nlog, Nx, Ny)
        z_data = np.transpose(z, (0, 2, 1))
    else:  # 1D (Nx,) unchanged
        z_data = z

    return z_data, x_data, y_data


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


def save_data(
    filepath: str,
    x_info: dict[str, Any],
    z_info: dict[str, Any],
    y_info: dict[str, Any] | None = None,
    comment: str | None = None,
    tag: str | None = None,
    server_ip: str | None = None,
    port: int = 4999,
) -> None:
    """
    Save data either locally or to a remote server.

    Args:
        filepath (str): The file path where the data will be saved.
        x_info (dict): Information about the x-axis data.
        z_info (dict): Information about the z-axis data.
        y_info (Optional[dict], optional): Information about the y-axis data. Defaults to None.
        comment (Optional[str], optional): A comment to include in the saved file. Defaults to None.
        tag (Optional[str], optional): Tags to include in the saved file. Defaults to None.
        server_ip (Optional[str], optional): The IP address of the server. Defaults to None.
        port (int, optional): The port number of the server. Defaults to 4999.
    """
    filepath = safe_labber_filepath(filepath)
    if server_ip is not None:
        save_local_data(filepath, x_info, z_info, y_info, comment, tag)
        success = upload_to_server(filepath, server_ip, port)
        if success:
            os.remove(filepath)
        else:
            print(
                f"Failed to upload {filepath} to server {server_ip}:{port}, file not deleted."
            )
    else:
        save_local_data(filepath, x_info, z_info, y_info, comment, tag)
    print("Successfully saved data to ", filepath)


@overload
def load_data(
    filepath: str,
    *,
    server_ip: str | None = None,
    port: int = 4999,
    return_comment: Literal[True],
) -> tuple[
    NDArray[np.complex128],
    NDArray[np.float64],
    NDArray[np.float64] | None,
    str | None,
]: ...


@overload
def load_data(
    filepath: str,
    *,
    server_ip: str | None = None,
    port: int = 4999,
    return_comment: Literal[False],
) -> tuple[NDArray[np.complex128], NDArray[np.float64], NDArray[np.float64] | None]: ...


def load_data(
    filepath: str,
    *,
    server_ip: str | None = None,
    port: int = 4999,
    return_comment: bool = False,
):
    """
    Load data either locally or from a remote server.

    Args:
        filepath (str): The path to the file to load.
        server_ip (Optional[str], optional): The IP address of the server. Defaults to None.
        port (int, optional): The port number of the server. Defaults to 4999.

    Returns:
        tuple[NDArray[np.complex128], NDArray[np.float64], Optional[NDArray[np.float64]]]: The loaded z, x, and y data arrays.
    """
    if server_ip is not None:
        if not os.path.exists(filepath):
            download_from_server(filepath, server_ip, port)
    z_data, x_data, y_data = load_local_data(filepath)
    if return_comment:
        comment = load_comment(filepath)
        return z_data, x_data, y_data, comment

    return z_data, x_data, y_data
