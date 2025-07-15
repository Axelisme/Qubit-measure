import json
import os
from datetime import datetime
from typing import Optional, Tuple

import numpy as np

from zcu_tools.config import config

KEYWORD = "Database"


def create_datafolder(root_dir: str, prefix: str = "") -> str:
    """
    Create a data folder structure based on the current date.

    Args:
        root_dir (str): The root directory where the data folder will be created.
        prefix (str, optional): An optional prefix for the folder structure. Defaults to "".

    Returns:
        str: The absolute path to the created data folder.
    """
    root_dir = os.path.abspath(os.path.join(root_dir, KEYWORD))
    yy, mm, dd = datetime.today().strftime("%Y-%m-%d").split("-")
    save_dir = os.path.join(root_dir, prefix, os.path.join(yy, mm, f"Data_{mm}{dd}"))
    if not config.DATA_DRY_RUN:
        os.makedirs(save_dir, exist_ok=True)
    return save_dir


def make_comment(cfg: dict, prepend: str = "") -> str:
    """
    Generate a formatted comment string from a configuration dictionary.

    Args:
        cfg (dict): Configuration dictionary to be converted to a string.
        prepend (str, optional): Additional string to prepend to the comment. Defaults to "".

    Returns:
        str: A formatted comment string.
    """
    # pretty convert cfg to string
    return prepend + "\n" + json.dumps(cfg, indent=2)


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


def safe_labber_filepath(filepath: str):
    """
    Ensure a unique file path by appending a numeric suffix if the file already exists.

    Args:
        filepath (str): The initial file path.

    Returns:
        str: A unique file path with a numeric suffix if necessary.
    """
    filepath = os.path.abspath(filepath)

    filepath = format_ext(filepath)

    def parse_filepath(filepath):
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
    x_info: dict,
    z_info: dict,
    y_info: Optional[dict] = None,
    comment: Optional[str] = None,
    tag: Optional[str] = None,
):
    """
    Save data locally in a Labber-compatible format.

    Args:
        filepath (str): The file path where the data will be saved.
        x_info (dict): Information about the x-axis data.
        z_info (dict): Information about the z-axis data.
        y_info (Optional[dict], optional): Information about the y-axis data. Defaults to None.
        comment (Optional[str], optional): A comment to include in the saved file. Defaults to None.
        tag (Optional[str], optional): Tags to include in the saved file. Defaults to None.
    """
    zdata = z_info["values"]
    z_info.update({"complex": True, "vector": False})
    log_channels = [z_info]
    step_channels = list(filter(None, [x_info, y_info]))

    filepath = remove_ext(filepath)  # because labber will add .hdf5 automatically

    if config.DATA_DRY_RUN:
        print("DRY RUN: Save data to ", filepath)
        return

    import Labber  # type: ignore

    fObj = Labber.createLogFile_ForData(filepath, log_channels, step_channels)
    if y_info:
        for trace in zdata:
            fObj.addEntry({z_info["name"]: trace})
    else:
        fObj.addEntry({z_info["name"]: zdata})

    if comment:
        fObj.setComment(comment)
    if tag:
        fObj.setTags(tag)

    # remove labber tmp directory
    # check path yyyy/mm/Data_mmdd/ is empty
    # if so, remove the empty empty folder
    try:
        yy, mm, dd = datetime.today().strftime("%Y-%m-%d").split("-")
        if os.path.exists(yy):
            os.rmdir(os.path.join(yy, mm, f"Data_{mm}{dd}"))
            os.rmdir(os.path.join(yy, mm))
            os.rmdir(yy)
    except OSError as e:
        print("Fail to remove empty folder: ", e)
        pass


def load_local_data(
    filepath: str,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Load data from a local HDF5 file.

    Args:
        file_path (str): The path to the HDF5 file.

    Returns:
        Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]: The loaded z, x, and y data arrays.
    """
    import h5py

    filepath = format_ext(filepath)

    if config.DATA_DRY_RUN:
        print("DRY RUN: Load data from ", filepath)
        return np.array([]), np.array([]), None

    def parser_data(
        data: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        if data.shape[2] == 1:  # 1D data,
            x_data = data[:, 0, 0][:]
            y_data = None
            z_data = data[:, -2, 0][:] + 1j * data[:, -1, 0][:]
        else:  # 2D data
            x_data = data[:, 0, 0][:]
            y_data = data[0, 1, :][:]
            z_data = data[:, -2, :][:] + 1j * data[:, -1, :][:]

        return z_data, x_data, y_data

    with h5py.File(filepath, "r") as file:
        data = np.array(file["Data"]["Data"])  # type: ignore
        z_data, x_data, y_data = parser_data(data)

        assert isinstance(x_data, np.ndarray)
        assert isinstance(z_data, np.ndarray)

        if "Log_2" in file:
            z_data = [z_data]

            def check_log_valid(
                z_i: np.ndarray, x_i: np.ndarray, y_i: np.ndarray
            ) -> None:
                if not x_data.shape == x_i.shape:
                    raise ValueError("x data shape mismatch")
                if y_i is not None:
                    if y_data is None:
                        raise ValueError("y data is None")
                    if not y_data.shape == y_i.shape:
                        raise ValueError("y data shape mismatch")

                if not np.allclose(x_data, x_i) or not np.allclose(y_data, y_i):
                    raise ValueError("Find different x or y data in log data")

            i = 2
            while f"Log_{i}" in file:
                log_data = np.array(file[f"Log_{i}"]["Data"]["Data"])  # type: ignore
                z_data_i, x_i, y_i = parser_data(log_data)
                check_log_valid(z_data_i, x_i, y_i)

                z_data.append(z_data_i)
                i += 1
            z_data = np.array(z_data)

    return z_data, x_data, y_data


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

    if config.DATA_DRY_RUN:
        print(f"DRY RUN: Upload {filepath} to {server_ip}:{port}")
        return True

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


def download_from_server(filepath: str, server_ip: str, port: int):
    """
    Download a file from a remote server.

    Args:
        filepath (str): The path where the downloaded file will be saved.
        server_ip (str): The IP address of the server.
        port (int): The port number of the server.
    """
    import requests

    if config.DATA_DRY_RUN:
        print(f"DRY RUN: Download {filepath} from {server_ip}:{port}")
        return

    url = f"http://{server_ip}:{port}/download"
    response = requests.post(url, json={"path": filepath})
    assert response.status_code == 200, f"Fail to download file: {response.text}"

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as file:
        file.write(response.content)

    print(f"Download file to {filepath}")


def save_data(
    filepath: str,
    x_info: dict,
    z_info: dict,
    y_info: Optional[dict] = None,
    comment: Optional[str] = None,
    tag: Optional[str] = None,
    server_ip: Optional[str] = None,
    port: int = 4999,
):
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


def load_data(
    filepath: str,
    server_ip: Optional[str] = None,
    port: int = 4999,
):
    """
    Load data either locally or from a remote server.

    Args:
        filepath (str): The path to the file to load.
        server_ip (Optional[str], optional): The IP address of the server. Defaults to None.
        port (int, optional): The port number of the server. Defaults to 4999.

    Returns:
        Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]: The loaded z, x, and y data arrays.
    """
    if server_ip is not None:
        if not os.path.exists(filepath):
            download_from_server(filepath, server_ip, port)
    z_data, x_data, y_data = load_local_data(filepath)
    return z_data, x_data, y_data
