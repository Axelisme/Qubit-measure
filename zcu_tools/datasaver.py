import os
from datetime import datetime


def create_datafolder(root_dir: str, prefix: str = "") -> str:
    root_dir = os.path.abspath(os.path.join(root_dir, "Database"))
    yy, mm, dd = datetime.today().strftime("%Y-%m-%d").split("-")
    save_dir = os.path.join(root_dir, prefix, f"{yy}/{mm}/Data_{mm}{dd}")
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def make_comment(cfg: dict, append: str = "") -> str:
    # pretty convert cfg to string
    import json

    comment = json.dumps(cfg, indent=2)
    comment += "\n" + append

    return comment


def safe_labber_filepath(filepath: str):
    if not filepath.endswith(".hdf5") or not filepath.endswith(".h5"):
        filepath += ".hdf5"  # labber save data as hdf5

    filepath = os.path.abspath(filepath)

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

    if filepath.endswith(".hdf5"):
        filepath = filepath[:-5]  # remove .hdf5

    return filepath


def save_data_local(
    filepath: str,
    x_info: dict,
    z_info: dict,
    y_info: dict = None,
    comment=None,
    tag=None,
):
    zdata = z_info["values"]
    z_info.update({"complex": True, "vector": False})
    log_channels = [z_info]
    step_channels = list(filter(None, [x_info, y_info]))

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
            os.rmdir(f"{yy}/{mm}/Data_{mm}{dd}")
            os.rmdir(f"{yy}/{mm}")
            os.rmdir(yy)
    except OSError as e:
        print("Fail to remove empty folder: ", e)
        pass


def load_data_local(file_path):
    import h5py

    with h5py.File(file_path, "r") as file:
        data = file["Data"]["Data"]
        if data.shape[2] == 1:  # 1D data,
            x_data = data[:, 0, 0][:]
            y_data = None
            z_data = data[:, 1, 0][:] + 1j * data[:, 2, 0][:]
        else:
            x_data = data[:, 0, 0][:]
            y_data = data[0, 1, :][:]
            z_data = data[:, 2, :][:] + 1j * data[:, 3, :][:]
    return z_data, x_data, y_data


def upload_file2server(filepath: str, server_ip: str, port: int):
    import requests

    filepath = os.path.abspath(filepath)
    url = f"http://{server_ip}:{port}/upload"
    with open(filepath, "rb") as file:
        files = {"file": (filepath, file)}
        response = requests.post(url, files=files)

    print(response.text)


def download_file2server(filepath: str, server_ip: str, port: int):
    import requests

    url = f"http://{server_ip}:{port}/download"
    response = requests.post(url, json={"path": filepath})
    assert response.status_code == 200, f"Fail to download file: {filepath}"

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as file:
        file.write(response.content)

    print(f"Download file to {filepath}")


def save_data(
    filepath: str,
    x_info: dict,
    z_info: dict,
    y_info: dict = None,
    comment=None,
    tag=None,
    server_ip: str = None,
    port: int = 4999,
):
    filepath = safe_labber_filepath(filepath)
    if server_ip is not None:
        save_data_local(filepath, x_info, z_info, y_info, comment, tag)
        if not filepath.endswith(".hdf5") or not filepath.endswith(".h5"):
            filepath += ".hdf5"
        upload_file2server(filepath, server_ip, port)
        os.remove(filepath)
    else:
        save_data_local(filepath, x_info, z_info, y_info, comment, tag)


def load_data(
    filepath: str,
    server_ip: str = None,
    port: int = 4999,
):
    if not filepath.endswith(".hdf5") and not filepath.endswith(".h5"):
        filepath += ".hdf5"
    if server_ip is not None:
        if not os.path.exists(filepath):
            download_file2server(filepath, server_ip, port)
    z_data, x_data, y_data = load_data_local(filepath)
    return z_data, x_data, y_data
