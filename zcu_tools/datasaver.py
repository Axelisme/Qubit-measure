import os
from datetime import datetime

import requests

from .tools import numpy2number


def create_datafolder(root_dir: str, prefix: str = "") -> str:
    root_dir = os.path.abspath(os.path.join(root_dir, "Database"))
    yy, mm, dd = datetime.today().strftime("%Y-%m-%d").split("-")
    save_dir = os.path.join(root_dir, prefix, f"{yy}/{mm}/Data_{mm}{dd}")
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def save_cfg(filepath: str, cfg: dict):
    import yaml

    if not filepath.endswith(".yaml"):
        filepath += ".yaml"

    cfg = numpy2number(cfg)

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)


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

    import labber_api.Labber as Labber  # type: ignore

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
        os.rmdir(f"{yy}/{mm}/Data_{mm}{dd}")
        os.rmdir(f"{yy}/{mm}")
        os.rmdir(yy)
    except OSError:
        print("Fail to remove empty folder")
        pass


def upload_file2server(filepath: str, server_ip: str, port: int):
    filepath = os.path.abspath(filepath)
    url = f"http://{server_ip}:{port}/upload"
    with open(filepath, "rb") as file:
        files = {"file": (filepath, file)}
        response = requests.post(url, files=files)

    print(response.text)


def make_comment(cfg: dict, append: str = "") -> str:
    # pretty convert cfg to string
    import json

    comment = json.dumps(cfg, indent=2)
    comment += "\n" + append

    return comment


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
    if server_ip is not None:
        save_data_local(filepath, x_info, z_info, y_info, comment, tag)
        filepath = filepath + ".hdf5"  # labber save data as hdf5
        upload_file2server(filepath, server_ip, port)
        os.remove(filepath)
    else:
        save_data_local(filepath, x_info, z_info, y_info, comment, tag)
