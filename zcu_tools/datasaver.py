import os
from datetime import datetime

from .tools import numpy2number


def create_datafolder(root_path: str) -> str:
    os.makedirs(root_path, exist_ok=True)
    yy, mm, dd = datetime.today().strftime("%Y-%m-%d").split("-")
    save_path = os.path.join(root_path, f"{yy}/{mm}/Data_{mm}{dd}")
    os.makedirs(save_path, exist_ok=True)
    return save_path


def save_cfg(filepath: str, cfg: dict):
    import yaml

    if not filepath.endswith(".yaml"):
        filepath += ".yaml"

    cfg = numpy2number(cfg)

    with open(filepath, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)


def save_data(
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
    # check filepath is Database/abcd/ef/Data_wxyz/xxx format
    # if so, remove the empty abdc/ef/Data_wxyz folder
    if "Database" in filepath:
        try:
            leafpath = os.path.dirname(filepath)
            leafpath = leafpath.split("Database/")[1]
            if os.path.exists(leafpath):
                for _ in range(3):
                    os.rmdir(leafpath)
                    leafpath = os.path.dirname(leafpath)
        except OSError:
            pass
