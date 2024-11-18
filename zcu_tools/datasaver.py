import os
from datetime import datetime

DRY_RUN = False


def create_datafolder(root_path: str) -> str:
    os.makedirs(root_path, exist_ok=True)
    yy, mm, dd = datetime.today().strftime("%Y-%m-%d").split("-")
    save_path = os.path.join(root_path, f"{yy}/{mm}/Data_{mm}{dd}")
    os.makedirs(save_path, exist_ok=True)
    return save_path


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

    if DRY_RUN:

        def cast_info(info):
            def convert2shape(v):
                if hasattr(v, "shape"):
                    if callable(v.shape):
                        return v.shape()
                    return v.shape
                return v

            return {k: convert2shape(v) for k, v in info.items()}

        print(f"DRY_RUN: Saving data to {filepath}")
        print(f"x_info: {cast_info(x_info)}")
        if y_info:
            print(f"y_info: {cast_info(y_info)}")
        print(f"z_info: {cast_info(z_info)}")
        print(f"comment: {comment}")
        print(f"tag: {tag}")
        return

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
