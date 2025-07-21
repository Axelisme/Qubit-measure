import sys
import traceback


def print_traceback() -> None:
    """
    印出當前的異常追蹤訊息。如果異常包含 `_pyroTraceback`，則印出該追蹤訊息。
    """
    err_msg = sys.exc_info()[1]
    if err_msg is None:
        return
    if hasattr(err_msg, "_pyroTraceback"):
        pyro_traceback = getattr(err_msg, "_pyroTraceback", None)
        if isinstance(pyro_traceback, list):
            print("".join(pyro_traceback))
    else:
        print(traceback.format_exc())
