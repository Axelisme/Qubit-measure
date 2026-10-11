"""Qt-free contract for the connection identity initialization and control RPCs."""

from zcu_tools.gui.remote.method_spec import MethodSpec
from zcu_tools.gui.remote.param_spec import JsonType, ParamSpec

CONTROL_METHOD_SPECS: dict[str, MethodSpec] = {
    "control.connect": MethodSpec(
        5.0,
        "Initialize fixed socket identity; owner auto-acquires unlocked Local.",
        params=(
            ParamSpec("label", JsonType.STRING, required=False, default="agent"),
            ParamSpec(
                "mode",
                JsonType.STRING,
                required=False,
                default="owner",
                enum=("owner", "readonly"),
            ),
        ),
        readonly=True,
    ),
    "control.status": MethodSpec(
        5.0,
        "Read holder, lock, online presence and whether this socket holds control.",
        readonly=True,
    ),
    "control.acquire": MethodSpec(
        5.0, "Acquire unlocked Local or an offline remote holder.", readonly=True
    ),
    "control.release": MethodSpec(
        5.0,
        "Return control to unlocked Local; requires an owner-mode holder.",
        readonly=True,
    ),
}
