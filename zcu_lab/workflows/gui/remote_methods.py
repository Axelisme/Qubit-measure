"""Qt-free declarations for RPC submission and control of a T1 run."""

from zcu_tools.gui.remote.method_spec import MethodSpec
from zcu_tools.gui.remote.param_spec import JsonType, ParamSpec

WIRE_VERSION = 1
GUI_VERSION = 1

_RUN_ID = ParamSpec(
    "run_id",
    JsonType.STRING,
    description="Explicit native identity supplied to the GUI Start.",
)

METHOD_SPECS: dict[str, MethodSpec] = {
    "workflow.start": MethodSpec(
        5.0,
        "Submit partial Start input; native operation_id is pending, not acceptance.",
        params=(
            _RUN_ID,
            ParamSpec(
                "plan",
                JsonType.JSON,
                description='Required batch array of {"path": ["field", ...], "value": JSON}; [] uses defaults.',
            ),
            ParamSpec(
                "tunables",
                JsonType.JSON,
                description="Required path/value batch; expressions are allowed. [] uses defaults, not GUI draft.",
            ),
        ),
    ),
    "workflow.tunables": MethodSpec(
        5.0,
        "Submit current parsed input with external edits to run_id; return native revision.",
        params=(
            _RUN_ID,
            ParamSpec(
                "batch",
                JsonType.JSON,
                description='Required array of {"path": ["field", ...], "value": JSON}; [] submits current draft, not defaults.',
            ),
        ),
    ),
    "workflow.status": MethodSpec(
        5.0,
        "Read the attached run_id, operation_id and native status; pending status is null.",
    ),
    "workflow.records": MethodSpec(
        5.0,
        "Read committed native records with commit_seq > since; terminal data remains readable.",
        params=(
            _RUN_ID,
            ParamSpec(
                "since",
                JsonType.INTEGER,
                required=False,
                default=0,
                description="Nonnegative native commit_seq cursor; omitted/null means 0.",
            ),
        ),
    ),
    "workflow.figures": MethodSpec(
        5.0,
        "List names in the current run's live Plots without rendering.",
        params=(_RUN_ID,),
    ),
    "workflow.figure": MethodSpec(
        5.0,
        "Render one named Figure as base64 PNG on the GUI owner, at 640 x 480.",
        params=(
            _RUN_ID,
            ParamSpec(
                "name",
                JsonType.STRING,
                description="Exact name returned by workflow.figures.",
            ),
        ),
    ),
    "workflow.pause": MethodSpec(
        5.0,
        "Request cooperative Pause of run_id; reply is not segment completion.",
        params=(_RUN_ID,),
    ),
    "workflow.resume": MethodSpec(
        5.0,
        "Submit Resume of run_id; operation_id is a pending segment token.",
        params=(_RUN_ID,),
    ),
    "workflow.stop": MethodSpec(
        5.0,
        "Request cooperative Stop of run_id, including pending startup.",
        params=(_RUN_ID,),
    ),
}
