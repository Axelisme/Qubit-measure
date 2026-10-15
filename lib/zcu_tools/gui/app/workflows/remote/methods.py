"""Qt-free declarations for RPC submission and control of a native workflow run."""

from zcu_tools.gui.remote.method_spec import MethodSpec
from zcu_tools.gui.remote.param_spec import JsonType, ParamSpec

# Form observations are lean; describe_form returns one layer from the exact
# call-local candidate. Authored input and its persistence format are unchanged.
WIRE_VERSION = 5
GUI_VERSION = 5

_RUN_ID = ParamSpec(
    "run_id",
    JsonType.STRING,
    description="Explicit native identity supplied to the GUI Start.",
)

_FORM_PARAMS = (
    ParamSpec(
        "form",
        JsonType.STRING,
        enum=("plan", "tunables"),
        description="Form to inspect; current run knobs use the same accepted binding.",
    ),
    ParamSpec(
        "run_id",
        JsonType.STRING,
        required=False,
        default=None,
        description="Required when a run is attached; omitted/null before Start.",
    ),
    ParamSpec(
        "batch",
        JsonType.JSON,
        required=False,
        default=None,
        description="Omitted/null reads GUI draft with local errors; a path/value array previews parsed input only, without writing.",
    ),
)

METHOD_SPECS: dict[str, MethodSpec] = {
    "workflow.start": MethodSpec(
        5.0,
        "Submit partial Start input; native operation_id is pending, not acceptance.",
        params=(
            _RUN_ID,
            ParamSpec(
                "name",
                JsonType.STRING,
                required=False,
                default=None,
                description=(
                    "Native workflow name: coherence_fluxdep, mist_fluxdep or overnight; "
                    "omitted/null selects coherence_fluxdep, not GUI draft selection."
                ),
            ),
            ParamSpec(
                "plan",
                JsonType.JSON,
                description='Required batch array of {"path": ["field", ...], "value": JSON}; [] uses defaults.',
            ),
            ParamSpec(
                "tunables",
                JsonType.JSON,
                description=(
                    "Required path/value batch; text fields allow expressions. "
                    "[] uses defaults, not GUI draft."
                ),
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
    "workflow.form": MethodSpec(
        5.0,
        "Read input/ordinary values/errors and native revision, or preview a batch "
        "without committing. Use workflow.describe_form for field metadata. "
        'Expressions {"expr": text} are accepted only where description.expression '
        "is true; selector, bool and library-key fields take plain values.",
        params=_FORM_PARAMS,
    ),
    "workflow.describe_form": MethodSpec(
        5.0,
        "Read single-layer metadata and input/ordinary values/errors for the same "
        "candidate in one owner turn. path=[] lists root fields; a field path "
        "returns its metadata and direct children only. Options carry values and "
        "commands, not branch fields. Commands are GUI actions with no workflow "
        "command RPC. selected is the core's first matching index "
        "or null. Full paths retain flattened wrappers. hidden/group/collapsed "
        "only affect presentation; constraints are hints, not validation. "
        "can_expand is structural and may expand to no children. selection_fixed "
        "means one non-null selector branch, not frozen inner choices. Null is "
        "enable/disable, not another branch, even while disabled. fixed/fixed_value "
        "fixes the complete value. Invalid paths return invalid_params with reason=form_path "
        "and data.form_path; declaration failures remain controller_error. "
        "Uses workflow.form's run target, fixed env and non-submitting batch rules.",
        params=(
            *_FORM_PARAMS,
            ParamSpec(
                "path",
                JsonType.ARRAY,
                description="Model-field names; [] is root. No container indices or mapping keys.",
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
