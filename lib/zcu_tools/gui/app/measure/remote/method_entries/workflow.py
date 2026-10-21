"""Fixed workflow namespace, independent of view mode and selected tab."""

from zcu_tools.gui.remote.method_spec import MethodSpec
from zcu_tools.gui.remote.param_spec import JsonType, ParamSpec

from ._params import default_integer, optional_string, required_string
from ._registry import AgentMethodPolicy, RemoteMethodEntry, method_entry

_TAB = required_string(
    "tab_id", "Opened workflow owner, independent of current selection"
)
_RUN = required_string(
    "run_id", "Native workflow run identity, not an operation handle"
)
_FORM = ParamSpec("form", JsonType.STRING, enum=("plan", "tunables"))
_BATCH = ParamSpec("batch", JsonType.JSON, required=False)
_REQUIRED_BATCH = ParamSpec("batch", JsonType.JSON)
_FORM_PARAMS = (_TAB, _FORM, _BATCH)


def _entry(
    name: str,
    description: str,
    params: tuple[ParamSpec, ...],
    *,
    readonly: bool = False,
    operation: bool = False,
) -> RemoteMethodEntry:
    return method_entry(
        f"workflow.{name}",
        f"workflow:h_workflow_{name}",
        MethodSpec(30.0, description, params, readonly=readonly),
        agent=AgentMethodPolicy(
            exposure="tool",
            tool_names=(f"workflow_{name}",),
            operation_key="workflow:{run_id}" if operation else None,
        ),
    )


METHODS: tuple[RemoteMethodEntry, ...] = (
    _entry(
        "open",
        "Open independent headless drafts for an explicit workflow name. Returns tab_id/name/tab snapshot; no focus or toggle change.",
        (required_string("name"),),
    ),
    _entry(
        "close",
        "Close an idle workflow owner. Returns closed=false while busy; never cancels, drains or disconnects.",
        (_TAB,),
    ),
    _entry(
        "status",
        "Read a native run snapshot, or catalog names and all opened tabs when run_id is omitted/null. Does not render figures.",
        (optional_string("run_id"),),
        readonly=True,
    ),
    _entry(
        "start",
        "Submit workflow Start under the shared RUN lease. Omitted/null batches use shared parsed drafts; explicit batches, including [], use fresh captured defaults. Returns pending operation_id, not native acceptance. No retry.",
        (
            _TAB,
            _RUN,
            ParamSpec("plan", JsonType.JSON, required=False),
            ParamSpec("tunables", JsonType.JSON, required=False),
        ),
        operation=True,
    ),
    _entry(
        "pause",
        "Request cooperative Pause, retaining the operation and RUN lease until resume/terminal delivery.",
        (_RUN,),
    ),
    _entry(
        "resume",
        "Resume a delivered paused run in the worker under the same RUN lease and operation.",
        (_RUN,),
        operation=True,
    ),
    _entry(
        "stop",
        "Request Stop. Completion and lease release wait for terminal owner delivery.",
        (_RUN,),
    ),
    _entry(
        "form",
        "Read authored input, ordinary values and Form errors. Optional batch previews current parsed draft without committing or lowering.",
        _FORM_PARAMS,
        readonly=True,
    ),
    _entry(
        "describe_form",
        "Read the same candidate plus single-layer metadata at typed path; [] is root. No schema tree or mutation.",
        (*_FORM_PARAMS, ParamSpec("path", JsonType.JSON)),
        readonly=True,
    ),
    _entry(
        "edit_form",
        "Edit the owner's shared draft and return its observation. Busy/structure/readonly rejection belongs to the owner; invalid model values remain editable.",
        (_TAB, _FORM, _REQUIRED_BATCH),
    ),
    _entry(
        "tunables",
        "Apply path/value batch to the run's shared tunables draft. [] submits current draft, never defaults. Returns native revision/values.",
        (_RUN, _REQUIRED_BATCH),
    ),
    _entry(
        "records",
        "Read native committed records with commit_seq strictly greater than since; terminal records remain readable.",
        (_RUN, default_integer("since", 0)),
        readonly=True,
    ),
    _entry(
        "figures",
        "List retained native figure names without rendering, including after terminal delivery.",
        (_RUN,),
        readonly=True,
    ),
    _entry(
        "figure",
        "Render an exact retained figure as png_b64 on the owner; MCP validates and delivers image content instead of base64 text.",
        (_RUN, required_string("name")),
        readonly=True,
    ),
    _entry(
        "answer",
        "Submit an answer matching workflow.status pending_question.answer_schema. Returns the original pending operation_id; the worker retains the same RUN lease through terminal owner delivery. Invalid answers preserve the paused question.",
        (
            _RUN,
            ParamSpec("answer", JsonType.JSON, required=False, omit_when_missing=True),
        ),
        operation=True,
    ),
)
