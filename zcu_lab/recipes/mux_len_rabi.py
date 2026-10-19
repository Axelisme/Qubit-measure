"""One bounded mux recipe; numeric acquisition remains with its native owner."""

from zcu_tools.mcp.measure.execution_reply import (
    SummaryElements,
    SummaryEstimate,
    SummaryEstimateRows,
    SummaryParameter,
)
from zcu_tools.mcp.measure.recipe import (
    RecipeDefinition,
    RecipeGenerator,
    RecipeSession,
)

from .mux_helpers import (
    DRIVE_PARAMETERS,
    ECHO_PI_PARAMETERS,
    INPUTS,
    MuxInline,
    MuxLibrary,
    MuxOverride,
    finish_mux,
    prepare_mux,
)


def mux_len_rabi(  # noqa: PLR0913 - Typed keywords are the declared recipe inputs.
    session: RecipeSession,
    *,
    target: str | None = None,
    reuse_tab_id: str | None = None,
    participants: list[str] | None = None,
    library: list[MuxLibrary] | None = None,
    inline: list[MuxInline] | None = None,
    overrides: list[MuxOverride] | None = None,
) -> RecipeGenerator:
    """Run mux length Rabi; review candidates before explicit Apply.

    session is driver-supplied. target binds the Reset mux alias; None retains
    global aliases. reuse_tab_id resets an idle mux/len_rabi tab.
    participants is an ordered name list; None keeps Reset defaults. library
    selects typed path/key pairs, then inline copies typed selector paths, then
    overrides sends one batch of typed scalar path/value edits. Caller order is
    preserved; list indexes are full original positions, never participant rows.
    Copy ref ancestors outer-first before editing children. None values retain
    input; expressions remain authored. No range/calibration inference occurs.
    Paths use MHz/us/native gain, matching the Form; count leaves use n.
    Shared reps/rounds/recovery and delay axes remain shared, not per-tone.
    All elements require equal sweep counts; Len Rabi supports const/flat_top only.
    Defaults report retained Form inputs, not their internal generation history.
    Run saves raw and Primary; cancellation stops later steps. Finish-early
    follows owner partial-data policy. Only an accepted writeback answer applies;
    skipped writes none. Invalid inputs/cfg and native/transport failures propagate
    at their author step; successful copies are not rolled back on later failure.
    """
    tab = prepare_mux(
        session,
        "mux/len_rabi",
        target=target,
        reuse_tab_id=reuse_tab_id,
        participants=participants,
        library=library,
        inline=inline,
        overrides=overrides,
    )
    yield from finish_mux(tab)


DEFINITION = RecipeDefinition(
    name="mux_len_rabi",
    description="Run mux length Rabi, save raw and analysis, and ask before Apply.",
    input_schema=INPUTS,
    run=mux_len_rabi,
    adapter_name="mux/len_rabi",
    summary_parameters=(
        SummaryElements(
            "tones",
            ("tones",),
            DRIVE_PARAMETERS
            + (
                SummaryParameter("start", ("sweep", "start"), "us"),
                SummaryParameter("stop", ("sweep", "stop"), "us"),
                SummaryParameter("points", ("sweep", "n")),
            ),
            source="default",
        ),
        SummaryParameter("participants", ("participants",)),
        SummaryParameter("reps", ("reps",), None),
        SummaryParameter("rounds", ("rounds",), None),
        SummaryParameter("relax_delay", ("relax_delay",), "us"),
    ),
    summary_estimates=(
        SummaryEstimateRows(
            "tones",
            ("tones",),
            (
                SummaryEstimate(
                    "pi_len",
                    ("pi_len",),
                    ("pi_len_err",),
                    "us",
                    element_key=("element",),
                ),
                SummaryEstimate(
                    "pi2_len",
                    ("pi2_len",),
                    ("pi2_len_err",),
                    "us",
                    element_key=("element",),
                ),
                SummaryEstimate(
                    "rabi_f",
                    ("rabi_f",),
                    ("rabi_f_err",),
                    "MHz",
                    element_key=("element",),
                ),
            ),
        ),
    ),
)
