"""Owner-turn interactive wire projection and commands over native app contexts.

Retain the requested context through synchronous GUI callbacks. Each reply uses
one committed snapshot for numbers; dedicated readonly rendering uses its
current committed snapshot. Native analysis/plotting owns projection policy. No input read establishes a published-resource observation.
"""

from __future__ import annotations

import base64
from collections.abc import Mapping
from typing import TYPE_CHECKING, TypeVar

import numpy as np
from matplotlib.figure import Figure

from zcu_tools.analysis.fluxdep.cross_selection import project_cross_selection
from zcu_tools.analysis.fluxdep.twotone import project_twotone_pick
from zcu_tools.gui.app.fluxdep.interactive import (
    ActiveInteractiveContext,
    CrossSelectionContext,
    FluxDepInteractiveOwner,
    LinePickContext,
    OneTonePickContext,
    TwoTonePickContext,
)
from zcu_tools.gui.expected_error import FailedPreconditionError, InvalidInputError
from zcu_tools.gui.interactive import PluginDefinition, Session
from zcu_tools.gui.plotting.figure_export import render_figure_png
from zcu_tools.plotting.fluxdep.cross_selection import make_cross_selection_figure
from zcu_tools.plotting.fluxdep.onetone import make_onetone_pick_figure
from zcu_tools.plotting.fluxdep.pick import make_flux_pick_figure
from zcu_tools.plotting.fluxdep.twotone import make_twotone_pick_figure

from .dto import (
    ContextReplyBase,
    InteractiveContextReply,
    InteractiveReply,
    LineChanges,
    OneToneChanges,
    PngImage,
    SelectionChanges,
    TwoToneChanges,
)

if TYPE_CHECKING:
    from .service import RemoteControlAdapter

_S = TypeVar("_S")
_R = TypeVar("_R")


def _png(figure: Figure) -> PngImage:
    data = render_figure_png(figure)
    return {"png_b64": base64.b64encode(data).decode("ascii"), "bytes": len(data)}


def _base(
    context_id: int,
    plugin: PluginDefinition[_S, _R],
    session: Session[_S],
    *,
    closed: bool,
) -> ContextReplyBase:
    return {
        "context_id": context_id,
        "closed": closed,
        "plugin": plugin.plugin_id,
        "can_undo": not closed and session.can_undo(),
    }


def _closed(owner: FluxDepInteractiveOwner, active: ActiveInteractiveContext) -> bool:
    current = (
        owner.inspect_selection()
        if active.spectrum_name is None
        else owner.inspect(active.spectrum_name)
    )
    return current is None or current.context_id != active.context_id


def _execute(
    owner: FluxDepInteractiveOwner,
    plugin: PluginDefinition[_S, _R],
    session: Session[_S],
    active: ActiveInteractiveContext,
    command: str | None,
    params: Mapping[str, object],
) -> list[str] | None:
    kind, name = active.kind, active.spectrum_name
    if command is None:
        return None
    if command == "undo":
        session.undo()
    elif command == "cancel":
        if name is None:
            owner.cancel_selection()
        else:
            owner.cancel(name)
    elif command == "apply":
        owner.apply_cross_selection()
    elif command == "finish":
        if kind == "line":
            return owner.finish_line_pick(name).invalidated
        if kind == "onetone":
            return owner.finish_onetone_pick(name).invalidated
        if kind == "selection":
            return owner.finish_cross_selection().invalidated
        return owner.finish_twotone_pick(name).invalidated
    else:
        plugin.execute_command(session, command, params)
    return None


def _reply(
    owner: FluxDepInteractiveOwner,
    active: ActiveInteractiveContext,
    command: str | None = None,
    params: Mapping[str, object] | None = None,
) -> InteractiveReply:
    context = active.context
    values = params if params is not None else {}
    wire: InteractiveContextReply
    changes: LineChanges | OneToneChanges | TwoToneChanges | SelectionChanges
    # Narrow context before reading its strongly typed Session. The retained
    # reference, not owner.inspect()'s successor, supplies the after snapshot.
    if isinstance(context, LinePickContext):
        before = context.session.snapshot()
        invalidated = _execute(
            owner,
            context.plugin,
            context.session,
            active,
            command,
            values,
        )
        state = context.session.snapshot()
        closed = _closed(owner, active) if command is not None else False
        wire = {
            **_base(
                active.context_id,
                context.plugin,
                context.session,
                closed=closed,
            ),
            "kind": "line",
            "spectrum_name": context.spectrum_name,
            "info": context.plugin.info(),
            "state": {
                "flux_half": state.flux_half,
                "flux_int": state.flux_int,
                "conjugate": state.conjugate,
                "magnitude_only": state.magnitude_only,
            },
        }
        changes = {
            "flux_half_before": before.flux_half,
            "flux_half_after": state.flux_half,
            "flux_int_before": before.flux_int,
            "flux_int_after": state.flux_int,
            "conjugate_before": before.conjugate,
            "conjugate_after": state.conjugate,
        }
    elif isinstance(context, OneTonePickContext):
        before_one = context.session.snapshot()
        invalidated = _execute(
            owner,
            context.plugin,
            context.session,
            active,
            command,
            values,
        )
        state_one = context.session.snapshot()
        closed = _closed(owner, active) if command is not None else False
        wire = {
            **_base(
                active.context_id,
                context.plugin,
                context.session,
                closed=closed,
            ),
            "kind": "onetone",
            "spectrum_name": context.spectrum_name,
            "info": {},
            "state": {
                "threshold": state_one.threshold,
                "peak_indices": list(state_one.peak_indices),
            },
        }
        changes = {
            "threshold_before": before_one.threshold,
            "threshold_after": state_one.threshold,
            "peaks_added": len(
                set(state_one.peak_indices) - set(before_one.peak_indices)
            ),
            "peaks_removed": len(
                set(before_one.peak_indices) - set(state_one.peak_indices)
            ),
        }
    elif isinstance(context, TwoTonePickContext):
        before_two = context.session.snapshot()
        invalidated = _execute(
            owner,
            context.plugin,
            context.session,
            active,
            command,
            values,
        )
        state_two = context.session.snapshot()
        previous_two = before_two if command is not None else None
        view_two = project_twotone_pick(
            context.plugin.inputs, state_two, previous=previous_two
        )
        closed = _closed(owner, active) if command is not None else False
        wire = {
            **_base(
                active.context_id,
                context.plugin,
                context.session,
                closed=closed,
            ),
            "kind": "twotone",
            "spectrum_name": context.spectrum_name,
            "info": {},
            "state": {
                "threshold": state_two.threshold,
                "sigma": state_two.sigma,
                "smooth_method": state_two.smooth_method,
                "width": state_two.width,
                "mode": state_two.mode,
                "mask_shape": list(state_two.mask.shape),
                "masked_count": int(np.count_nonzero(state_two.mask)),
                "point_count": int(view_two.result.dev_values.size),
            },
        }
        changes = {
            "mask_added": view_two.mask_added,
            "mask_removed": view_two.mask_removed,
            "points_added": int(view_two.added_points.shape[0]),
            "points_removed": int(view_two.removed_points.shape[0]),
        }
    else:
        before_selection = context.session.snapshot()
        invalidated = _execute(
            owner,
            context.plugin,
            context.session,
            active,
            command,
            values,
        )
        state_selection = context.session.snapshot()
        previous_selection = before_selection if command is not None else None
        view_selection = project_cross_selection(
            context.plugin.inputs, state_selection, previous=previous_selection
        )
        per_spectrum: dict[str, list[int]] = {}
        offset = 0
        # Retained source sizes include zero-point spectra and preserve identities.
        for name, total in context.source_point_counts:
            count = int(
                np.count_nonzero(
                    view_selection.result.selected[offset : offset + total]
                )
            )
            per_spectrum[name] = [count, total]
            offset += total
        closed = _closed(owner, active) if command is not None else False
        wire = {
            **_base(
                active.context_id,
                context.plugin,
                context.session,
                closed=closed,
            ),
            "kind": "selection",
            "spectrum_name": None,
            "info": {},
            "state": {
                "min_distance": state_selection.min_distance,
                "width": state_selection.width,
                "mode": state_selection.mode,
                "selected": state_selection.selected.tolist(),
                "selected_count": int(np.count_nonzero(view_selection.result.selected)),
                "per_spectrum": per_spectrum,
            },
        }
        changes = {
            "points_added": int(view_selection.added_points.shape[0]),
            "points_removed": int(view_selection.removed_points.shape[0]),
        }
    reply: InteractiveReply = {
        "context": wire,
        "effect": {"command": command, "closed": closed, "changes": changes}
        if command is not None
        else None,
    }
    if invalidated is not None:
        reply["invalidated"] = invalidated
    return reply


def _read_target(
    owner: FluxDepInteractiveOwner, params: Mapping[str, object]
) -> ActiveInteractiveContext | None:
    """Use explicit wire target; never infer it from name, display or identity."""
    if params["target"] == "selection":
        return owner.inspect_selection()
    name = params.get("name")
    if not isinstance(name, str) or not name:
        raise InvalidInputError("spectrum target requires a nonempty name")
    return owner.inspect(name)


def h_interactive_read(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> InteractiveReply:
    """Read a named draft on the owner loop without opening input or switching view.

    params requires target=spectrum with literal nonempty name, or
    target=selection without name. Invalid envelopes raise INVALID_PARAMS.
    Optional context_id refuses cancelled, removed, replaced or wrong-target
    input with interactive_context_changed. Unbound reads return null
    context/effect when absent. Native failures propagate.
    """
    owner = adapter.ctrl.interactive
    active = _read_target(owner, params)
    context_id = params.get("context_id")
    if context_id is not None and (active is None or active.context_id != context_id):
        raise FailedPreconditionError(
            "interactive identity or target changed",
            reason_code="interactive_context_changed",
        )
    return (
        _reply(owner, active)
        if active is not None
        else {"context": None, "effect": None}
    )


def h_spectrum_interactive_open(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> InteractiveReply:
    """Open/reuse adapter's picker on the owner loop.

    params is ParamSpec-validated name (nonempty literal) and kind
    (line/sample/onetone/twotone). sample delegates type selection to the owner;
    the reply reports the actual native kind. Return identity/state with null
    effect; discarded reports
    a replaced unfinished align/sample draft.
    Native owner rejects unknown/unaligned/wrong-type sources. A
    synchronous reaction retiring this input raises interactive_context_changed.
    Native projection errors propagate without rollback.
    """
    name, kind = params["name"], params["kind"]
    assert isinstance(name, str)  # Shared ParamSpec validated name and kind.
    owner = adapter.ctrl.interactive
    previous = owner.inspect(name)
    if kind == "line":
        context = owner.begin_line_pick(name)
    elif kind == "sample":
        context = owner.begin_sample_pick(name)
    elif kind == "onetone":
        context = owner.begin_onetone_pick(name)
    else:
        context = owner.begin_twotone_pick(name)
    active = owner.inspect(name)
    if active is None or active.context is not context:
        raise FailedPreconditionError(
            "opened context retired during GUI reaction",
            reason_code="interactive_context_changed",
        )
    reply = _reply(owner, active)
    if previous is not None and previous.context is not context:
        reply["discarded"] = "align" if previous.kind == "line" else "sample"
    return reply


def h_selection_interactive_open(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> InteractiveReply:
    """Open/reuse adapter's joint-cloud input on the owner loop; params unused.

    Return identity/state with null effect. Native owner rejects
    unavailable cloud/disposed input. Synchronous retirement raises
    interactive_context_changed. Projection errors propagate; no rollback.
    Shared route, not this handler, enforces all-source observations.
    """
    del params
    owner = adapter.ctrl.interactive
    context = owner.begin_cross_selection()
    active = owner.inspect_selection()
    if active is None or active.context is not context:
        raise FailedPreconditionError(
            "opened context retired during GUI reaction",
            reason_code="interactive_context_changed",
        )
    return _reply(owner, active)


def _command_envelope(
    params: Mapping[str, object],
) -> tuple[int, str, dict[str, object]]:
    """Validate the native envelope; plugin remains domain parameter authority."""
    context_id, command, values = (
        params["context_id"],
        params["command"],
        params["params"],
    )
    # ParamSpec already checked primitive types. This layer owns only envelope
    # positivity/identity and reserved verbs; plugin owns all domain parameters.
    assert isinstance(context_id, int)
    assert isinstance(command, str)
    assert isinstance(values, dict)
    if context_id <= 0:
        raise InvalidInputError(
            "context_id must be positive", reason_code="invalid_interactive_command"
        )
    if any(not isinstance(key, str) for key in values):
        raise InvalidInputError(
            "params keys must be strings", reason_code="invalid_interactive_command"
        )
    command_params: dict[str, object] = {key: value for key, value in values.items()}
    return context_id, command, command_params


def _command(
    adapter: RemoteControlAdapter, params: Mapping[str, object], *, selection: bool
) -> InteractiveReply:
    context_id, command, command_params = _command_envelope(params)
    owner = adapter.ctrl.interactive
    if selection:
        active = owner.inspect_selection()
    else:
        name = params["name"]
        assert isinstance(name, str)
        active = owner.inspect(name)
    if active is None:
        raise FailedPreconditionError(
            "no live interactive context", reason_code="no_interactive_context"
        )
    if (
        active.context_id != context_id
        or isinstance(active.context, CrossSelectionContext) != selection
        or (not selection and active.spectrum_name != params["name"])
    ):
        raise FailedPreconditionError(
            "interactive identity or target changed",
            reason_code="interactive_context_changed",
        )
    if command in ("undo", "finish", "apply", "cancel") and (
        command_params or (not selection and command == "apply")
    ):
        raise InvalidInputError(
            "reserved command has invalid params or target kind",
            reason_code="invalid_interactive_command",
        )
    return _reply(owner, active, command, command_params)


def h_spectrum_interactive_command(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> InteractiveReply:
    """Command adapter's picker on the owner loop after shared source guards.

    params has validated name, positive context_id, command and params object
    (default empty). Require current identity/target before mutation. Plugin
    validates domain commands; undo/finish/cancel reject nonempty params.
    Return retained identity's state and effect, even after GUI replacement.
    ExpectedError categories/reasons and native failures propagate unchanged;
    publication failure does not imply rollback or reopened input.
    """
    return _command(adapter, params, selection=False)


def h_selection_interactive_command(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> InteractiveReply:
    """Command adapter's joint input on the owner loop after all-source guards.

    params has validated positive context_id, command and params object
    (default empty). Require current joint identity before mutation. Plugin
    validates domain commands; undo/finish/apply/cancel reject nonempty params.
    Finish publishes and closes the global draft.
    Return retained identity's state/effect. Apply publishes without closing
    valid input/Undo. ExpectedError and native failures propagate unchanged;
    publication failure does not imply rollback.
    """
    return _command(adapter, params, selection=True)


def h_interactive_render(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> PngImage:
    """Render explicit target's current draft, without editing input or display.

    params requires target=spectrum plus literal nonempty name, or
    target=selection without name. Invalid envelopes raise INVALID_PARAMS.
    Missing draft raises no_interactive_context; native rendering errors
    propagate. Never falls back to published data or another target.
    """
    active = _read_target(adapter.ctrl.interactive, params)
    if active is None:
        raise FailedPreconditionError(
            "no live interactive context", reason_code="no_interactive_context"
        )
    context = active.context
    if isinstance(context, LinePickContext):
        spec_type = adapter.ctrl.state.get_spectrum(context.spectrum_name).spec_type
        figure = make_flux_pick_figure(
            context.plugin.inputs,
            context.session.snapshot(),
            onetone=spec_type == "OneTone",
        )
    elif isinstance(context, OneTonePickContext):
        figure = make_onetone_pick_figure(
            context.plugin.inputs,
            context.session.snapshot(),
            flux_half=context.flux_half,
            flux_int=context.flux_int,
        )
    elif isinstance(context, TwoTonePickContext):
        figure = make_twotone_pick_figure(
            context.plugin.inputs, context.session.snapshot(), show_mask=True
        )
    else:
        figure = make_cross_selection_figure(
            context.plugin.inputs, context.session.snapshot()
        )
    return _png(figure)
