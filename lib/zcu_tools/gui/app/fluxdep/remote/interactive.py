"""Owner-turn interactive wire projection and commands over native app contexts.

Retain the requested context through synchronous GUI callbacks. Each reply uses
one committed snapshot for numbers and PNG; native analysis/plotting owns the
projection policy. No input read establishes a published-resource observation.
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
from zcu_tools.gui.remote.param_spec import build_input_schema
from zcu_tools.plotting.fluxdep.cross_selection import make_cross_selection_figure
from zcu_tools.plotting.fluxdep.onetone import make_onetone_pick_figure
from zcu_tools.plotting.fluxdep.pick import make_flux_pick_figure
from zcu_tools.plotting.fluxdep.twotone import make_twotone_pick_figure

from .dto import (
    CommandDeclaration,
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
    selection: bool,
    figure: Figure,
) -> ContextReplyBase:
    commands: list[CommandDeclaration] = []
    if not closed:
        commands = [
            {"name": command.name, "schema": build_input_schema(command.params)}
            for command in plugin.commands
        ]
        commands.extend(
            {"name": name, "schema": build_input_schema(())}
            for name in ("undo", "apply" if selection else "finish", "cancel")
        )
    return {
        "context_id": context_id,
        "closed": closed,
        "plugin": plugin.plugin_id,
        "commands": commands,
        "can_undo": not closed and session.can_undo(),
        "figure": _png(figure),
    }


def _closed(owner: FluxDepInteractiveOwner, context_id: int) -> bool:
    current = owner.inspect()
    return current is None or current.context_id != context_id


def _execute(
    owner: FluxDepInteractiveOwner,
    plugin: PluginDefinition[_S, _R],
    session: Session[_S],
    kind: str,
    command: str | None,
    params: Mapping[str, object],
) -> None:
    if command is None:
        return
    if command == "undo":
        session.undo()
    elif command == "cancel":
        owner.cancel()
    elif command == "apply":
        owner.apply_cross_selection()
    elif command == "finish":
        if kind == "line":
            owner.finish_line_pick()
        elif kind == "onetone":
            owner.finish_onetone_pick()
        else:
            owner.finish_twotone_pick()
    else:
        plugin.execute_command(session, command, params)


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
        _execute(owner, context.plugin, context.session, "line", command, values)
        state = context.session.snapshot()
        closed = _closed(owner, active.context_id) if command is not None else False
        wire = {
            **_base(
                active.context_id,
                context.plugin,
                context.session,
                closed=closed,
                selection=False,
                figure=make_flux_pick_figure(context.plugin.inputs, state),
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
        _execute(owner, context.plugin, context.session, "onetone", command, values)
        state_one = context.session.snapshot()
        closed = _closed(owner, active.context_id) if command is not None else False
        wire = {
            **_base(
                active.context_id,
                context.plugin,
                context.session,
                closed=closed,
                selection=False,
                figure=make_onetone_pick_figure(
                    context.plugin.inputs,
                    state_one,
                    flux_half=context.flux_half,
                    flux_int=context.flux_int,
                ),
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
        _execute(owner, context.plugin, context.session, "twotone", command, values)
        state_two = context.session.snapshot()
        previous_two = before_two if command is not None else None
        view_two = project_twotone_pick(
            context.plugin.inputs, state_two, previous=previous_two
        )
        closed = _closed(owner, active.context_id) if command is not None else False
        wire = {
            **_base(
                active.context_id,
                context.plugin,
                context.session,
                closed=closed,
                selection=False,
                figure=make_twotone_pick_figure(
                    context.plugin.inputs,
                    state_two,
                    previous=previous_two,
                    show_changes=True,
                    show_mask=True,
                ),
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
        _execute(owner, context.plugin, context.session, "selection", command, values)
        state_selection = context.session.snapshot()
        previous_selection = before_selection if command is not None else None
        view_selection = project_cross_selection(
            context.plugin.inputs, state_selection, previous=previous_selection
        )
        closed = _closed(owner, active.context_id) if command is not None else False
        wire = {
            **_base(
                active.context_id,
                context.plugin,
                context.session,
                closed=closed,
                selection=True,
                figure=make_cross_selection_figure(
                    context.plugin.inputs,
                    state_selection,
                    previous=previous_selection,
                    show_changes=True,
                ),
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
            },
        }
        changes = {
            "points_added": int(view_selection.added_points.shape[0]),
            "points_removed": int(view_selection.removed_points.shape[0]),
        }
    return {
        "context": wire,
        "effect": {"command": command, "closed": closed, "changes": changes}
        if command is not None
        else None,
    }


def h_interactive_read(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> InteractiveReply:
    """Inspect adapter's live input on the owner loop; params is unused.

    Return committed state/commands/PNG without opening input or switching view.
    Inactive returns null context/effect. Native projection/render errors propagate.
    """
    del params
    owner = adapter.ctrl.interactive
    active = owner.inspect()
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
    (line/onetone/twotone). Return identity/state/commands/PNG with null effect.
    Native owner rejects unknown/inactive/unaligned/wrong-type sources. A
    synchronous reaction retiring this input raises interactive_context_changed.
    Native projection/render errors propagate without rollback.
    """
    name, kind = params["name"], params["kind"]
    assert isinstance(name, str)  # Shared ParamSpec validated name and kind.
    owner = adapter.ctrl.interactive
    if kind == "line":
        context = owner.begin_line_pick(name)
    elif kind == "onetone":
        context = owner.begin_onetone_pick(name)
    else:
        context = owner.begin_twotone_pick(name)
    active = owner.inspect()
    if active is None or active.context is not context:
        raise FailedPreconditionError(
            "opened context retired during GUI reaction",
            reason_code="interactive_context_changed",
        )
    return _reply(owner, active)


def h_selection_interactive_open(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> InteractiveReply:
    """Open/reuse adapter's joint-cloud input on the owner loop; params unused.

    Return identity/state/commands/PNG with null effect. Native owner rejects
    unavailable cloud/disposed input. Synchronous retirement raises
    interactive_context_changed. Projection/render errors propagate; no rollback.
    Shared route, not this handler, enforces all-source observations.
    """
    del params
    owner = adapter.ctrl.interactive
    context = owner.begin_cross_selection()
    active = owner.inspect()
    if active is None or active.context is not context:
        raise FailedPreconditionError(
            "opened context retired during GUI reaction",
            reason_code="interactive_context_changed",
        )
    return _reply(owner, active)


def _command(
    adapter: RemoteControlAdapter, params: Mapping[str, object], *, selection: bool
) -> InteractiveReply:
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
    owner = adapter.ctrl.interactive
    active = owner.inspect()
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
        command_params
        or (selection and command == "finish")
        or (not selection and command == "apply")
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
    Return retained identity's state/PNG and effect, even after GUI replacement.
    ExpectedError categories/reasons and native failures propagate unchanged;
    publication/render failure does not imply rollback or reopened input.
    """
    return _command(adapter, params, selection=False)


def h_selection_interactive_command(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> InteractiveReply:
    """Command adapter's joint input on the owner loop after all-source guards.

    params has validated positive context_id, command and params object
    (default empty). Require current joint identity before mutation. Plugin
    validates domain commands; undo/apply/cancel reject nonempty params.
    Return retained identity's state/PNG/effect. Apply publishes without closing
    valid input/Undo. ExpectedError and native failures propagate unchanged;
    publication/render failure does not imply rollback.
    """
    return _command(adapter, params, selection=True)
