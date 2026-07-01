"""Shared handler typing and render-view access."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

from zcu_tools.gui.remote.errors import ErrorCode, RemoteError

if TYPE_CHECKING:
    from zcu_tools.gui.app.main.controller import RenderView

    from ..service import RemoteControlAdapter

Handler = Callable[["RemoteControlAdapter", Mapping[str, object]], Mapping[str, object]]


def render_view(adapter: RemoteControlAdapter) -> RenderView:
    """Return the canvas View's pure-read surface or fail in headless mode."""
    rv = adapter.render_view
    if rv is None:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            "no render view attached (headless process)",
        )
    return rv
