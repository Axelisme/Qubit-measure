"""GuiEvent → change-buffer (category, object_id) mapping.

The per-connection change buffer (see ``service.py``) folds the fine-grained
``GuiEvent`` stream into a few coarse *categories*, each tagged with the
*object* it affected (a tab_id / device name / context label) when the payload
carries one. The buffer drives two things:

- a per-RPC-reply notification summary (``gui_changes``), and
- the stale-operation guard (run / commit), which matches a category+object
  against the operation's dependency.

Keeping this table Qt-free and exhaustive (``assert_exhaustive``) means a new
``GuiEvent`` cannot silently escape the buffer — it must be categorised or
explicitly ignored.
"""

from __future__ import annotations

from typing import Callable, Optional

from zcu_tools.gui.event_bus import GuiEvent, Payload

# ---------------------------------------------------------------------------
# Categories (coarse, agent-facing). Object id is category-specific:
#   cfg_edited       -> editor_id (set by the editor change stream, not here)
#   tab_changed      -> tab_id
#   context_changed  -> None (global to the active context)
#   device_changed   -> device name (or None)
#   soc_changed      -> None
#   run_changed      -> tab_id (the affected tab, when known)
#   predictor_changed-> None
# ---------------------------------------------------------------------------

CAT_CFG_EDITED = "cfg_edited"
CAT_TAB_CHANGED = "tab_changed"
CAT_CONTEXT_CHANGED = "context_changed"
CAT_DEVICE_CHANGED = "device_changed"
CAT_SOC_CHANGED = "soc_changed"
CAT_RUN_CHANGED = "run_changed"
CAT_PREDICTOR_CHANGED = "predictor_changed"


def _tab_id(payload: Payload) -> Optional[str]:
    return getattr(payload, "tab_id", None)


def _device_name(payload: Payload) -> Optional[str]:
    return getattr(payload, "name", None)


def _none(payload: Payload) -> Optional[str]:
    del payload
    return None


# GuiEvent -> (category, object extractor). ``None`` value = intentionally
# ignored (not buffered). Every GuiEvent must appear here.
_EVENT_CATEGORY: dict[
    GuiEvent, Optional[tuple[str, Callable[[Payload], Optional[str]]]]
] = {
    GuiEvent.MD_CHANGED: (CAT_CONTEXT_CHANGED, _none),
    GuiEvent.ML_CHANGED: (CAT_CONTEXT_CHANGED, _none),
    GuiEvent.CONTEXT_SWITCHED: (CAT_CONTEXT_CHANGED, _none),
    GuiEvent.SOC_CHANGED: (CAT_SOC_CHANGED, _none),
    GuiEvent.TAB_ADDED: (CAT_TAB_CHANGED, _tab_id),
    GuiEvent.TAB_CLOSED: (CAT_TAB_CHANGED, _tab_id),
    GuiEvent.TAB_CONTENT_CHANGED: (CAT_RUN_CHANGED, _tab_id),
    GuiEvent.TAB_INTERACTION_CHANGED: (CAT_RUN_CHANGED, _tab_id),
    GuiEvent.RUN_LOCK_CHANGED: (CAT_RUN_CHANGED, _tab_id),
    GuiEvent.PREDICTOR_CHANGED: (CAT_PREDICTOR_CHANGED, _none),
    GuiEvent.DEVICE_CHANGED: (CAT_DEVICE_CHANGED, _device_name),
    GuiEvent.DEVICE_SETUP_CHANGED: (CAT_DEVICE_CHANGED, _none),
}


def assert_exhaustive() -> None:
    """Fail fast if a GuiEvent has no category mapping (run at import)."""
    missing = [e for e in GuiEvent if e not in _EVENT_CATEGORY]
    if missing:
        raise RuntimeError(
            f"change_categories missing mapping for GuiEvent(s): {missing}"
        )


def category_for(
    event: GuiEvent, payload: Payload
) -> Optional[tuple[str, Optional[str]]]:
    """Return ``(category, object_id)`` for an event, or ``None`` if ignored."""
    entry = _EVENT_CATEGORY.get(event)
    if entry is None:
        return None
    category, extract = entry
    return category, extract(payload)


assert_exhaustive()
