"""Detached RAM carriers for live workspace reload, without disk ownership."""

from pydantic import BaseModel, ConfigDict, JsonValue


class WorkspaceTab(BaseModel):
    """One reloadable tab: adapter registry name and authored FormInput JSON.

    form_input contains data and explicit selections, including invalid drafts.
    Sources, results and widgets are never included. The mutable JSON is detached
    from the live tab; callers own this RAM copy. No file format is defined here.
    """

    model_config = ConfigDict(frozen=True)
    adapter_name: str
    form_input: dict[str, JsonValue]


class WorkspaceSnapshot(BaseModel):
    """RAM-only ordered tabs and focused index for experiment-catalog reload.

    tabs contains detached authored drafts. active_tab_index is a zero-based
    index into tabs or None for no focus. This snapshot does not drive startup
    restoration or persistent storage.
    """

    model_config = ConfigDict(frozen=True)
    tabs: tuple[WorkspaceTab, ...] = ()
    active_tab_index: int | None = None
