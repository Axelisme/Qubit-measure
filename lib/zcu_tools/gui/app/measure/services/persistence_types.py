"""Pure Qt appearance state owned by the measure application."""

from pydantic import BaseModel, ConfigDict, Field

from zcu_tools.gui.session.persistence import PersistenceError as PersistenceError
from zcu_tools.gui.session.state import (
    DEFAULT_LEFT_PANEL_WIDTH as DEFAULT_LEFT_PANEL_WIDTH,
)

APP_STATE_VERSION = 1


class AppPersistedState(BaseModel):
    """Window dimensions and tab panel width, in Qt logical pixels.

    version identifies the layout-only codec. All dimensions must be positive.
    This state owns only visual layout, never session resources or tab drafts.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    version: int = APP_STATE_VERSION
    window_width: int = Field(default=1280, gt=0)
    window_height: int = Field(default=750, gt=0)
    left_panel_width: int = Field(default=DEFAULT_LEFT_PANEL_WIDTH, gt=0)
