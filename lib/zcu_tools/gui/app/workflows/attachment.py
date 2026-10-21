"""Standalone presentation adapters over the measure headless registration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel

from zcu_tools.gui.app.measure.services.workflow_attachment import (
    Attachment as HeadlessAttachment,
)
from zcu_tools.gui.app.measure.services.workflow_attachment import (
    PreparedWorkflow as PreparedWorkflow,
)
from zcu_tools.gui.app.measure.services.workflow_attachment import (
    WorkflowAttachment as HeadlessWorkflowAttachment,
)
from zcu_tools.gui.form import FormEnv, FormModel, FormState

if TYPE_CHECKING:
    from qtpy.QtWidgets import QWidget

    from .panel import WorkflowFormPanel, WorkflowPlanPanel


class Attachment[C](HeadlessAttachment[C], Protocol):
    """Headless declaration plus temporary standalone panel builders."""

    def plan_panel(self, env: FormEnv, parent: QWidget | None) -> WorkflowPlanPanel:
        """Build the standalone plan editor in the caller's fixed capture."""
        ...

    def knobs_panel(
        self,
        state: FormState[FormModel],
        env: FormEnv,
        parent: QWidget | None,
    ) -> WorkflowFormPanel[FormModel]:
        """Render the shared draft without creating another owning form state."""
        ...


class WorkflowAttachment[
    P: BaseModel,
    T: BaseModel,
    S,
    R,
    FP: FormModel,
    FT: FormModel,
    C,
](HeadlessWorkflowAttachment[P, T, S, R, FP, FT, C]):
    """Keep standalone widgets separate from native conversion and submission.

    All native lowering and run binding live in the measure headless owner.
    The standalone host retains only these panel factories until its removal.
    """

    def plan_panel(self, env: FormEnv, parent: QWidget | None) -> WorkflowPlanPanel:
        """Build defaults and preview from the same headless plan converter."""
        from .panel import WorkflowPlanPanel

        return WorkflowPlanPanel(self.plan_model, self._plan, env, parent)

    def knobs_panel(
        self,
        state: FormState[FormModel],
        env: FormEnv,
        parent: QWidget | None,
    ) -> WorkflowFormPanel[FormModel]:
        """Render the unique draft; closing detaches editor subscriptions only."""
        from .panel import WorkflowFormPanel

        return WorkflowFormPanel(state, env, parent)
