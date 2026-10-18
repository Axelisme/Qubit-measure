"""Own per-workflow parsed Start drafts and current attachment panel replacement.

Controller owns accepted-run knobs and their fixed capture. Window supplies that
input when switching/reloading; this coordinator never reads Controller, source,
hardware or storage. It retains inactive parsed drafts in RAM, swaps attachments
only after both candidate panels build, and delegates compatibility to FormState.
The memento stays a single current plan/knobs snapshot, not a run. Local unparsed renderer text is not an inactive draft.
"""

from dataclasses import dataclass

from qtpy.QtWidgets import QWidget

from zcu_tools.gui.app.workflows.panel import WorkflowFormPanel, WorkflowPlanPanel
from zcu_tools.gui.app.workflows.persistence import (
    WorkflowInputRestoreReport,
    WorkflowInputSnapshot,
)
from zcu_tools.gui.form import FormEnv, FormInput, FormModel, FormState

from .attachment import Attachment
from .catalog import WorkflowCatalog


@dataclass(frozen=True)
class WorkflowDraft:
    """Detached parsed RAM drafts, separate from persistent strict envelopes.

    plan and knobs are authored FormInputs with their original declarations and
    independent selection overrides. Reload/switching may migrate their leaves
    to new declarations and report discarded paths. No renderer text is saved.
    """

    plan: FormInput[FormModel]
    knobs: FormInput[FormModel]


class WorkflowForms[C]:
    """Current catalog/attachment, its two Start panels, and inactive RAM drafts.

    catalog is the current source pairing; attachment is the selected workflow.
    plan/knobs are its rendered parsed drafts in a fixed FormEnv. Switching keeps
    one detached memento per name, then restores only that name's inputs using its
    current independent models. Explicit Prepare replaces selected defaults with
    Window's new capture. Reload replaces current declarations but preserves
    captures and compatible drafts. No operation, run or accepted Host is owned.
    """

    def __init__(
        self, catalog: WorkflowCatalog[C], env: FormEnv, parent: QWidget
    ) -> None:
        """Build source-free/partial default panels for catalog's first attachment.

        Use on QApplication owner; source/editor failures propagate without a
        fallback. No context/device capture or native execution occurs.
        """
        self.catalog = catalog
        self.attachment = catalog.select()
        self._drafts: dict[str, WorkflowDraft] = {}
        self.plan, self.knobs, _ = self.build_panels(self.attachment, env, parent)

    def build_panels(
        self,
        attachment: Attachment[C],
        env: FormEnv,
        parent: QWidget,
        snapshot: WorkflowDraft | None = None,
    ) -> tuple[
        WorkflowPlanPanel, WorkflowFormPanel[FormModel], WorkflowInputRestoreReport
    ]:
        """Construct candidate defaults and migrate compatible RAM draft leaves.

        Declaration changes may discard raw leaves or selection overrides;
        return all discarded paths without running converters or options hooks.

        env is fixed by Window. snapshot None means fresh defaults. Defaults,
        source and compatibility errors propagate; unpublished panels are closed
        and detached on failure. Return both candidates and dropped paths without
        changing current attachment, drafts, input or lifecycle.
        """
        plan = attachment.plan_panel(env, parent)
        knobs = None
        try:
            state = FormState[FormModel](attachment.knobs_model.default_input(env))
            knobs = attachment.knobs_panel(state, env, parent)
            report = WorkflowInputRestoreReport((), ())
            if snapshot is not None:
                report = WorkflowInputRestoreReport(
                    plan.state.migrate_input(snapshot.plan, env),
                    knobs.state.migrate_input(snapshot.knobs, env),
                )
            for panel in (plan, knobs):
                panel.state.reset(panel.state.input)
                panel.editor.refresh(env)
            return plan, knobs, report
        except Exception:
            plan.close()
            plan.deleteLater()
            if knobs is not None:
                knobs.close()
                knobs.deleteLater()
            raise

    def select(
        self, name: str, snapshot: WorkflowDraft, parent: QWidget
    ) -> WorkflowInputRestoreReport:
        """Save current parsed drafts and select name with the same fixed capture.

        Window checks idle admission and supplies its authoritative current knobs
        input in snapshot. Unknown names raise before mutation. Inactive drafts
        migrate by current models, returning discarded leaves/overrides for
        display. Candidate errors leave current panels intact.
        Window owns removal/close of the old views after this succeeds.
        """
        attachment = self.catalog.select(name)
        plan, knobs, report = self.build_panels(
            attachment, self.plan.env, parent, self._drafts.get(name)
        )
        self._drafts[self.attachment.name] = snapshot
        self.attachment = attachment
        self.plan, self.knobs = plan, knobs
        return report

    def prepare(self, env: FormEnv, parent: QWidget) -> None:
        """Replace selected drafts with fresh defaults in Window's explicit capture.

        Previous selected parsed edits and local text are discarded on success;
        inactive workflow drafts remain. Failures preserve both current views.
        """
        plan, knobs, _ = self.build_panels(self.attachment, env, parent)
        self.plan, self.knobs = plan, knobs

    def reload(
        self,
        catalog: WorkflowCatalog[C],
        snapshot: WorkflowDraft,
        parent: QWidget,
    ) -> WorkflowInputRestoreReport:
        """Replace current models with compatible drafts, retaining fixed capture.

        Window owns source loading, admission and failed-reload Start blocking.
        Retain a still-declared name; otherwise select the new catalog's default.
        Build both panels before publication; candidate failures keep old drafts.
        Removed/incompatible leaves and overrides may be discarded and are
        returned for display. Inactive RAM drafts migrate lazily against new
        models when selected. No converter or options hook runs in migration.
        """
        name = self.attachment.name if self.attachment.name in catalog.names else None
        attachment = catalog.select(name)
        plan, knobs, report = self.build_panels(
            attachment, self.plan.env, parent, snapshot
        )
        self.catalog, self.attachment = catalog, attachment
        self.plan, self.knobs = plan, knobs
        return report

    def restore(self, snapshot: WorkflowInputSnapshot) -> WorkflowInputRestoreReport:
        """Restore both complete drafts after validating both trusted identities.

        Window checks idle admission. Raw expressions, missing/null and invalid
        ordinary values remain authored input. A malformed or incompatible
        envelope raises ValueError before either draft changes. No catalog
        selection, converter, evaluation or source capture occurs.
        """
        plan, knobs = snapshot.inputs(
            self.attachment.plan_model, self.attachment.knobs_model, self.plan.env
        )
        report = WorkflowInputRestoreReport((), ())
        self.plan.state.reset(plan)
        self.knobs.state.reset(knobs)
        for panel in (self.plan, self.knobs):
            panel.editor.refresh(panel.env)
            panel.editor.discard_local_text()
        return report
