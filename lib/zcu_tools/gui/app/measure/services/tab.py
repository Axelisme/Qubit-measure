from __future__ import annotations

import logging
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.gui.app.measure.adapter.analyze_params import describe_analyze_params
from zcu_tools.gui.app.measure.state import Session, TabInteractionState
from zcu_tools.gui.app.measure.writeback import WritebackItem
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.form import FormPath, FormPreview
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.gui.session.types import ParameterContext

from ..adapter import AdapterGuide, AnalysisMode
from ..writeback import WritebackDraft
from .plot_lifecycle import release_retired_plots
from .ports import (
    AnalysisPaneSnapshot,
    PostAnalysisPaneSnapshot,
    RunPaneSnapshot,
    SavePaneSnapshot,
    TabSnapshot,
)
from .tab_input import StartupTab, TabInput
from .workspace_types import WorkspaceTab

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from zcu_tools.gui.app.measure.registry import Registry
    from zcu_tools.gui.app.measure.state import State
    from zcu_tools.resources.entry import PointStorageBinding

    from .ports import WritebackLifecyclePort
    from .tab_form import TabFormResources, TabFormSources


# Characters allowed verbatim in a tab-id slug; everything else (notably the
# adapter '/') collapses to '-'. The slug is cosmetic — the 8-hex suffix carries
# uniqueness — so a human/agent reads 'twotone-freq-1a2b3c4d' instead of a bare
# UUID while the id stays an opaque string key.
_SLUG_OK = set("abcdefghijklmnopqrstuvwxyz0123456789")


def _slug(name: str) -> str:
    out = "".join(c if c in _SLUG_OK else "-" for c in name.lower())
    # Collapse runs of '-' and trim, so 'twotone/rabi/amp_rabi' -> 'twotone-rabi-amp-rabi'.
    parts = [p for p in out.split("-") if p]
    return "-".join(parts) or "tab"


@dataclass(frozen=True)
class AnalysisPreparation:
    """Analysis readiness after a committed result, independent of its success."""

    has_params: bool
    error: str | None = None


class TabService:
    """Tab aggregate read model, cfg state, and tab lifecycle primitives."""

    def __init__(
        self,
        state: State,
        registry: Registry,
        writeback: WritebackLifecyclePort,
        cfg_resources: TabFormResources,
        form_sources: Callable[[Mapping[str, str] | None], TabFormSources],
        captured_parameters: Callable[
            [Mapping[str, str] | None], ParameterContext | None
        ],
    ) -> None:
        """Inject one owner-sequence form source capture for Open and Reset only.

        form_sources captures parameters/library/devices/value cache with optional
        named aliases overrides. It performs no hardware I/O. captured_parameters
        serves explicit consumer queries, not cfg preview or Run; EMPTY returns
        None without overrides. Errors propagate before registering a new tab.
        """
        self._state = state
        self._registry = registry
        # The tab lifecycle previews pane drafts and tears them down on close via
        # one narrow port, without depending on the concrete sibling service.
        self._writeback = writeback
        self._cfg_resources = cfg_resources
        self._form_sources = form_sources
        self._captured_parameters = captured_parameters

    def capture_parameters(
        self, tab_id: str, *, aliases: Mapping[str, str] | None = None
    ) -> ParameterContext | None:
        """Freeze current point/library/aliases for an existing tab's consumer.

        aliases maps alias names to concrete components for this capture only.
        None and {} use global rows. EMPTY yields None without overrides.
        Unknown tab and source or alias failures propagate; no observation guard is refreshed.
        """
        self._state.get_tab(tab_id)
        return self._captured_parameters(aliases)

    def preview_cfg(
        self, tab_id: str, *, aliases: Mapping[str, str] | None = None
    ) -> FormPreview:
        """Preview current input leaves using the tab's fixed source snapshot.

        None and empty aliases leave the snapshot unchanged. Nonempty aliases
        raise FailedPreconditionError; use resource.reset(aliases=...) to replace
        defaults with a one-shot source capture. No input or source is mutated.
        """
        if aliases:
            raise FailedPreconditionError("Reset with aliases before previewing cfg")
        return self._state.get_tab(tab_id).cfg.preview()

    def get_snapshot(self, tab_id: str) -> TabSnapshot:
        """Build the immutable full render model for one tab (all live fields
        populated). cfg_input is raw live data. RAM reload uses the
        separate WorkspaceTab FormInput JSON contract."""
        tab = self._state.get_tab(tab_id)
        ctx = self._state.session_env
        is_running = self._state.is_tab_running(tab_id)
        interaction = TabInteractionState(
            global_run_active=self._state.is_run_active() and not is_running,
            startup_active=self._state.is_startup_active(),
            has_context=ctx.has_context(),
            has_active_context=ctx.is_active(),
            has_soc=ctx.has_soc(),
            is_running=is_running,
            is_analyzing=tab.is_analyzing,
            has_run_result=tab.has_run_result(),
            has_analyze_result=tab.has_analyze_result(),
            has_figure=tab.has_figure(),
            has_post_analyze_result=tab.has_post_analyze_result(),
        )

        # Pane-owned writeback items via opaque drafts.
        def _items_for_pane(pane) -> tuple[WritebackItem, ...]:
            draft = pane.writeback_draft
            if draft is None:
                return ()
            return tuple(self._writeback.preview_draft(draft))

        analysis_items = _items_for_pane(tab.analysis)
        post_items = _items_for_pane(tab.post_analysis)
        metadata = None if tab.run.result is None else tab.run.result.metadata
        return TabSnapshot(
            adapter_name=tab.adapter_name,
            cfg_input=tab.cfg.input.data,
            tab_id=tab_id,
            interaction=interaction,
            capabilities=tab.adapter.capabilities,
            run=RunPaneSnapshot(
                result=tab.run.result,
                partial_result=tab.run.partial_result,
                source_path=tab.run.source_path,
                source_operation_id=tab.run.source_operation_id,
                run_id=tab.run.confirmed_run_id,
                acquisition_run_id=(metadata.run_id if metadata is not None else None),
            ),
            analysis=AnalysisPaneSnapshot(
                params=tab.analysis.params,
                result=tab.analysis.result,
                figures=tab.analysis.plots,
                writeback_items=analysis_items,
                unavailable_reasons=(
                    tab.analysis.writeback_draft.unavailable_reasons
                    if isinstance(tab.analysis.writeback_draft, WritebackDraft)
                    else ()
                ),
                has_writeback_draft=tab.analysis.writeback_draft is not None,
                source_operation_id=tab.analysis.source_operation_id,
                result_params=tab.analysis.result_params,
                figures_replaced=tab.analysis.figures_replaced,
            ),
            post_analysis=PostAnalysisPaneSnapshot(
                params=tab.post_analysis.params,
                result=tab.post_analysis.result,
                figures=tab.post_analysis.plots,
                writeback_items=post_items,
                unavailable_reasons=(
                    tab.post_analysis.writeback_draft.unavailable_reasons
                    if isinstance(tab.post_analysis.writeback_draft, WritebackDraft)
                    else ()
                ),
                has_writeback_draft=tab.post_analysis.writeback_draft is not None,
                source_operation_id=tab.post_analysis.source_operation_id,
                result_params=tab.post_analysis.result_params,
                figures_replaced=tab.post_analysis.figures_replaced,
            ),
            save=SavePaneSnapshot(
                comment=tab.save.comment,
                display_bases=_display_bases(tab.run.storage),
                saves_to_entry=tab.adapter.experiment_tag is not None,
            ),
            artifacts=self._state.get_artifact_snapshots(tab_id),
        )

    def new_tab(
        self,
        adapter_name: str,
        from_dict: WorkspaceTab | None = None,
        *,
        input: TabInput | None = None,
    ) -> str:
        """Create a tab form with one fixed source snapshot and fresh defaults.

        Defaults use current global aliases. Unresolved consumed fields may make
        the form Invalid without blocking tab creation. from_dict optionally
        supplies caller-owned RAM WorkspaceTab FormInput JSON for reload. input
        instead accepts an opaque authored TabInput; supplying both raises
        ValueError. Malformed input raises ValueError before publication.
        Compatible restoration logs discarded paths, also available through
        dropped_input_paths. Source captures and pane results are not restored.
        Owner-only; GUI owns the created tab, caller retains its input copy.
        Failed preparation publishes no partial tab.
        """
        if from_dict is not None and input is not None:
            raise ValueError("Specify either reload input or authored input, not both")
        adapter = self._registry.create(adapter_name)
        tab_id = f"{_slug(adapter_name)}-{uuid.uuid4().hex[:8]}"
        logger.info(
            "new_tab: adapter=%r tab_id=%r restore=%s",
            adapter_name,
            tab_id,
            from_dict is not None,
        )

        def make_form(ctx: FormContext) -> Form[ConfigBase]:
            form = adapter.make_form(ctx)
            form.bind_templates(self._registry.templates, adapter=adapter_name)
            return form

        cfg = self._cfg_resources.create(
            tab_id,
            make_form,
            capture=self._form_sources,
            initial=from_dict.form_input if from_dict is not None else input,
        )
        if cfg.dropped_paths:
            logger.warning(
                "Tab %s discarded incompatible input paths: %s",
                adapter_name,
                cfg.dropped_paths,
            )
        self._state.add_tab(
            tab_id,
            Session(
                adapter_name=adapter_name,
                adapter=adapter,
                cfg=cfg,
            ),
        )

        def cfg_changed() -> None:
            self._state.version.bump(f"tab:{tab_id}:cfg")

        cfg.subscribe(cfg_changed)
        return tab_id

    def read_inputs(self) -> tuple[StartupTab, ...]:
        """Read detached authored inputs in current tab order on the owner thread.

        No capture, evaluation or hardware I/O occurs. Later edits do not change
        these copies; callers own their destination and lifetime.
        """
        return tuple(
            StartupTab(tab.adapter_name, TabInput(tab.cfg.input.to_json()))
            for tab in self._state.tabs.values()
        )

    def dropped_input_paths(self, tab_id: str) -> tuple[FormPath, ...]:
        """Return discarded paths from this tab's compatible initial restoration."""
        return self._state.get_tab(tab_id).cfg.dropped_paths

    def list_adapter_names(self) -> list[str]:
        return self._registry.list_names()

    def adapter_guide(self, adapter_name: str) -> AdapterGuide:
        """Read a registered adapter's Markdown and resource directory, without a tab.

        adapter_name is a registry name. Unknown names raise KeyError; adapter
        construction and guide source errors propagate unchanged.
        """
        return self._registry.create(adapter_name).guide()

    def analyze_param_definitions(
        self, adapter_name: str, *, stage: Literal["primary", "post"]
    ) -> list[dict[str, Any]]:
        """Describe canonical adapter parameters before or after a result exists."""
        adapter = self._registry.create(adapter_name)
        params_cls = (
            adapter.analyze_params_cls()
            if stage == "primary"
            else adapter.post_analyze_params_cls()
        )
        return describe_analyze_params(params_cls)

    def close_tab(self, tab_id: str) -> None:
        logger.info("close_tab: tab_id=%r", tab_id)
        if self._state.is_tab_busy(tab_id):
            raise RuntimeError(f"Cannot close busy tab {tab_id!r}")
        self._cfg_resources.retire(tab_id)
        retired = self._state.remove_tab(tab_id)
        for draft in retired.writeback_drafts:
            try:
                self._writeback.teardown_draft(draft)
            except Exception:
                logger.exception("closed-tab draft teardown failed")
        release_retired_plots(retired)

    def get_tab_analyze_result(self, tab_id: str) -> object | None:
        return self._state.get_tab(tab_id).analysis.result

    def get_tab_adapter_name(self, tab_id: str) -> str:
        return self._state.get_tab(tab_id).adapter_name

    def prepare_result_analysis(self, tab_id: str) -> AnalysisPreparation:
        """Seed params from this Run/Load result and its saved source.

        tab_id selects a tab with a result. Use its publication parameters,
        including context-free None, never current tab/session bindings.
        Publish returned params; adapter failure becomes AnalysisPreparation.error
        without changing the result. Missing result raises RuntimeError.
        """
        tab = self._state.get_tab(tab_id)
        if tab.run.result is None:
            raise RuntimeError("No run result available to build analyze params")
        if tab.adapter.capabilities.analysis is AnalysisMode.NONE:
            return AnalysisPreparation(has_params=False)
        try:
            instance = tab.adapter.get_analyze_params(
                tab.run.result,
                replace(self._state.session_env, parameters=tab.run.parameters),
            )
        except Exception as exc:
            logger.exception("Analysis preparation failed for tab %s", tab_id)
            return AnalysisPreparation(has_params=False, error=str(exc))
        self._state.update_tab_analyze_param_instance(tab_id, instance)
        return AnalysisPreparation(has_params=True)

    def update_tab_analyze_param_instance(self, tab_id: str, instance: object) -> None:
        self._state.update_tab_analyze_param_instance(tab_id, instance)

    def initialize_tab_post_analyze_params(self, tab_id: str) -> object:
        """Seed and store Post params from the primary result and result source.

        tab_id selects a tab with a primary analysis result. Its Run/Load saved
        parameters supply the source, including None without fallback. Return
        the stored params. Missing primary result or adapter failure raises.
        """
        tab = self._state.get_tab(tab_id)
        if not tab.has_analyze_result():
            raise RuntimeError(
                "No primary analyze result available to build post-analysis params"
            )
        instance = tab.adapter.get_post_analyze_params(
            tab.analysis.result,
            replace(self._state.session_env, parameters=tab.run.parameters),
        )
        self._state.update_tab_post_analyze_param_instance(tab_id, instance)
        return instance

    def update_tab_post_analyze_param_instance(
        self, tab_id: str, instance: object
    ) -> None:
        self._state.update_tab_post_analyze_param_instance(tab_id, instance)

    def get_tab_post_analyze_result(self, tab_id: str) -> object | None:
        return self._state.get_tab(tab_id).post_analysis.result


def _display_bases(storage: PointStorageBinding | None) -> tuple[Path, ...]:
    """Parents of the Run's result/database roots, or () without storage."""
    if storage is None:
        return ()
    # Entry directories are <root>/<entry name>; one level above the root keeps
    # the root and entry names visible in the displayed path.
    return (storage.database_path.parent.parent, storage.result_path.parent.parent)
