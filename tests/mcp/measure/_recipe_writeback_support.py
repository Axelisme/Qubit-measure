"""Real writeback owner behind a synthetic recipe's GUI transport."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock

from zcu_tools.cfg_model import ConstWaveformCfg, PulseSpec
from zcu_tools.format_version import YamlMap, YamlValue
from zcu_tools.gui.app.measure.adapter import AdapterCapabilities, AnalysisMode
from zcu_tools.gui.app.measure.services.ports import (
    AnalysisPaneSnapshot,
    PostAnalysisPaneSnapshot,
    TabSnapshot,
)
from zcu_tools.gui.app.measure.state import Session, State
from zcu_tools.gui.app.measure.writeback import (
    ModuleWriteback,
    ParameterWriteback,
    WritebackService,
)
from zcu_tools.gui.app.measure.writeback.control import WritebackControlFacet
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.expected_error import InvalidInputError
from zcu_tools.gui.session.context_control import ContextControlFacet
from zcu_tools.gui.session.context_types import ContextTarget, PointUseRequest
from zcu_tools.gui.session.services.context import ContextService
from zcu_tools.gui.session.state import SessionState
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.resources.entry import ResultEntry
from zcu_tools.resources.waveform_assets import ArbWaveformDatabase

from tests.gui.app.measure.remote._helpers import dispatch_handler

from ._script_recipe_support import SyntheticGui


class RejectingContext(ContextService):
    """Context collaborator rejects one setter after real committed writes."""

    def set_parameter(
        self,
        target: ContextTarget,
        path: str,
        value: YamlValue,
        *,
        metadata: YamlMap | None = None,
    ) -> YamlValue:
        if path == "general.ext.rejected":
            raise InvalidInputError("synthetic setter refused the value")
        return super().set_parameter(target, path, value, metadata=metadata)


class WritebackGui(SyntheticGui):
    """Keep acquisition synthetic while dispatching real writeback RPC handlers."""

    render_view = None

    def __init__(
        self, tmp_path: Path, *, module: bool = False, partial: bool = False
    ) -> None:
        super().__init__()
        self.owner = ThreadPoolExecutor(max_workers=1)
        try:
            self.owner.submit(
                self.initialize, tmp_path, module=module, partial=partial
            ).result()
        except BaseException:
            self.owner.shutdown()
            raise

    def initialize(self, tmp_path: Path, *, module: bool, partial: bool) -> None:
        entry = ResultEntry.create(
            "entry",
            result_root=tmp_path / "results",
            database_root=tmp_path / "database",
        )
        entry.new_point("point")
        self.context = RejectingContext(
            SessionState(SessionEnv(parameters=None, soc=None, soccfg=None)),
            BaseEventBus(),
            bind_assets=ArbWaveformDatabase.info_at,
        )
        self.context.install_entry(entry)
        self.context.use_point(PointUseRequest("point"))
        control = ContextControlFacet(context=self.context)
        control.add_component("Q1", "fake/drive/a", {})
        self.service = WritebackService(control, arb_names=lambda: ())
        self.drafts = {
            "analysis": self.service.create_draft(
                [ParameterWriteback("general.ext.first", "first", 20.0)],
                parameters=self.context.read_context(),
                experiment_name="synthetic",
            ),
            "post_analysis": self.service.create_draft(
                [
                    ModuleWriteback(
                        "Q1.pulses.pi",
                        "pulse",
                        PulseSpec(
                            ch=0,
                            nqz=1,
                            freq=100.0,
                            gain=0.2,
                            waveform=ConstWaveformCfg(length=0.1),
                        ),
                    )
                ]
                if module
                else [
                    ParameterWriteback("general.ext.second", "second", 30.0),
                    ParameterWriteback("general.ext.rejected", "rejected", 40.0),
                    ParameterWriteback("general.ext.later", "later", 50.0),
                ]
                if partial
                else [ParameterWriteback("general.ext.second", "second", 30.0)],
                parameters=self.context.read_context(),
                experiment_name="synthetic",
            ),
        }
        self.state = State(self.context.get_session_env())
        self.state.add_tab(
            "tab-1",
            Session(adapter_name="synthetic", adapter=MagicMock(), cfg=MagicMock()),
        )
        tab = self.state.get_tab("tab-1")
        tab.analysis.writeback_draft = self.drafts["analysis"]
        tab.post_analysis.writeback_draft = self.drafts["post_analysis"]
        self.writeback_control = WritebackControlFacet(
            state=self.state,
            guard=MagicMock(),
            writeback=self.service,
            resource_versions=lambda: {},
            bus=BaseEventBus(),
        )

    def respond(self, method: str, params: dict[str, object]) -> dict[str, object]:
        if method == "tab.get_post_analyze_result":
            return {"summary": {"estimate": 3.5}, "invalid": []}
        if method.startswith("tab.writeback_"):
            return dict(
                self.owner.submit(dispatch_handler, self, method, params).result()
            )
        return super().respond(method, params)

    def read_context(self):
        return self.owner.submit(self.context.read_context).result()

    def replace_post(self) -> None:
        def replace() -> None:
            original = self.drafts["post_analysis"]
            self.drafts["post_analysis"] = self.service.create_draft(
                original.items,
                parameters=original.parameters,
                experiment_name="replacement",
            )
            self.state.get_tab("tab-1").post_analysis.writeback_draft = self.drafts[
                "post_analysis"
            ]

        self.owner.submit(replace).result()

    def require_analysis_operation(
        self, tab_id: str, pane: str, operation_id: int
    ) -> None:
        pass

    def has_tab(self, tab_id: str) -> bool:
        return tab_id == "tab-1"

    def get_session_env(self) -> SessionEnv:
        return self.context.get_session_env()

    def get_tab_snapshot(self, tab_id: str) -> TabSnapshot:
        return TabSnapshot(
            tab_id=tab_id,
            adapter_name="synthetic",
            cfg_input={},
            capabilities=AdapterCapabilities(
                analysis=AnalysisMode.FIT, post_analysis=True
            ),
            analysis=AnalysisPaneSnapshot(
                params=None,
                result=None,
                figures=None,
                writeback_items=self.drafts["analysis"].items,
                has_writeback_draft=True,
            ),
            post_analysis=PostAnalysisPaneSnapshot(
                params=None,
                result=None,
                figures=None,
                writeback_items=self.drafts["post_analysis"].items,
                has_writeback_draft=True,
            ),
        )
