"""RunAnalyzeControlFacet public contract tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

from zcu_tools.gui.app.main.adapter import AnalysisMode
from zcu_tools.gui.app.main.events.tab import TabContentChangedPayload
from zcu_tools.gui.app.main.services.load import LoadTabResultOutcome
from zcu_tools.gui.app.main.services.run_analyze_control import RunAnalyzeControlFacet

from tests.gui._control_fakes import CallLog, call


class RecordingState:
    def __init__(
        self, log: CallLog, *, analysis: AnalysisMode = AnalysisMode.FIT
    ) -> None:
        self._log = log
        self.running_tab_id: str | None = "running-tab"
        self.exp_context = SimpleNamespace(md="md", ml="ml", predictor="predictor")
        self.tab = SimpleNamespace(
            adapter=RecordingAdapter(log, analysis=analysis),
            run_result="run-result",
        )

    def has_tab(self, tab_id: str) -> bool:
        self._log.add("state", "has_tab", tab_id)
        return tab_id == "tab-1"

    def get_tab(self, tab_id: str) -> object:
        self._log.add("state", "get_tab", tab_id)
        return self.tab


class RecordingAdapter:
    def __init__(self, log: CallLog, *, analysis: AnalysisMode) -> None:
        self._log = log
        self.capabilities = SimpleNamespace(analysis=analysis)

    def setup_interactive_analysis(self, req: object, host: object) -> object:
        self._log.add("adapter", "setup_interactive_analysis", req, host)
        return "interactive-session"


class RecordingGuard:
    def __init__(self, log: CallLog) -> None:
        self._log = log

    def acquire_run_permit(self, tab_id: str) -> object:
        self._log.add("guard", "acquire_run_permit", tab_id)
        return "run-permit"

    def acquire_load_permit(self, tab_id: str) -> object:
        self._log.add("guard", "acquire_load_permit", tab_id)
        return SimpleNamespace(tab_id=tab_id)

    def acquire_analyze_permit(self, tab_id: str) -> object:
        self._log.add("guard", "acquire_analyze_permit", tab_id)
        return "analyze-permit"


class RecordingRun:
    def __init__(self, log: CallLog) -> None:
        self._log = log

    def start_run(self, permit: object, live_container: object) -> int:
        self._log.add("run", "start_run", permit, live_container)
        return 11

    def cancel_run(self) -> bool:
        self._log.add("run", "cancel_run")
        return True


class RecordingLoad:
    def __init__(self, log: CallLog) -> None:
        self._log = log

    def load_result(self, permit: object, data_path: str) -> LoadTabResultOutcome:
        self._log.add("load", "load_result", permit, data_path)
        return LoadTabResultOutcome(
            tab_id="tab-1",
            data_path=data_path,
            result_type="Result",
            has_cfg_snapshot=True,
            has_analyze_params=False,
        )


class RecordingAnalyze:
    def __init__(self, log: CallLog) -> None:
        self._log = log

    def start_analyze(
        self, permit: object, analyze_params_instance: object, figure_container: object
    ) -> int:
        self._log.add(
            "analyze",
            "start_analyze",
            permit,
            analyze_params_instance,
            figure_container,
        )
        return 22

    def start_interactive(self, permit: object) -> int:
        self._log.add("analyze", "start_interactive", permit)
        return 23

    def finish_interactive(self, tab_id: str, session: object) -> None:
        self._log.add("analyze", "finish_interactive", tab_id, session)

    def cancel_interactive(self, tab_id: str) -> bool:
        self._log.add("analyze", "cancel_interactive", tab_id)
        return True


class RecordingPostAnalyze:
    def __init__(self, log: CallLog) -> None:
        self._log = log

    def start_post_analyze(
        self,
        tab_id: str,
        post_analyze_params_instance: object,
        figure_container: object,
    ) -> int:
        self._log.add(
            "post_analyze",
            "start_post_analyze",
            tab_id,
            post_analyze_params_instance,
            figure_container,
        )
        return 33


class RecordingTab:
    def __init__(self, log: CallLog) -> None:
        self._log = log
        self.snapshot = object()
        self.analyze_result = object()
        self.post_analyze_result = object()

    def get_snapshot(self, tab_id: str) -> object:
        self._log.add("tab", "get_snapshot", tab_id)
        return self.snapshot

    def initialize_tab_analyze_params(self, tab_id: str) -> None:
        self._log.add("tab", "initialize_tab_analyze_params", tab_id)

    def get_tab_analyze_result(self, tab_id: str) -> object:
        self._log.add("tab", "get_tab_analyze_result", tab_id)
        return self.analyze_result

    def get_tab_post_analyze_result(self, tab_id: str) -> object:
        self._log.add("tab", "get_tab_post_analyze_result", tab_id)
        return self.post_analyze_result


class RecordingBus:
    def __init__(self, log: CallLog) -> None:
        self._log = log
        self.payloads: list[object] = []

    def emit(self, payload: object) -> None:
        self._log.add("bus", "emit", type(payload).__name__)
        self.payloads.append(payload)


class RecordingRenderHost:
    def __init__(self, log: CallLog) -> None:
        self._log = log

    def make_live_container(self, tab_id: str) -> Any:
        self._log.add("host", "make_live_container", tab_id)
        return "figure-container"

    def mount_interactive_analysis(
        self, tab_id: str, session_factory: object, on_finish: object
    ) -> None:
        self._log.add(
            "host", "mount_interactive_analysis", tab_id, session_factory, on_finish
        )

    def unmount_interactive_analysis(self, tab_id: str) -> None:
        self._log.add("host", "unmount_interactive_analysis", tab_id)


def _facet(
    *, analysis: AnalysisMode = AnalysisMode.FIT
) -> tuple[RunAnalyzeControlFacet, CallLog, RecordingState, RecordingBus]:
    log = CallLog()
    state = RecordingState(log, analysis=analysis)
    bus = RecordingBus(log)
    host = RecordingRenderHost(log)
    return (
        RunAnalyzeControlFacet(
            state=cast(Any, state),
            bus=cast(Any, bus),
            guard=cast(Any, RecordingGuard(log)),
            tab=cast(Any, RecordingTab(log)),
            load=cast(Any, RecordingLoad(log)),
            run=cast(Any, RecordingRun(log)),
            analyze=cast(Any, RecordingAnalyze(log)),
            post_analyze=cast(Any, RecordingPostAnalyze(log)),
            render_host=lambda: host,
        ),
        log,
        state,
        bus,
    )


def test_run_control_starts_with_guard_and_live_container() -> None:
    facet, log, _state, _bus = _facet()

    assert facet.start_run("tab-1") == 11

    assert log.calls == [
        call("guard", "acquire_run_permit", "tab-1"),
        call("host", "make_live_container", "tab-1"),
        call("run", "start_run", "run-permit", "figure-container"),
    ]


def test_load_result_initializes_analyze_params_and_emits_content_changed() -> None:
    facet, log, _state, bus = _facet()

    outcome = facet.load_tab_result("tab-1", "/tmp/result.hdf5")

    assert outcome.has_analyze_params is True
    assert log.calls == [
        call("guard", "acquire_load_permit", "tab-1"),
        call(
            "load", "load_result", SimpleNamespace(tab_id="tab-1"), "/tmp/result.hdf5"
        ),
        call("state", "get_tab", "tab-1"),
        call("tab", "initialize_tab_analyze_params", "tab-1"),
        call("bus", "emit", "TabContentChangedPayload"),
    ]
    assert isinstance(bus.payloads[0], TabContentChangedPayload)


def test_fit_analyze_uses_worker_service_and_live_container() -> None:
    facet, log, _state, _bus = _facet()
    params = object()

    assert facet.analyze("tab-1", params) == 22

    assert log.calls == [
        call("guard", "acquire_analyze_permit", "tab-1"),
        call("state", "get_tab", "tab-1"),
        call("host", "make_live_container", "tab-1"),
        call("analyze", "start_analyze", "analyze-permit", params, "figure-container"),
    ]


def test_interactive_analyze_mounts_render_host_session() -> None:
    facet, log, _state, _bus = _facet(analysis=AnalysisMode.INTERACTIVE)

    assert facet.analyze("tab-1", "params") == 23

    assert [entry.target for entry in log.calls] == [
        "guard",
        "state",
        "state",
        "analyze",
        "host",
    ]
    assert log.calls[3] == call("analyze", "start_interactive", "analyze-permit")


def test_post_analyze_uses_shared_live_container() -> None:
    facet, log, _state, _bus = _facet()

    assert facet.start_post_analyze("tab-1", "post-params") == 33

    assert log.calls == [
        call("host", "make_live_container", "tab-1"),
        call(
            "post_analyze",
            "start_post_analyze",
            "tab-1",
            "post-params",
            "figure-container",
        ),
    ]
