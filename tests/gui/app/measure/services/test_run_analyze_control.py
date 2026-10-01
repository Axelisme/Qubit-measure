"""RunAnalyzeControlFacet public contract tests."""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, cast

import pytest
from zcu_tools.gui.app.measure.adapter import AnalysisMode
from zcu_tools.gui.app.measure.events.tab import (
    TabContentChangedPayload,
    TabContentFact,
)
from zcu_tools.gui.app.measure.interactive import PluginDefinition
from zcu_tools.gui.app.measure.services.analyze import ActiveInteractive
from zcu_tools.gui.app.measure.services.load import LoadTabResultOutcome
from zcu_tools.gui.app.measure.services.run_analyze_control import (
    RunAnalyzeControlFacet,
)
from zcu_tools.gui.app.measure.ui.interactive_frontend import (
    InteractiveFrontend,
    InteractiveFrontendEnv,
)
from zcu_tools.gui.cfg.resource import CfgId, CfgRef, CfgRevision, CfgStaleError
from zcu_tools.gui.session.adapters.manual_owner_scheduler import ManualOwnerScheduler
from zcu_tools.plotting.plots import Plots

from tests.gui._control_fakes import CallLog, call


class RecordingState:
    def __init__(
        self,
        log: CallLog,
        *,
        analysis: AnalysisMode = AnalysisMode.FIT,
        busy: bool = False,
    ) -> None:
        self._log = log
        self._busy = busy
        self.running_tab_id: str | None = "running-tab"
        self.session_env = SimpleNamespace(md="md", ml="ml", predictor="predictor")
        self.tab = SimpleNamespace(
            adapter=RecordingAdapter(log, analysis=analysis),
            run=SimpleNamespace(result="run-result"),
            cfg=SimpleNamespace(
                observe=lambda: SimpleNamespace(
                    ref=CfgRef(CfgId("cfg-1"), CfgRevision(0))
                )
            ),
        )

    def has_tab(self, tab_id: str) -> bool:
        self._log.add("state", "has_tab", tab_id)
        return tab_id == "tab-1"

    def get_tab(self, tab_id: str) -> object:
        self._log.add("state", "get_tab", tab_id)
        return self.tab

    def is_tab_busy(self, tab_id: str) -> bool:
        self._log.add("state", "is_tab_busy", tab_id)
        return self._busy


class RecordingAdapter:
    def __init__(self, log: CallLog, *, analysis: AnalysisMode) -> None:
        self._log = log
        self.capabilities = SimpleNamespace(analysis=analysis)

    def make_interactive_plugin(
        self, req: object, *, plots: Plots
    ) -> PluginDefinition[int, object]:
        self._log.add("adapter", "make_interactive_plugin", req, plots)
        return PluginDefinition("test", 0, (), lambda _state: None, lambda state: state)

    def make_interactive_frontend(
        self, plugin, session, env, request_finish, request_cancel, *, plots: Plots
    ) -> object:
        self._log.add(
            "adapter", "make_interactive_frontend", plugin, session, env, plots
        )
        return object()


class RecordingGuard:
    def __init__(self, log: CallLog) -> None:
        self._log = log

    def acquire_run_permit(self, tab_id: str, *, expected_revision: int) -> object:
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

    def start_run(self, permit: object, *, plots: Plots) -> int:
        self._log.add("run", "start_run", permit, plots)
        return 11

    def release_view_plots(self, tab_id: str) -> None:
        self._log.add("run", "release_view_plots", tab_id)

    def cancel_run(self) -> bool:
        self._log.add("run", "cancel_run")
        return True

    @property
    def active_token(self) -> int | None:
        return 11


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
        self, permit: object, analyze_params_instance: object, *, plots: Plots
    ) -> int:
        self._log.add(
            "analyze", "start_analyze", permit, analyze_params_instance, plots
        )
        return 22

    def start_plugin(
        self,
        permit: object,
        plugin: PluginDefinition[Any, Any],
        owner: ManualOwnerScheduler,
        *,
        analyze_params_instance: object,
        plots: Plots,
    ) -> int:
        self._log.add(
            "analyze",
            "start_plugin",
            permit,
            plugin,
            owner,
            analyze_params_instance,
            plots,
        )
        self.active = ActiveInteractive(plugin, plugin.open(owner), plots)
        return 23

    def get_interactive(self, tab_id: str) -> ActiveInteractive:
        self._log.add("analyze", "get_interactive", tab_id)
        return self.active

    def finish_plugin(self, tab_id: str) -> bool:
        self._log.add("analyze", "finish_plugin", tab_id)
        return True

    def cancel_interactive(self, tab_id: str) -> bool:
        self._log.add("analyze", "cancel_interactive", tab_id)
        return True

    def active_operations(self) -> tuple[tuple[str, int], ...]:
        return (("gui-fit", 22),)


class RecordingPostAnalyze:
    def __init__(self, log: CallLog) -> None:
        self._log = log

    def start_post_analyze(
        self,
        tab_id: str,
        post_analyze_params_instance: object,
        *,
        plots: Plots,
    ) -> int:
        self._log.add(
            "post_analyze",
            "start_post_analyze",
            tab_id,
            post_analyze_params_instance,
            plots,
        )
        return 33

    def active_operations(self) -> tuple[tuple[str, int], ...]:
        return (("gui-post", 33),)


class RecordingTab:
    def __init__(self, log: CallLog) -> None:
        self._log = log
        self.snapshot = object()
        self.analysis = SimpleNamespace(result=object())
        self.post_analysis = SimpleNamespace(result=object())

    def get_snapshot(self, tab_id: str) -> object:
        self._log.add("tab", "get_snapshot", tab_id)
        return self.snapshot

    def prepare_result_analysis(self, tab_id: str):
        from zcu_tools.gui.app.measure.services.tab import AnalysisPreparation

        self._log.add("tab", "prepare_result_analysis", tab_id)
        return AnalysisPreparation(has_params=True)

    def update_tab_analyze_param_instance(self, tab_id: str, instance: object) -> None:
        self._log.add("tab", "update_tab_analyze_param_instance", tab_id, instance)

    def get_tab_analyze_result(self, tab_id: str) -> object:
        self._log.add("tab", "get_tab_analyze_result", tab_id)
        return self.analysis.result

    def get_tab_post_analyze_result(self, tab_id: str) -> object:
        self._log.add("tab", "get_tab_post_analyze_result", tab_id)
        return self.post_analysis.result


class RecordingBus:
    def __init__(self, log: CallLog) -> None:
        self._log = log
        self.payloads: list[object] = []

    def emit(self, payload: object) -> None:
        self._log.add("bus", "emit", type(payload).__name__)
        self.payloads.append(payload)


class RecordingRenderHost:
    def interactive_presentation(self, tab_id: str) -> None:
        return None

    def discard_interactive_preview(self, tab_id: str) -> None:
        self._log.add("host", "discard_interactive_preview", tab_id)

    def __init__(self, log: CallLog, *, mount_error: Exception | None = None) -> None:
        self._log = log
        self._mount_error = mount_error

    def make_run_container(self, tab_id: str) -> Any:
        self._log.add("host", "make_run_container", tab_id)
        return "figure-container"

    def make_analysis_container(self, tab_id: str) -> Any:
        self._log.add("host", "make_analysis_container", tab_id)
        return "figure-container"

    def make_post_analysis_container(self, tab_id: str) -> Any:
        self._log.add("host", "make_post_analysis_container", tab_id)
        return "figure-container"

    def mount_interactive_analysis(
        self,
        tab_id: str,
        frontend_factory: Callable[[InteractiveFrontendEnv], InteractiveFrontend],
    ) -> None:
        self._log.add("host", "mount_interactive_analysis", tab_id, frontend_factory)
        if self._mount_error is not None:
            raise self._mount_error
        self.frontend = frontend_factory(cast(InteractiveFrontendEnv, object()))

    def unmount_interactive_analysis(
        self, tab_id: str, *, restore_result: bool = False
    ) -> None:
        self._log.add("host", "unmount_interactive_analysis", tab_id, restore_result)


def _facet(
    *,
    analysis: AnalysisMode = AnalysisMode.FIT,
    busy: bool = False,
    mount_error: Exception | None = None,
) -> tuple[RunAnalyzeControlFacet, CallLog, RecordingState, RecordingBus]:
    log = CallLog()
    state = RecordingState(log, analysis=analysis, busy=busy)
    bus = RecordingBus(log)
    host = RecordingRenderHost(log, mount_error=mount_error)
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
            owner_scheduler=ManualOwnerScheduler(),
        ),
        log,
        state,
        bus,
    )


def test_gui_started_run_and_both_analysis_stages_are_indexed() -> None:
    facet, _log, _state, _bus = _facet()

    assert [(op.op, op.tab, op.kind) for op in facet.active_tab_operations()] == [
        (11, "running-tab", "run"),
        (22, "gui-fit", "analyze"),
        (33, "gui-post", "analyze"),
    ]


def test_run_control_starts_with_guard_and_live_container() -> None:
    facet, log, _state, _bus = _facet()

    assert facet.start_run("tab-1", CfgRef(CfgId("cfg-1"), CfgRevision(0))) == 11

    assert log.calls[:4] == [
        call("state", "get_tab", "tab-1"),
        call("guard", "acquire_run_permit", "tab-1"),
        call("state", "is_tab_busy", "tab-1"),
        call("host", "make_run_container", "tab-1"),
    ]
    assert log.calls[4].target == "run"
    assert log.calls[4].method == "start_run"
    assert log.calls[4].args[0] == "run-permit"
    assert isinstance(log.calls[4].args[1], Plots)


@pytest.mark.parametrize(
    "expected",
    [CfgRef(CfgId("cfg-1"), CfgRevision(1)), CfgRef(CfgId("other"), CfgRevision(0))],
)
def test_run_rejects_a_different_publication_before_permit_or_presentation(
    expected: CfgRef,
) -> None:
    facet, log, _state, _bus = _facet()

    with pytest.raises(CfgStaleError) as caught:
        facet.start_run("tab-1", expected)

    assert caught.value.expected == expected
    assert caught.value.actual == CfgRef(CfgId("cfg-1"), CfgRevision(0))
    assert log.calls == [call("state", "get_tab", "tab-1")]


def test_load_result_initializes_analyze_params_and_emits_content_changed() -> None:
    facet, log, _state, bus = _facet()

    outcome = facet.load_tab_result("tab-1", "/tmp/result.hdf5")

    assert outcome.has_analyze_params is True
    assert log.calls == [
        call("guard", "acquire_load_permit", "tab-1"),
        call(
            "load", "load_result", SimpleNamespace(tab_id="tab-1"), "/tmp/result.hdf5"
        ),
        call("run", "release_view_plots", "tab-1"),
        call("state", "get_tab", "tab-1"),
        call("tab", "initialize_tab_analyze_params", "tab-1"),
        call("bus", "emit", "TabContentChangedPayload"),
    ]
    payload = bus.payloads[0]
    assert isinstance(payload, TabContentChangedPayload)
    content_payload = payload
    assert content_payload.fact is TabContentFact.LOADED_RESULT_COMMITTED


def test_fit_analyze_uses_worker_service_and_live_container() -> None:
    facet, log, _state, _bus = _facet()
    params = object()

    assert facet.analyze("tab-1", params) == 22

    assert log.calls[:4] == [
        call("guard", "acquire_analyze_permit", "tab-1"),
        call("state", "is_tab_busy", "tab-1"),
        call("state", "get_tab", "tab-1"),
        call("host", "make_analysis_container", "tab-1"),
    ]
    assert log.calls[4].method == "start_analyze"
    assert log.calls[4].args[:2] == ("analyze-permit", params)
    assert isinstance(log.calls[4].args[2], Plots)


def test_interactive_analyze_mounts_render_host_session() -> None:
    facet, log, _state, _bus = _facet(analysis=AnalysisMode.INTERACTIVE)

    assert facet.analyze("tab-1", "params") == 23

    assert [entry.target for entry in log.calls] == [
        "guard",
        "state",
        "state",
        "state",
        "adapter",
        "analyze",
        "analyze",
        "host",
        "adapter",
    ]
    assert log.calls[5].method == "start_plugin"
    assert log.calls[5].args[3] == "params"


def test_interactive_mount_failure_unmounts_and_cancels_operation() -> None:
    error = RuntimeError("mount failed")
    facet, log, _state, _bus = _facet(
        analysis=AnalysisMode.INTERACTIVE, mount_error=error
    )

    with pytest.raises(RuntimeError) as exc_info:
        facet.analyze("tab-1", "params")

    assert exc_info.value is error
    assert [entry.method for entry in log.calls[-3:]] == [
        "mount_interactive_analysis",
        "unmount_interactive_analysis",
        "cancel_interactive",
    ]


def test_post_analyze_uses_shared_live_container() -> None:
    facet, log, _state, _bus = _facet()

    assert facet.start_post_analyze("tab-1", "post-params") == 33

    assert log.calls[:2] == [
        call("state", "is_tab_busy", "tab-1"),
        call("host", "make_post_analysis_container", "tab-1"),
    ]
    assert log.calls[2].method == "start_post_analyze"
    assert log.calls[2].args[:2] == ("tab-1", "post-params")
    assert isinstance(log.calls[2].args[2], Plots)


@pytest.mark.parametrize(
    ("operation", "forbidden_host_call"),
    [
        (
            lambda facet: facet.start_run(
                "tab-1", CfgRef(CfgId("cfg-1"), CfgRevision(0))
            ),
            "make_run_container",
        ),
        (lambda facet: facet.analyze("tab-1", object()), "make_analysis_container"),
        (
            lambda facet: facet.start_post_analyze("tab-1", object()),
            "make_post_analysis_container",
        ),
    ],
)
def test_busy_invocation_rejects_before_clearing_presentation(
    operation, forbidden_host_call: str
) -> None:
    facet, log, _state, _bus = _facet(busy=True)

    with pytest.raises(RuntimeError, match="busy"):
        operation(facet)

    assert all(entry.method != forbidden_host_call for entry in log.calls)
