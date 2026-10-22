"""Accepted evidence and fixed destinations at the real writeback/owner seam."""

from collections.abc import Generator
from dataclasses import dataclass, replace
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from qtpy.QtWidgets import QApplication
from zcu_tools.cfg_model import ConstWaveformCfg, PulseSpec
from zcu_tools.core.element import ElementIdentity
from zcu_tools.gui.app.measure.adapter import (
    AnalyzeRequest,
    PostAnalyzeRequest,
    PostWritebackRequest,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.services.analyze import AnalyzeService
from zcu_tools.gui.app.measure.services.post_analyze import PostAnalyzeService
from zcu_tools.gui.app.measure.writeback import (
    ModuleWriteback,
    ParameterWriteback,
    WritebackDraft,
    WritebackEdit,
    WritebackItem,
    WritebackService,
    WritebackWritten,
)
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.expected_error import InvalidInputError
from zcu_tools.gui.interactive import PluginDefinition
from zcu_tools.gui.session.adapters.manual_owner_scheduler import ManualOwnerScheduler
from zcu_tools.gui.session.context_control import ContextControlFacet
from zcu_tools.gui.session.context_types import (
    CommandOrigin,
    PointUseRequest,
)
from zcu_tools.gui.session.expression import evaluate_scalar_expr
from zcu_tools.gui.session.operation_runner import OperationRunner
from zcu_tools.gui.session.services.context import ContextService
from zcu_tools.gui.session.state import SessionState
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.plotting.plots import NonPresentingHost, Plots
from zcu_tools.resources.entry import (
    ResultEntry,
    document_meta,
    document_value,
    snapshot,
)
from zcu_tools.resources.entry.ledger import RecordsLedger
from zcu_tools.resources.entry.ledger_models import AcceptedPayload, LedgerEvent
from zcu_tools.resources.entry.registry import RoleSpec
from zcu_tools.resources.waveform_assets import ArbWaveformDatabase

from tests.gui.app.measure._native_save_support import PaneSummary, native_save_session
from tests.gui.app.measure._record_support import require_metadata, require_record
from tests.resources.entry.fakes import register_fakes, registry_state, restore_registry


@dataclass
class WritebackSession:
    entry: ResultEntry
    control: ContextControlFacet
    service: WritebackService


@pytest.fixture
def session(tmp_path: Path) -> Generator[WritebackSession]:
    before = registry_state()
    try:
        register_fakes()
        installed = registry_state()
        entry = ResultEntry.create(
            "entry",
            result_root=tmp_path / "results",
            database_root=tmp_path / "Database",
        )
        bus = BaseEventBus()
        context = ContextService(
            SessionState(SessionEnv(parameters=None, soc=None, soccfg=None)),
            bus,
            bind_assets=ArbWaveformDatabase.info_at,
        )
        entry.new_point("working")
        context.install_entry(entry)
        context.use_point(PointUseRequest("working"))
        control = ContextControlFacet(context=context)
        control.add_component("Q1", "fake/drive/a", {})
        service = WritebackService(control, arb_names=lambda: ())
        yield WritebackSession(entry, control, service)
        assert registry_state() == installed
    finally:
        restore_registry(before)


@pytest.fixture
def saved_session(session: WritebackSession) -> WritebackSession:
    session.entry.new_point("saved")
    session.control.use_point(PointUseRequest("saved"))
    session.control.add_component("Q1", "fake/drive/a", {})
    return session


def test_draft_detaches_author_element_identity(session: WritebackSession) -> None:
    identity = ElementIdentity(index=2, name="Q3")
    proposal = ParameterWriteback(
        "Q1.ext.gain", "Gain", proposed_value=0.4, element=identity
    )
    draft = session.service.create_draft(
        [proposal],
        parameters=session.control.read_context(),
        experiment_name="synthetic",
    )
    try:
        identity["name"] = "later"
        identity["index"] = 0
        assert session.service.preview_draft(draft)[0].element == {
            "index": 2,
            "name": "Q3",
        }
    finally:
        session.service.teardown_draft(draft)


@pytest.mark.parametrize("save_before_accept", [False, True])
def test_accepted_retains_run_identity_with_or_without_save(
    tmp_path: Path, qapp: QApplication, save_before_accept: bool
) -> None:
    with native_save_session(tmp_path, qapp) as native:
        control = ContextControlFacet(context=native.context)
        service = WritebackService(control, arb_names=lambda: ())
        tab = native.state.get_tab("tab")
        captured = control.read_context()
        assert require_record(tab.run).metadata is not None
        source_run_id = require_metadata(tab.run).run_id
        draft = service.create_draft(
            [ParameterWriteback("general.ext.flux_value", "Flux", proposed_value=2.0)],
            parameters=captured,
            run_id=source_run_id,
            experiment_name="test/writeback",
        )
        if save_before_accept:
            native.save()
        assert (
            any(event.kind == "saved" for event in native.entry.ledger.events())
            is save_before_accept
        )
        tab.run.result = replace(
            require_record(tab.run),
            metadata=native.metadata.model_copy(update={"run_id": "later-run"}),
        )
        service.apply_draft(draft)
        event = native.entry.ledger.events()[-1]
        assert event.kind == "accepted"
        assert isinstance(event.payload, AcceptedPayload)
        assert event.payload.run_id == source_run_id
        current = control.read_target(captured.target)
        assert document_meta(current.document(), "general.ext.flux_value") == {}
        assert draft.run_id == source_run_id
        service.teardown_draft(draft)


@pytest.mark.parametrize("interactive", [False, True])
@pytest.mark.parametrize("source", ["run", "load", "without_metadata", "context_free"])
def test_analysis_draft_uses_start_run_identity(
    tmp_path: Path,
    qapp: QApplication,
    interactive: bool,
    source: str,
    monkeypatch,
) -> None:
    with native_save_session(tmp_path, qapp) as native:
        tab = native.state.get_tab("tab")
        if source == "load":
            native.state.update_tab_loaded_result(
                "tab",
                replace(require_record(tab.run), metadata=native.metadata),
                "historical.h5",
                parameters=tab.run.parameters,
                storage=tab.run.storage,
            )
        elif source in ("without_metadata", "context_free"):
            tab.run.result = replace(require_record(tab.run), metadata=None)
            if source == "context_free":
                tab.run.parameters = None
        captured = tab.run.parameters
        expected = (
            require_metadata(tab.run).run_id
            if require_record(tab.run).metadata is not None
            else None
        )
        seen_analyze: list[AnalyzeRequest[object, object]] = []
        seen_proposals: list[WritebackRequest[object, PaneSummary]] = []

        def analyze(
            request: AnalyzeRequest[object, object], *, plots: Plots
        ) -> PaneSummary:
            del plots
            seen_analyze.append(request)
            return PaneSummary()

        def proposals(
            request: WritebackRequest[object, PaneSummary],
        ) -> list[WritebackItem]:
            seen_proposals.append(request)
            return []

        monkeypatch.setattr(tab.adapter, "analyze", analyze)
        monkeypatch.setattr(tab.adapter, "get_writeback_items", proposals)
        control = ContextControlFacet(context=native.context)
        writeback = WritebackService(control, arb_names=lambda: ())
        runner = OperationRunner(
            None, native.handles, None, native.background, native.bus
        )
        service = AnalyzeService(
            native.state, runner, native.bus, writeback, native.handles
        )
        plots = Plots(NonPresentingHost())
        current = control.read_context() if captured is None else None
        native.state.set_context(replace(native.state.session_env, parameters=current))
        if interactive:
            plugin = PluginDefinition(
                "pick", 2, (), lambda _state: None, lambda _state: PaneSummary()
            )
            service.start_plugin(
                "tab",
                plugin,
                ManualOwnerScheduler(),
                analyze_params_instance="submitted",
                plots=plots,
            )
        else:
            service.start_analyze("tab", object(), plots=plots)
        tab.run.result = replace(
            require_record(tab.run),
            metadata=native.metadata.model_copy(update={"run_id": "later-run"}),
        )
        if interactive:
            assert service.finish_plugin("tab")
        else:
            assert native.background.quiesce()
            assert not tab.is_analyzing
        assert isinstance(tab.analysis.result, PaneSummary)
        draft = tab.analysis.writeback_draft
        assert isinstance(draft, WritebackDraft)
        assert draft.run_id == expected
        assert draft.parameters is captured
        assert len(seen_proposals) == 1
        assert seen_proposals[0].ctx.parameters is captured
        if not interactive:
            assert len(seen_analyze) == 1
            assert seen_analyze[0].parameters is captured
        writeback.teardown_draft(draft)


@pytest.mark.parametrize("save_before_accept", [False, True], ids=["unsaved", "saved"])
@pytest.mark.parametrize(
    "source",
    ["run", "load", "without_metadata", "context_free", "without_primary_draft"],
)
def test_post_analysis_draft_retains_primary_identity(
    tmp_path: Path,
    qapp: QApplication,
    source: str,
    save_before_accept: bool,
    monkeypatch,
) -> None:
    with native_save_session(tmp_path, qapp) as native:
        tab = native.state.get_tab("tab")
        original_capture = tab.run.parameters
        if source == "load":
            native.state.update_tab_loaded_result(
                "tab",
                replace(require_record(tab.run), metadata=native.metadata),
                "historical.h5",
                parameters=original_capture,
                storage=tab.run.storage,
            )
        elif source in ("without_metadata", "context_free"):
            tab.run.result = replace(require_record(tab.run), metadata=None)
            if source == "context_free":
                tab.run.parameters = None
        captured = tab.run.parameters
        expected = native.metadata.run_id if source in ("run", "load") else None
        control = ContextControlFacet(context=native.context)
        writeback = WritebackService(control, arb_names=lambda: ())
        primary_draft = (
            None
            if source == "without_primary_draft"
            else writeback.create_draft(
                [],
                parameters=captured,
                run_id=expected,
                experiment_name="test/writeback",
            )
        )
        primary_result = PaneSummary()
        native.state.update_tab_analyze(
            "tab", primary_result, None, writeback_draft=primary_draft
        )
        if save_before_accept:
            # Save is independent of whether Primary has a native identity.
            tab.run.result = replace(require_record(tab.run), metadata=native.metadata)
            native.save()
        seen_requests: list[PostAnalyzeRequest[object, PaneSummary, object]] = []
        seen_proposals: list[
            PostWritebackRequest[object, PaneSummary, PaneSummary]
        ] = []

        def post_analyze(
            request: PostAnalyzeRequest[object, PaneSummary, object], *, plots: Plots
        ) -> PaneSummary:
            del plots
            seen_requests.append(request)
            return PaneSummary()

        def proposals(
            request: PostWritebackRequest[object, PaneSummary, PaneSummary],
        ) -> list[WritebackItem]:
            seen_proposals.append(request)
            return (
                []
                if captured is None
                else [
                    ParameterWriteback(
                        "general.ext.flux_value", "Flux", proposed_value=3.0
                    )
                ]
            )

        monkeypatch.setattr(tab.adapter, "post_analyze", post_analyze)
        monkeypatch.setattr(tab.adapter, "get_post_writeback_items", proposals)
        runner = OperationRunner(
            None, native.handles, None, native.background, native.bus
        )
        service = PostAnalyzeService(
            native.state, runner, native.bus, native.handles, writeback
        )
        # A later live Run id must not replace the Primary draft's identity.
        tab.run.result = replace(
            require_record(tab.run),
            metadata=native.metadata.model_copy(
                update={"run_id": "not-primary-source"}
            ),
        )
        native.entry.new_point("later-point")
        control.use_point(PointUseRequest("later-point"))
        current = control.read_context()
        assert current is not captured
        service.start_post_analyze("tab", object(), plots=Plots(NonPresentingHost()))
        # Terminal delivery must use the operation-start identity and context.
        tab.analysis.writeback_draft = None
        tab.run.result = replace(
            require_record(tab.run),
            metadata=native.metadata.model_copy(update={"run_id": "terminal-live-run"}),
        )
        tab.run.parameters = current
        assert native.background.quiesce()
        draft = tab.post_analysis.writeback_draft
        assert isinstance(draft, WritebackDraft)
        assert draft.run_id == expected
        assert draft.parameters is captured
        assert [request.parameters is captured for request in seen_requests] == [True]
        assert [proposal.ctx.parameters is captured for proposal in seen_proposals] == [
            True
        ]
        writeback.teardown_draft(draft)
        if primary_draft is not None:
            writeback.teardown_draft(primary_draft)


def test_post_rejects_incompatible_primary_draft_before_submission(
    tmp_path: Path, qapp: QApplication
) -> None:
    with native_save_session(tmp_path, qapp) as native:
        native.state.update_tab_analyze(
            "tab", PaneSummary(), None, writeback_draft=object()
        )
        runner = OperationRunner(
            None, native.handles, None, native.background, native.bus
        )
        service = PostAnalyzeService(
            native.state, runner, native.bus, native.handles, MagicMock()
        )
        plots = Plots(NonPresentingHost())
        try:
            with pytest.raises(TypeError, match="Primary writeback draft"):
                service.start_post_analyze("tab", object(), plots=plots)
            assert not native.state.get_tab("tab").is_analyzing
        finally:
            assert native.background.quiesce()
            plots.finish()


def test_writeback_evaluates_module_expression_from_creation_capture(
    saved_session: WritebackSession,
) -> None:
    session = saved_session
    initial = session.control.read_context()
    session.control.set_parameter(initial.target, "general.ext.flux_value", 0.5)
    captured = session.control.read_context()
    draft = session.service.create_draft(
        [
            ModuleWriteback(
                "Q1.pulses.pi",
                "Captured pulse",
                PulseSpec(
                    ch=0,
                    nqz=1,
                    freq=100.0,
                    gain=0.2,
                    waveform=ConstWaveformCfg(length=0.1),
                ),
            )
        ],
        parameters=captured,
        experiment_name="test/writeback",
    )
    draft.edit("ml-1", edits={("gain",): {"expr": "general.ext.flux_value"}})
    session.control.set_parameter(captured.target, "general.ext.flux_value", 0.9)
    session.service.apply_draft(draft)
    current = session.control.read_context()
    written = current.library.module_spec("Q1.pulses.pi")
    assert isinstance(written, PulseSpec) and written.gain == 0.5
    event = session.entry.ledger.events()[-1]
    assert event.kind == "accepted"
    assert isinstance(event.payload, AcceptedPayload)
    assert event.payload.writes[0].path == "Q1.pulses.pi"
    source = current.library.meta("Q1.pulses.pi.gain")
    assert source is not None and source.source == event.id


def test_append_failure_leaves_parameter_untouched(
    saved_session: WritebackSession, monkeypatch
) -> None:
    session = saved_session
    captured = session.control.read_context()
    draft = session.service.create_draft(
        [ParameterWriteback("general.ext.flux_value", "Value", proposed_value=2.0)],
        parameters=captured,
        experiment_name="test/writeback",
    )

    def fail_append(self: RecordsLedger, event: LedgerEvent, **_kwargs: object) -> None:
        raise OSError("ledger unavailable")

    monkeypatch.setattr(RecordsLedger, "append", fail_append)
    with pytest.raises(OSError, match="ledger unavailable"):
        session.service.apply_draft(draft)
    assert not session.control.read_context().lookup("general.ext.flux_value").present
    assert session.entry.ledger.events() == ()


def test_complex_writeback_preview_apply_and_disk_keep_captured_role_target(
    session: WritebackSession,
) -> None:
    session.entry.new_point("a")
    first = session.control.use_point(PointUseRequest("a"))
    session.control.add_component(
        "Q1",
        "fake/drive/a",
        {"ext": {"readout_cal": {"g_center": {"__complex__": [0.2, -0.1]}}}},
    )
    captured = session.control.read_context().bind_roles(
        {"driver": RoleSpec("fake/drive/*")}, focus="Q1", explicit={}
    )
    centers = (complex(3.125, -1.625), complex(-0.4, 0.1))
    tags = [{"__complex__": [value.real, value.imag]} for value in centers]
    draft = session.service.create_draft(
        [
            ParameterWriteback(
                "$driver.ext.readout_cal.g_center",
                "Ground center",
                proposed_value=centers[0],
                stderr=0.01,
            ),
            ParameterWriteback(
                "$driver.ext.readout_cal.e_center",
                "Excited center",
                proposed_value=centers[1],
                stderr=0.02,
            ),
        ],
        parameters=captured,
        experiment_name="test/writeback",
    )
    session.control.set_parameter(
        first, "Q1.ext.readout_cal.g_center", {"__complex__": [0.6, 0.1]}
    )
    session.entry.new_point("b")
    second = session.control.use_point(PointUseRequest("b"))
    session.control.add_component("Q2", "fake/drive/a", {})
    selected = session.control.read_context().document()
    preview = session.service.preview_values(draft)
    assert preview["param-1"].current == {"__complex__": [0.6, 0.1]}
    assert preview["param-2"].current is None
    assert [preview[item].actual_path for item in ("param-1", "param-2")] == [
        "Q1.ext.readout_cal.g_center",
        "Q1.ext.readout_cal.e_center",
    ]
    assert [preview[item].proposed for item in ("param-1", "param-2")] == tags
    assert session.entry.ledger.events() == ()
    session.service.apply_draft(draft, origin=CommandOrigin("mcp"))
    current = session.control.read_target(first, roles=captured.roles)
    paths = ("Q1.ext.readout_cal.g_center", "Q1.ext.readout_cal.e_center")
    assert [current.value(path) for path in paths] == tags
    assert (
        evaluate_scalar_expr("$driver.ext.readout_cal.g_center", current) == centers[0]
    )
    assert captured.value("$driver.ext.readout_cal.g_center") == {
        "__complex__": [0.2, -0.1]
    }
    assert session.control.read_context().target == second
    assert session.control.read_context().document() == selected
    event = session.entry.ledger.events()[-1]
    assert event.origin == "mcp"
    assert isinstance(event.payload, AcceptedPayload)
    assert [write.path for write in event.payload.writes] == list(paths)
    assert [write.value for write in event.payload.writes] == tags
    assert [document_meta(current.document(), path) for path in paths] == [
        {"stderr": 0.01},
        {"stderr": 0.02},
    ]
    reopened = ResultEntry.open(
        "entry",
        result_root=session.entry.result_path.parent,
        database_root=session.entry.database_path.parent,
    )
    document = snapshot(reopened.use_point("a"))
    assert [document_value(document, path) for path in paths] == tags


def test_receipt_reads_actual_role_target_after_point_change(
    session: WritebackSession,
) -> None:
    session.entry.new_point("a")
    first = session.control.use_point(PointUseRequest("a"))
    session.control.add_component("Q1", "fake/drive/a", {"rate": 0.1})
    captured = session.control.read_context().bind_roles(
        {"driver": RoleSpec("fake/drive/*")}, focus="Q1", explicit={}
    )
    draft = session.service.create_draft(
        [ParameterWriteback("$driver.rate", "Rate", proposed_value=0.2, stderr=0.05)],
        parameters=captured,
        run_id="original-run",
        experiment_name="test/writeback",
    )
    session.control.set_parameter(first, "Q1.rate", 0.6)
    session.entry.new_point("b")
    second = session.control.use_point(PointUseRequest("b"))
    session.control.add_component("Q2", "fake/drive/a", {"rate": 0.8})
    selected = session.control.read_context().document()
    assert session.service.preview_values(draft)["param-1"].current == 0.6
    draft.edit("param-1", selected=False)
    written = session.service.write_draft(
        draft, (WritebackEdit("param-1"),), origin=CommandOrigin("mcp")
    )
    assert written == [WritebackWritten("param-1", "parameter", "Q1.rate", 0.6, 0.2)]
    assert captured.value("$driver.rate") == 0.1
    assert session.control.read_context().target == second
    assert session.control.read_context().document() == selected
    assert not draft.items[0].selected
    assert session.service.get_all_applied(draft) == {"param-1": True}
    event = session.entry.ledger.events()[-1]
    assert event.origin == "mcp"
    target = session.control.read_target(first)
    assert document_meta(target.document(), "Q1.rate") == {"stderr": 0.05}
    assert isinstance(event.payload, AcceptedPayload)
    assert event.payload.run_id == draft.run_id == "original-run"


def test_library_receipt_reads_actual_raw_entry_and_uses_capture_for_proposal(
    saved_session: WritebackSession,
) -> None:
    session = saved_session
    initial = session.control.read_context()
    session.control.set_parameter(initial.target, "general.ext.flux_value", 0.2)
    captured = session.control.read_context()
    cfg = PulseSpec(
        ch=0, nqz=1, freq=100.0, gain=0.2, waveform=ConstWaveformCfg(length=0.1)
    )
    item = ModuleWriteback("Q1.pulses.pi", "Pulse", cfg, stderr=0.02)
    session.control.set_module(
        captured.target, item.target_name, cfg.model_copy(update={"gain": 0.6})
    )
    item_id, field = "ml-1", "gain"
    draft = session.service.create_draft(
        [item],
        parameters=captured,
        run_id="original-run",
        experiment_name="test/writeback",
    )
    draft.edit(item_id, edits={(field,): {"expr": "general.ext.flux_value"}})
    session.control.set_parameter(captured.target, "general.ext.flux_value", 9.0)
    preview = session.service.preview_values(draft)[item_id]
    assert isinstance(preview.current, dict) and preview.current[field] == 0.6
    assert isinstance(preview.proposed, dict) and preview.proposed[field] == 0.2
    receipt = session.service.write_draft(draft, (WritebackEdit(item_id),))[0]
    assert receipt.target == item.target_name
    assert isinstance(receipt.before, dict) and receipt.before[field] == 0.6
    assert isinstance(receipt.after, dict) and receipt.after[field] == 0.2
    source = session.control.read_context().library.meta(f"{item.target_name}.{field}")
    assert source is not None and source.stderr == item.stderr
    event = session.entry.ledger.events()[-1]
    assert isinstance(event.payload, AcceptedPayload)
    assert event.payload.run_id == source.run_id == draft.run_id == "original-run"


def test_each_parameter_write_replaces_metadata_with_its_own_stderr(
    saved_session: WritebackSession,
) -> None:
    session = saved_session
    target = session.control.read_context().target
    for path in ("Q1.rate", "Q1.coherence"):
        session.control.set_parameter(
            target, path, 1.0, metadata={"stderr": 9.0, "comment": "old"}
        )
    captured = session.control.read_context()
    draft = session.service.create_draft(
        [
            ParameterWriteback("Q1.rate", "Rate", proposed_value=0.2, stderr=0.01),
            ParameterWriteback(
                "Q1.duration", "Duration", proposed_value=3.0, stderr=0.4
            ),
            ParameterWriteback("Q1.coherence", "Unknown error", proposed_value=5.0),
        ],
        parameters=captured,
        experiment_name="test/writeback",
    )
    session.service.apply_draft(draft)
    document = session.control.read_context().document()
    assert [
        document_meta(document, path)
        for path in ("Q1.rate", "Q1.duration", "Q1.coherence")
    ] == [{"stderr": 0.01}, {"stderr": 0.4}, {}]
    event = session.entry.ledger.events()[-1]
    assert isinstance(event.payload, AcceptedPayload)
    assert event.payload.run_id is None
    assert draft.run_id is None


def test_duplicate_concrete_role_destinations_fail_before_accepted(
    session: WritebackSession,
) -> None:
    captured = session.control.read_context().bind_roles(
        {"driver": RoleSpec("fake/drive/*")}, focus="Q1", explicit={}
    )
    draft = session.service.create_draft(
        [
            ParameterWriteback("$driver.rate", "Role", proposed_value=0.2),
            ParameterWriteback("Q1.rate", "Direct", proposed_value=0.3),
        ],
        parameters=captured,
        experiment_name="test/writeback",
    )
    with pytest.raises(InvalidInputError, match="duplicate writeback destination"):
        session.service.apply_draft(draft)
    assert session.entry.ledger.events() == ()
    assert not session.control.read_context().lookup("Q1.rate").present


def test_empty_draft_reads_and_writes_without_context(
    session: WritebackSession,
) -> None:
    draft = session.service.create_draft(
        [], parameters=None, experiment_name="test/writeback"
    )
    assert session.service.preview_values(draft) == {}
    assert session.service.write_draft(draft, ()) == []
    assert session.service.apply_draft(draft)["applied_ids"] == []
    assert session.entry.ledger.events() == ()


def test_preview_proposed_data_is_isolated_at_validation_boundary(
    session: WritebackSession,
) -> None:
    draft = session.service.create_draft(
        [ParameterWriteback("Q1.ext.samples", "Samples", proposed_value=[1, 2])],
        parameters=session.control.read_context(),
        experiment_name="test/writeback",
    )
    preview = session.service.preview_values(draft)["param-1"]
    assert preview.current is None and preview.proposed == [1, 2]
    assert isinstance(preview.proposed, list)
    preview.proposed.append(3)
    assert session.service.preview_values(draft)["param-1"].proposed == [1, 2]
