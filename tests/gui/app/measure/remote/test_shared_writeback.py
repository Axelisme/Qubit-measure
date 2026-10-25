"""Sample proposals and original cause projection through existing wire methods."""

from datetime import datetime
from pathlib import Path

import pytest
from pydantic import TypeAdapter
from zcu_tools.format_version import YamlValue
from zcu_tools.gui.app.measure import adapter as proposals
from zcu_tools.gui.app.measure.services.ports import AnalysisPaneSnapshot, TabSnapshot
from zcu_tools.gui.app.measure.writeback import (
    WritebackDraft,
    WritebackEdit,
    WritebackService,
)
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.expected_error import InvalidInputError
from zcu_tools.gui.remote.errors import ErrorCode, RemoteError
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.context_control import ContextControlFacet
from zcu_tools.gui.session.context_types import (
    ContextTarget,
    PointUseRequest,
    SavedPoint,
)
from zcu_tools.gui.session.services.context import ContextService
from zcu_tools.gui.session.state import SessionState
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.resources.entry import ResultEntry
from zcu_tools.resources.entry.writeback import ParameterUpdate
from zcu_tools.resources.waveform_assets import ArbWaveformDatabase

from ._helpers import dispatch_handler


class RejectingContext(ContextService):
    """A producer opt-in expected failure of the entire point group."""

    def set_parameters(
        self, target: ContextTarget, updates: tuple[ParameterUpdate, ...]
    ) -> tuple[YamlValue, ...]:
        if any(update.path == "general.ext.rejected" for update in updates):
            raise InvalidInputError("rejected fit", reason_code="fit_rejected")
        return super().set_parameters(target, updates)


class WritebackHarness:
    """Supply pane reads and commands to the existing handler test adapter."""

    render_view = None

    def __init__(
        self, context: ContextService, service: WritebackService, draft: WritebackDraft
    ) -> None:
        self.context = context
        self.service = service
        self.draft = draft

    def has_tab(self, tab_id: str) -> bool:
        return tab_id == "tab"

    def get_session_env(self) -> SessionEnv:
        return self.context.get_session_env()

    def get_tab_snapshot(self, tab_id: str) -> TabSnapshot:
        return TabSnapshot(
            adapter_name="test/sample",
            cfg_input={},
            tab_id=tab_id,
            capabilities=proposals.AdapterCapabilities(
                analysis=proposals.AnalysisMode.FIT
            ),
            analysis=AnalysisPaneSnapshot(
                params=None,
                result=None,
                figures=None,
                writeback_items=self.draft.items,
                has_writeback_draft=True,
                unavailable_reasons=self.draft.unavailable_reasons,
            ),
        )

    def get_writeback_values_for_pane(
        self, tab_id: str, pane: str, *, session_id: str | None = None
    ):
        return self.service.preview_values(self.draft, session_id=session_id)

    def write_writeback_for_pane(
        self, tab_id: str, pane: str, changes: tuple[WritebackEdit, ...]
    ):
        return self.service.write_draft(self.draft, changes)


def context_at(
    tmp_path: Path, *, rejecting: bool = False
) -> tuple[ResultEntry, ContextService]:
    entry = ResultEntry.create(
        "entry", result_root=tmp_path / "results", database_root=tmp_path / "database"
    )
    entry.new_point("a")
    context_type = RejectingContext if rejecting else ContextService
    context = context_type(
        SessionState(SessionEnv(parameters=None, soc=None, soccfg=None)),
        BaseEventBus(),
        bind_assets=ArbWaveformDatabase.info_at,
    )
    context.install_entry(entry)
    context.use_point(PointUseRequest("a"))
    return entry, context


def test_sample_write_receipt_reads_latest_jsonl_before_and_after_at_fixed_point(
    tmp_path: Path,
) -> None:
    entry, context = context_at(tmp_path)
    context.set_alias_rows((AliasRow("flux_dev", "bias_at_run"),))
    service = WritebackService(
        ContextControlFacet(context=context), arb_names=lambda: ()
    )
    draft = service.create_draft(
        [
            proposals.RecordWriteback(
                "Q1",
                "record",
                record={"t1_us": 20.09},
            )
        ],
        parameters=context.read_context(),
        experiment_name="test/t1",
        run_id="run-fixed",
        device_snapshot={"bias_at_run": {"value": 0.00125, "value_unit": "A"}},
    )
    entry.new_point("b")
    context.use_point(PointUseRequest("b"))
    harness = WritebackHarness(context, service, draft)
    params = {"tab_id": "tab", "subtab_id": "analysis", "write": [{"id": "record-1"}]}
    first = dispatch_handler(harness, "tab.writeback_write", params)
    row = entry.records("Q1").latest("run-fixed")
    assert row is not None
    assert first == {
        "written": [
            {
                "id": "record-1",
                "kind": "record",
                "target": "Q1",
                "before": None,
                "after": row,
            }
        ]
    }
    assert row == {
        "t1_us": 20.09,
        "run_id": "run-fixed",
        "point": "a",
        "at": row["at"],
        "source": entry.ledger.events()[-1].id,
    }
    entry.records("Q1").append(run_id="run-fixed", point="a", t1_us=35.0)
    latest = entry.records("Q1").latest("run-fixed")
    second = dispatch_handler(harness, "tab.writeback_write", params)
    updated = entry.records("Q1").latest("run-fixed")
    assert updated is not None
    assert second == {
        "written": [
            {
                "id": "record-1",
                "kind": "record",
                "target": "Q1",
                "before": latest,
                "after": updated,
            }
        ]
    }
    assert updated == row | {
        "at": updated["at"],
        "source": entry.ledger.events()[-1].id,
    }
    assert updated["source"] != row["source"]
    assert context.read_context().target == SavedPoint(
        entry_id=entry.entry_id, point="b"
    )
    assert service.get_all_applied(draft) == {"record-1": True}


def test_record_preview_selection_and_apply_preserve_adapter_json(
    tmp_path: Path,
) -> None:
    entry, context = context_at(tmp_path)
    context.set_alias_rows((AliasRow("flux_dev", "bias_at_run"),))
    capture = context.read_context()
    service = WritebackService(
        ContextControlFacet(context=context), arb_names=lambda: ()
    )
    draft = service.create_draft(
        [
            proposals.ParameterWriteback(
                "general.description", "skip", proposed_value="unselected"
            ),
            proposals.RecordWriteback(
                "Q1",
                "record",
                record={"t1_us": 20.09},
            ),
        ],
        parameters=capture,
        experiment_name="test/t1",
        run_id="run-fixed",
        device_snapshot={"bias_at_run": {"value": 0.00125, "value_unit": "A"}},
    )
    context.set_alias_rows((AliasRow("flux_dev", "later_device"),))
    entry.new_point("b")
    context.use_point(PointUseRequest("b"))
    harness = WritebackHarness(context, service, draft)
    params = {"tab_id": "tab", "subtab_id": "analysis"}
    preview = dispatch_handler(harness, "tab.writeback_preview", params)
    expected = {"t1_us": 20.09}
    expected_preview = [
        {
            "id": "param-1",
            "kind": "parameter",
            "target_name": "general.description",
            "description": "skip",
            "selected": True,
            "proposed_value": "unselected",
            "actual_path": "general.description",
            "stderr": None,
            "current": None,
            "proposed": "unselected",
        },
        {
            "id": "record-1",
            "kind": "record",
            "target_name": "Q1",
            "description": "record",
            "selected": True,
            "proposed_value": expected,
            "actual_path": "Q1",
            "stderr": None,
            "current": None,
            "proposed": expected,
        },
    ]
    expected_preview = [row | {"note": None} for row in expected_preview]
    assert preview["items"] == expected_preview
    service.edit_draft(draft, "param-1", selected=False)
    receipt = dispatch_handler(
        harness, "tab.writeback_write", params | {"write": [{"id": "record-1"}]}
    )
    written = TypeAdapter(list[dict[str, object]]).validate_python(receipt["written"])
    assert [item["id"] for item in written] == ["record-1"]
    assert entry.use_point("a").description is None
    row = entry.records("Q1").read().iloc[0].to_dict()
    event = entry.ledger.events()[-1]
    assert datetime.fromisoformat(row["at"]) == datetime.fromisoformat(event.at)
    assert row == expected | {
        "run_id": "run-fixed",
        "point": "a",
        "at": row["at"],
        "source": event.id,
    }
    assert context.read_context().target == SavedPoint(
        entry_id=entry.entry_id, point="b"
    )
    assert service.get_all_applied(draft) == {"param-1": False, "record-1": True}
    assert dispatch_handler(harness, "tab.writeback_preview", params)["items"] == [
        expected_preview[0] | {"selected": False},
        expected_preview[1] | {"current": row},
    ]


def test_preview_displays_adapter_note_and_current_manual_selection(
    tmp_path: Path,
) -> None:
    _entry, context = context_at(tmp_path)
    original = proposals.ParameterWriteback("general.ext.fit", "fit", 5.0)
    original.selected = False
    original.note = "Fit needs manual review"
    service = WritebackService(
        ContextControlFacet(context=context), arb_names=lambda: ()
    )
    draft = service.create_draft(
        [original],
        parameters=context.read_context(),
        experiment_name="test/review",
    )
    harness = WritebackHarness(context, service, draft)
    params = {"tab_id": "tab", "subtab_id": "analysis"}
    preview = dispatch_handler(harness, "tab.writeback_preview", params)
    rows = TypeAdapter(list[dict[str, object]]).validate_python(preview["items"])
    assert rows[0]["selected"] is False
    assert rows[0]["note"] == "Fit needs manual review"
    service.edit_draft(draft, "param-1", selected=True)
    selected = dispatch_handler(harness, "tab.writeback_preview", params)
    assert selected["items"] == [rows[0] | {"selected": True}]
    dispatch_handler(
        harness, "tab.writeback_write", params | {"write": [{"id": "param-1"}]}
    )
    applied = dispatch_handler(harness, "tab.writeback_preview", params)
    assert applied["items"] == [rows[0] | {"selected": True, "current": 5.0}]
    assert not original.selected


def test_partial_failure_preserves_expected_cause_wire_category(tmp_path: Path) -> None:
    entry, context = context_at(tmp_path, rejecting=True)
    service = WritebackService(
        ContextControlFacet(context=context), arb_names=lambda: ()
    )
    draft = service.create_draft(
        [
            proposals.ParameterWriteback(
                "general.description", "first", proposed_value="retained"
            ),
            proposals.ParameterWriteback(
                "general.ext.rejected", "invalid", proposed_value=1
            ),
            proposals.ParameterWriteback("general.ext.later", "last", proposed_value=2),
        ],
        parameters=context.read_context(),
        experiment_name="test/expected",
    )
    harness = WritebackHarness(context, service, draft)
    before_description = entry.use_point("a").description
    with pytest.raises(RemoteError) as caught:
        dispatch_handler(
            harness,
            "tab.writeback_write",
            {
                "tab_id": "tab",
                "subtab_id": "analysis",
                "write": [{"id": f"param-{i}"} for i in (1, 2, 3)],
            },
        )
    error = caught.value
    assert (error.code, error.message, error.reason, error.data) == (
        ErrorCode.INVALID_PARAMS,
        "rejected fit",
        "fit_rejected",
        None,
    )
    assert service.get_all_applied(draft) == {
        "param-1": False,
        "param-2": False,
        "param-3": False,
    }
    assert entry.use_point("a").description == before_description
    assert not context.read_context().lookup("general.ext.rejected").present
    assert not context.read_context().lookup("general.ext.later").present
    assert entry.ledger.events()[-1].kind == "accepted"
