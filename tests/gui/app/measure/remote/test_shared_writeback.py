"""Sample proposals and original cause projection through existing wire methods."""

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import pytest
from pydantic import TypeAdapter
from zcu_tools.format_version import YamlMap, YamlValue
from zcu_tools.gui.app.measure import adapter as proposals
from zcu_tools.gui.app.measure.services.ports import (
    AnalysisPaneSnapshot,
    CfgEdit,
    CfgEditorPort,
    CfgEditResult,
    TabSnapshot,
)
from zcu_tools.gui.app.measure.services.writeback import (
    WritebackDraft,
    WritebackEdit,
    WritebackService,
)
from zcu_tools.gui.cfg import CfgSchema
from zcu_tools.gui.cfg.binding import CfgDraft, SettableTarget
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
from zcu_tools.gui.session.types import ParameterContext, SessionEnv
from zcu_tools.resources.entry import ResultEntry
from zcu_tools.resources.waveform_assets import ArbWaveformDatabase

from ._helpers import dispatch_handler


class NoCfgEditors(CfgEditorPort):
    """The parameter/sample tests do not open cfg-editor sessions."""

    def open_seeded(
        self,
        seed: CfgSchema,
        *,
        parameters: ParameterContext | None,
        gc: bool = False,
        owner_key: str | None = None,
    ) -> tuple[str, tuple[SettableTarget, ...]]:
        raise AssertionError("unexpected cfg editor")

    def teardown(self, editor_id: str, *, reason: str = "discarded") -> None:
        raise AssertionError("unexpected cfg teardown")

    def get_draft(self, editor_id: str) -> CfgDraft:
        raise AssertionError("unexpected cfg draft")

    def set_field(self, editor_id: str, path: str, value: object) -> CfgEditResult:
        raise AssertionError("unexpected cfg edit")

    def set_fields(
        self, editor_id: str, edits: Sequence[CfgEdit], *, agent_edit: bool = False
    ) -> CfgEditResult:
        raise AssertionError("unexpected cfg edits")


class RejectingContext(ContextService):
    """A producer opt-in expected failure after a real committed prefix."""

    def set_parameter(
        self,
        target: ContextTarget,
        path: str,
        value: YamlValue,
        *,
        metadata: YamlMap | None = None,
    ) -> None:
        if path == "general.ext.rejected":
            raise InvalidInputError("rejected fit", reason_code="fit_rejected")
        super().set_parameter(target, path, value, metadata=metadata)


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

    def get_context_version(self) -> int:
        return 1

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

    def set_writeback_item_for_pane(
        self,
        tab_id: str,
        pane: str,
        session_id: str,
        *,
        selected: bool | None = None,
        target_name: str | None = None,
    ):
        return self.service.edit_draft(
            self.draft, session_id, selected=selected, target_name=target_name
        )

    def apply_writeback_for_pane(
        self, tab_id: str, pane: str, *, item_ids: tuple[str, ...] | None = None
    ):
        return self.service.apply_draft(self.draft, item_ids=item_ids)

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
    service = WritebackService(NoCfgEditors(), ContextControlFacet(context=context))
    draft = service.create_draft(
        [proposals.SampleWriteback("Q1", "sample", proposed_value={"t1_us": 20.09})],
        parameters=context.read_context(),
        experiment_name="test/t1",
        run_id="run-fixed",
        device_snapshot={"bias_at_run": {"value": 0.00125, "value_unit": "A"}},
    )
    entry.new_point("b")
    context.use_point(PointUseRequest("b"))
    harness = WritebackHarness(context, service, draft)
    params = {"tab_id": "tab", "subtab_id": "analysis", "write": [{"id": "sample-1"}]}
    first = dispatch_handler(harness, "tab.writeback_write", params)
    row = entry.samples("Q1").latest("run-fixed")
    assert row is not None
    assert first == {
        "written": [
            {
                "id": "sample-1",
                "kind": "sample",
                "target": "Q1",
                "before": None,
                "after": row,
            }
        ]
    }
    assert row == {
        "t1_us": 20.09,
        "dev_value": 0.00125,
        "dev_unit": "A",
        "run_id": "run-fixed",
        "point": "a",
        "at": row["at"],
        "source": entry.ledger.events()[-1].id,
    }
    entry.samples("Q1").append(run_id="run-fixed", point="a", t1_us=35.0)
    latest = entry.samples("Q1").latest("run-fixed")
    second = dispatch_handler(harness, "tab.writeback_write", params)
    updated = entry.samples("Q1").latest("run-fixed")
    assert updated is not None
    assert second == {
        "written": [
            {
                "id": "sample-1",
                "kind": "sample",
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
    assert service.get_all_applied(draft) == {"sample-1": True}


def test_sample_preview_selection_and_apply_use_run_device_snapshot(
    tmp_path: Path,
) -> None:
    entry, context = context_at(tmp_path)
    context.set_alias_rows((AliasRow("flux_dev", "bias_at_run"),))
    capture = context.read_context()
    service = WritebackService(NoCfgEditors(), ContextControlFacet(context=context))
    draft = service.create_draft(
        [
            proposals.ParameterWriteback(
                "general.description", "skip", proposed_value="unselected"
            ),
            proposals.SampleWriteback("Q1", "sample", proposed_value={"t1_us": 20.09}),
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
    expected = {"t1_us": 20.09, "dev_value": 0.00125, "dev_unit": "A"}
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
            "id": "sample-1",
            "kind": "sample",
            "target_name": "Q1",
            "description": "sample",
            "selected": True,
            "proposed_value": expected,
            "actual_path": "Q1",
            "stderr": None,
            "current": None,
            "proposed": expected,
        },
    ]
    assert preview["items"] == expected_preview
    assert preview["unavailable_reasons"] == []
    dispatch_handler(
        harness, "tab.writeback_set", params | {"id": "param-1", "selected": False}
    )
    receipt = dispatch_handler(harness, "tab.writeback_apply", params)
    assert receipt["applied_ids"] == ["sample-1"]
    assert receipt["written"] == {
        "parameter": [],
        "ml_modules": [],
        "ml_waveforms": [],
        "samples": ["Q1"],
    }
    assert entry.use_point("a").description is None
    row = entry.samples("Q1").read().iloc[0].to_dict()
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
    assert service.get_all_applied(draft) == {"param-1": False, "sample-1": True}
    assert dispatch_handler(harness, "tab.writeback_preview", params)["items"] == [
        expected_preview[0] | {"selected": False},
        expected_preview[1] | {"current": row},
    ]


@pytest.mark.parametrize(
    "bound,snapshot,reason",
    [
        (False, {"bias_at_run": {"value": 1.0, "value_unit": "V"}}, "$flux_dev"),
        (True, {}, "bias_at_run"),
        (True, {"bias_at_run": {"value_unit": "V"}}, "value"),
        (True, {"bias_at_run": {"value": 1.0}}, "value_unit"),
    ],
)
def test_sample_omission_reason_is_projected_without_a_sample(
    tmp_path: Path, bound, snapshot, reason
) -> None:
    _entry, context = context_at(tmp_path)
    if bound:
        context.set_alias_rows((AliasRow("flux_dev", "bias_at_run"),))
    service = WritebackService(NoCfgEditors(), ContextControlFacet(context=context))
    draft = service.create_draft(
        [
            proposals.SampleWriteback(
                "Q1", "sample", proposed_value={"freq_mhz": 1250.0}
            )
        ],
        parameters=context.read_context(),
        experiment_name="test/freq",
        run_id="run-fixed",
        device_snapshot=snapshot,
    )
    assert draft.items == ()
    preview = dispatch_handler(
        WritebackHarness(context, service, draft),
        "tab.writeback_preview",
        {"tab_id": "tab", "subtab_id": "analysis"},
    )
    assert preview["items"] == []
    reasons = TypeAdapter(list[str]).validate_python(
        preview["unavailable_reasons"], strict=True
    )
    assert len(reasons) == 1
    assert reason in reasons[0]


def test_partial_failure_preserves_expected_cause_wire_category(tmp_path: Path) -> None:
    entry, context = context_at(tmp_path, rejecting=True)
    service = WritebackService(NoCfgEditors(), ContextControlFacet(context=context))
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
    with pytest.raises(RemoteError) as caught:
        dispatch_handler(
            harness, "tab.writeback_apply", {"tab_id": "tab", "subtab_id": "analysis"}
        )
    error = caught.value
    assert (error.code, error.message, error.reason, error.data) == (
        ErrorCode.INVALID_PARAMS,
        "rejected fit",
        "fit_rejected",
        None,
    )
    assert service.get_all_applied(draft) == {
        "param-1": True,
        "param-2": False,
        "param-3": False,
    }
    assert entry.use_point("a").description == "retained"
    assert not context.read_context().lookup("general.ext.later").present
    assert entry.ledger.events()[-1].kind == "accepted"
