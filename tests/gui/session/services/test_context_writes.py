"""Tests for ContextService ml/md writes — the single write authority (ADR-0067).

ml writes go through ``apply_ml_writes``, which registers the entries (lowered by
the app-injected ``lower_module`` / ``lower_waveform`` callbacks — here the real
``cfg_lowering`` ones), bumps "context", and emits at most one MD/ML_CHANGED per
batch. The CfgSchema lowering itself is experiment-coupled and lives app-side
(``cfg_lowering`` / the Controller's ContextWritePort façade).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest
from zcu_tools.gui.app.measure.adapter import ContextReadiness
from zcu_tools.gui.app.measure.cfg_schemas import (
    module_cfg_to_value,
    waveform_cfg_to_value,
)
from zcu_tools.gui.app.measure.services.cfg_lowering import lower_module, lower_waveform
from zcu_tools.gui.app.measure.state import SessionEnv, State
from zcu_tools.gui.cfg import CfgSchema
from zcu_tools.gui.event_bus import BaseEventBus as EventBus
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.session.events import MdChangedPayload, MlChangedPayload
from zcu_tools.gui.session.services.context import (
    ContextService,
    MlEntryValidationError,
)
from zcu_tools.gui.session.services.io_manager import IOManager
from zcu_tools.resources.context import MetaDict, ModuleLibrary

_READOUT_RAW = {
    "type": "readout/direct",
    "ro_ch": 0,
    "ro_freq": 6000.0,
    "ro_length": 1.0,
    "trig_offset": 0.0,
}
_WAVEFORM_RAW = {"style": "gauss", "length": 0.1, "sigma": 0.02}


def _module_schema(raw: dict[str, Any]) -> CfgSchema:
    spec, value = module_cfg_to_value(raw)
    return CfgSchema(spec=spec, value=value)


def _waveform_schema(raw: dict[str, Any]) -> CfgSchema:
    spec, value = waveform_cfg_to_value(raw)
    return CfgSchema(spec=spec, value=value)


def _make_svc_with_state(bus: EventBus | None = None) -> tuple[ContextService, State]:
    state = State(
        SessionEnv(
            md=MetaDict(),
            ml=ModuleLibrary(),
            soc=None,
            soccfg=None,
            result_dir="",
            readiness=ContextReadiness.DRAFT,
        )
    )
    return ContextService(
        state, IOManager(), bus if bus is not None else EventBus()
    ), state


def _make_svc() -> ContextService:
    return _make_svc_with_state()[0]


def _apply(
    svc: ContextService,
    *,
    md: dict[str, Any] | None = None,
    modules: dict[str, Any] | None = None,
    waveforms: dict[str, Any] | None = None,
    dump: bool = True,
) -> None:
    svc.apply_ml_writes(
        md or {},
        modules or {},
        waveforms or {},
        lower_module=lower_module,
        lower_waveform=lower_waveform,
        dump=dump,
    )


@pytest.mark.parametrize("failure_stage", ["module", "waveform"])
def test_failed_late_preparation_keeps_live_context_and_version(failure_stage):
    bus = EventBus()
    svc, state = _make_svc_with_state(bus)
    events: list[object] = []
    bus.subscribe(MdChangedPayload, events.append)
    bus.subscribe(MlChangedPayload, events.append)
    svc.get_current_md().update(offset=1.0)
    before = state.version.get("context")

    def reject(entry, library, metadata):
        assert metadata.offset == 9.0
        assert svc.get_current_md().offset == 1.0
        assert svc.get_current_ml().modules == {}
        raise MlEntryValidationError("injected later preparation failure")

    with pytest.raises(MlEntryValidationError, match="later preparation"):
        svc.apply_ml_writes(
            {"offset": 9.0},
            {"first": _module_schema(_READOUT_RAW)},
            {"later": _waveform_schema(_WAVEFORM_RAW)},
            lower_module=reject if failure_stage == "module" else lower_module,
            lower_waveform=reject,
            dump=False,
        )
    assert svc.get_current_md().offset == 1.0
    assert svc.get_current_ml().modules == {}
    assert svc.get_current_ml().waveforms == {}
    assert state.version.get("context") == before
    assert events == []


def test_candidate_lowering_sees_earlier_writes_without_publishing_them():
    svc, state = _make_svc_with_state()
    library = svc.get_current_ml()
    metadata = svc.get_current_md()
    metadata.update(offset=1.0)

    def lower(entry, candidate_ml, candidate_md):
        assert candidate_md.offset == 9.0
        assert metadata.offset == 1.0
        assert library.modules == {}
        if entry == "second":
            assert "first" in candidate_ml.modules
        return _READOUT_RAW

    svc.apply_ml_writes(
        {"offset": 9.0},
        {"first": "first", "second": "second"},
        {},
        lower_module=lower,
        lower_waveform=lower_waveform,
        dump=False,
    )
    assert svc.get_current_md() is metadata
    assert svc.get_current_ml() is library
    assert metadata.offset == 9.0
    assert set(library.modules) == {"first", "second"}
    assert state.version.get("context") == 1


def test_storage_failure_reports_applied_and_preserves_published_batch(
    tmp_path, monkeypatch, caplog
):
    bus = EventBus()
    svc, state = _make_svc_with_state(bus)
    events: list[tuple[int, bool]] = []

    def observe(_payload: object) -> None:
        events.append((state.version.get("context"), "first" in library.modules))

    bus.subscribe(MdChangedPayload, observe)
    bus.subscribe(MlChangedPayload, observe)
    library = ModuleLibrary(tmp_path / "modules.yaml")
    state.set_context(replace(state.session_env, ml=library))
    defect = OSError("disk full")
    saves = []

    def fail_save():
        saves.append(state.version.get("context"))
        raise defect

    monkeypatch.setattr(library, "dump", fail_save)
    with pytest.raises(RuntimeError, match="applied, but saving failed") as caught:
        _apply(svc, md={"offset": 9.0}, modules={"first": _module_schema(_READOUT_RAW)})
    assert caught.value.__cause__ is defect
    assert svc.get_current_md().offset == 9.0
    assert "first" in library.modules
    assert state.version.get("context") == 1
    assert saves == [1]
    assert events == [(1, True), (1, True)]
    assert "Context settings applied, but saving failed" in caplog.text


def test_apply_ml_writes_registers_module():
    svc = _make_svc()
    _apply(svc, modules={"readout_rf": _module_schema(_READOUT_RAW)}, dump=False)
    assert "readout_rf" in svc.get_current_ml().modules


def test_apply_ml_writes_registers_waveform():
    svc = _make_svc()
    _apply(svc, waveforms={"drive_wav": _waveform_schema(_WAVEFORM_RAW)}, dump=False)
    assert "drive_wav" in svc.get_current_ml().waveforms


def test_md_write_bumps_context_version():
    # Concurrency guards on ``context`` (tab.run_start / editor.commit / tab.writeback_apply)
    # must detect md edits: a semantic md write bumps the context version.
    svc, state = _make_svc_with_state()
    before = state.version.get("context")
    svc.set_md_attr("r_f", 6000.0)
    assert state.version.get("context") == before + 1
    svc.del_md_attr("r_f")
    assert state.version.get("context") == before + 2


def test_ml_write_bumps_context_version():
    svc, state = _make_svc_with_state()
    before = state.version.get("context")
    _apply(svc, waveforms={"drive_wav": _waveform_schema(_WAVEFORM_RAW)}, dump=False)
    assert state.version.get("context") == before + 1
    svc.del_ml_waveform("drive_wav")
    assert state.version.get("context") == before + 2


def test_apply_ml_writes_batch_is_one_bump():
    # A batch of md + ml writes lands as a single context bump (not N).
    svc, state = _make_svc_with_state()
    before = state.version.get("context")
    _apply(
        svc,
        md={"r_f": 6000.0},
        modules={"readout_rf": _module_schema(_READOUT_RAW)},
        waveforms={"drive_wav": _waveform_schema(_WAVEFORM_RAW)},
    )
    assert state.version.get("context") == before + 1
    ml = svc.get_current_ml()
    assert "readout_rf" in ml.modules
    assert "drive_wav" in ml.waveforms


def test_apply_ml_writes_empty_is_noop():
    svc, state = _make_svc_with_state()
    before = state.version.get("context")
    _apply(svc)
    assert state.version.get("context") == before


def test_apply_ml_writes_emits_once_per_kind():
    bus = EventBus()
    svc, _ = _make_svc_with_state(bus)
    md_events = 0
    ml_events = 0

    def _on_md(_payload: object) -> None:
        nonlocal md_events
        md_events += 1

    def _on_ml(_payload: object) -> None:
        nonlocal ml_events
        ml_events += 1

    bus.subscribe(MdChangedPayload, _on_md)
    bus.subscribe(MlChangedPayload, _on_ml)
    _apply(
        svc,
        md={"r_f": 6000.0, "rf_w": 1.0},
        modules={"readout_rf": _module_schema(_READOUT_RAW)},
        waveforms={"drive_wav": _waveform_schema(_WAVEFORM_RAW)},
    )
    assert md_events == 1  # one MD_CHANGED for two md writes
    assert ml_events == 1  # one ML_CHANGED for module + waveform


def test_replace_ml_module_is_one_atomic_content_mutation():
    bus = EventBus()
    svc, state = _make_svc_with_state(bus)
    _apply(svc, modules={"readout_rf": _module_schema(_READOUT_RAW)}, dump=False)
    ml_events = 0

    def _on_ml(_payload: object) -> None:
        nonlocal ml_events
        ml_events += 1

    bus.subscribe(MlChangedPayload, _on_ml)
    before = state.version.get("context")
    replacement = dict(_READOUT_RAW, ro_freq=6123.0)
    svc.replace_ml_module_from_schema(
        "readout_rf",
        "readout_v2",
        _module_schema(replacement),
        lower_module=lower_module,
        lower_waveform=lower_waveform,
        dump=False,
    )

    assert "readout_rf" not in svc.get_current_ml().modules
    assert svc.get_current_ml().modules["readout_v2"].to_dict()["ro_freq"] == 6123.0
    assert state.version.get("context") == before + 1
    assert ml_events == 1


def test_replace_ml_waveform_uses_waveform_lowering_and_store():
    svc, state = _make_svc_with_state()
    _apply(svc, waveforms={"drive_wav": _waveform_schema(_WAVEFORM_RAW)}, dump=False)
    before = state.version.get("context")

    svc.replace_ml_waveform_from_schema(
        "drive_wav",
        "drive_wav_v2",
        _waveform_schema({**_WAVEFORM_RAW, "length": 0.2}),
        lower_module=lower_module,
        lower_waveform=lower_waveform,
    )

    assert "drive_wav" not in svc.get_current_ml().waveforms
    assert svc.get_current_ml().waveforms["drive_wav_v2"].to_dict()["length"] == 0.2
    assert state.version.get("context") == before + 1


def test_replace_ml_collision_and_lowering_failure_leave_live_content_intact():
    svc, state = _make_svc_with_state()
    _apply(
        svc,
        modules={
            "readout_rf": _module_schema(_READOUT_RAW),
            "other": _module_schema(dict(_READOUT_RAW, ro_freq=6100.0)),
        },
        dump=False,
    )
    original = svc.get_current_ml().modules["readout_rf"].to_dict()
    before = state.version.get("context")

    with pytest.raises(FailedPreconditionError, match="already exists"):
        svc.replace_ml_module_from_schema(
            "readout_rf",
            "other",
            _module_schema(dict(_READOUT_RAW, ro_freq=6200.0)),
            lower_module=lower_module,
            lower_waveform=lower_waveform,
            dump=False,
        )
    assert svc.get_current_ml().modules["readout_rf"].to_dict() == original
    assert state.version.get("context") == before

    def _fail(*_args: object) -> object:
        raise MlEntryValidationError("bad cfg")

    with pytest.raises(MlEntryValidationError, match="bad cfg"):
        svc.replace_ml_module_from_schema(
            "readout_rf",
            "readout_v2",
            _module_schema(dict(_READOUT_RAW, ro_freq=6300.0)),
            lower_module=_fail,
            lower_waveform=lower_waveform,
            dump=False,
        )
    assert svc.get_current_ml().modules["readout_rf"].to_dict() == original
    assert "readout_v2" not in svc.get_current_ml().modules
    assert state.version.get("context") == before
