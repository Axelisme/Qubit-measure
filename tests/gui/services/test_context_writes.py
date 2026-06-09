"""Tests for ContextService ml/md writes — the single write authority (ADR-0006).

Writes take an un-lowered CfgSchema (``set_ml_*_from_schema`` / ``apply_writes``);
ContextService lowers + registers + bumps "context" + emits. There is no public
raw-dict entry anymore.
"""

from __future__ import annotations

from zcu_tools.gui.app.main.adapter import CfgSchema, ContextReadiness
from zcu_tools.gui.app.main.cfg_schemas import (
    module_cfg_to_value,
    waveform_cfg_to_value,
)
from zcu_tools.gui.app.main.event_bus import EventBus
from zcu_tools.gui.app.main.io_manager import IOManager
from zcu_tools.gui.app.main.services.context import ContextService
from zcu_tools.gui.app.main.services.ports import ContextWrites
from zcu_tools.gui.app.main.state import ExpContext, State
from zcu_tools.meta_tool import MetaDict, ModuleLibrary

_READOUT_RAW = {
    "type": "readout/direct",
    "ro_ch": 0,
    "ro_freq": 6000.0,
    "ro_length": 1.0,
    "trig_offset": 0.0,
}
_WAVEFORM_RAW = {"style": "gauss", "length": 0.1, "sigma": 0.02}


def _module_schema(raw: dict) -> CfgSchema:
    spec, value = module_cfg_to_value(raw)
    return CfgSchema(spec=spec, value=value)


def _waveform_schema(raw: dict) -> CfgSchema:
    spec, value = waveform_cfg_to_value(raw)
    return CfgSchema(spec=spec, value=value)


def _make_svc_with_state() -> tuple[ContextService, State]:
    state = State(
        ExpContext(
            md=MetaDict(),
            ml=ModuleLibrary(),
            soc=None,
            soccfg=None,
            result_dir="",
            readiness=ContextReadiness.DRAFT,
        )
    )
    return ContextService(state, IOManager(), EventBus()), state


def _make_svc() -> ContextService:
    return _make_svc_with_state()[0]


def test_set_ml_module_from_schema_registers_module():
    svc = _make_svc()
    svc.set_ml_module_from_schema("readout_rf", _module_schema(_READOUT_RAW))
    assert "readout_rf" in svc.get_current_ml().modules


def test_set_ml_waveform_from_schema_registers_waveform():
    svc = _make_svc()
    svc.set_ml_waveform_from_schema("drive_wav", _waveform_schema(_WAVEFORM_RAW))
    assert "drive_wav" in svc.get_current_ml().waveforms


def test_md_write_bumps_context_version():
    # Concurrency guards on ``context`` (run.start / editor.commit / writeback)
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
    svc.set_ml_waveform_from_schema("drive_wav", _waveform_schema(_WAVEFORM_RAW))
    assert state.version.get("context") == before + 1
    svc.del_ml_waveform("drive_wav")
    assert state.version.get("context") == before + 2


def test_apply_writes_batch_is_one_bump():
    # A batch of md + ml writes lands as a single context bump (not N).
    svc, state = _make_svc_with_state()
    before = state.version.get("context")
    svc.apply_writes(
        ContextWrites(
            md={"r_f": 6000.0},
            ml_modules={"readout_rf": _module_schema(_READOUT_RAW)},
            ml_waveforms={"drive_wav": _waveform_schema(_WAVEFORM_RAW)},
        )
    )
    assert state.version.get("context") == before + 1
    ml = svc.get_current_ml()
    assert "readout_rf" in ml.modules
    assert "drive_wav" in ml.waveforms


def test_apply_writes_empty_is_noop():
    svc, state = _make_svc_with_state()
    before = state.version.get("context")
    svc.apply_writes(ContextWrites(md={}, ml_modules={}, ml_waveforms={}))
    assert state.version.get("context") == before


def test_apply_writes_emits_once_per_kind():
    svc = _make_svc()
    from zcu_tools.gui.session.events import MdChangedPayload, MlChangedPayload

    md_events = 0
    ml_events = 0

    def _on_md(_payload: object) -> None:
        nonlocal md_events
        md_events += 1

    def _on_ml(_payload: object) -> None:
        nonlocal ml_events
        ml_events += 1

    svc._bus.subscribe(MdChangedPayload, _on_md)
    svc._bus.subscribe(MlChangedPayload, _on_ml)
    svc.apply_writes(
        ContextWrites(
            md={"r_f": 6000.0, "rf_w": 1.0},
            ml_modules={"readout_rf": _module_schema(_READOUT_RAW)},
            ml_waveforms={"drive_wav": _waveform_schema(_WAVEFORM_RAW)},
        )
    )
    assert md_events == 1  # one MD_CHANGED for two md writes
    assert ml_events == 1  # one ML_CHANGED for module + waveform
