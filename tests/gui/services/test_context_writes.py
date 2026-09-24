"""Tests for ContextService ml/md writes — the single write authority (ADR-0006).

ml writes go through ``apply_ml_writes``, which registers the entries (lowered by
the app-injected ``lower_module`` / ``lower_waveform`` callbacks — here the real
``cfg_lowering`` ones), bumps "context", and emits at most one MD/ML_CHANGED per
batch. The CfgSchema lowering itself is experiment-coupled and lives app-side
(``cfg_lowering`` / the Controller's ContextWritePort façade).
"""

from __future__ import annotations

import pytest
from zcu_tools.gui.app.main.adapter import ContextReadiness
from zcu_tools.gui.app.main.cfg_schemas import (
    module_cfg_to_value,
    waveform_cfg_to_value,
)
from zcu_tools.gui.app.main.services.cfg_lowering import lower_module, lower_waveform
from zcu_tools.gui.app.main.state import ExpContext, State
from zcu_tools.gui.cfg import CfgSchema
from zcu_tools.gui.event_bus import BaseEventBus as EventBus
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.session.services.context import (
    ContextService,
    MlEntryValidationError,
)
from zcu_tools.gui.session.services.io_manager import IOManager
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


def _apply(
    svc: ContextService,
    *,
    md: dict | None = None,
    modules: dict | None = None,
    waveforms: dict | None = None,
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
    _apply(
        svc,
        md={"r_f": 6000.0, "rf_w": 1.0},
        modules={"readout_rf": _module_schema(_READOUT_RAW)},
        waveforms={"drive_wav": _waveform_schema(_WAVEFORM_RAW)},
    )
    assert md_events == 1  # one MD_CHANGED for two md writes
    assert ml_events == 1  # one ML_CHANGED for module + waveform


def test_replace_ml_module_is_one_atomic_content_mutation():
    svc, state = _make_svc_with_state()
    _apply(svc, modules={"readout_rf": _module_schema(_READOUT_RAW)}, dump=False)
    ml_events = 0

    from zcu_tools.gui.session.events import MlChangedPayload

    def _on_ml(_payload: object) -> None:
        nonlocal ml_events
        ml_events += 1

    svc._bus.subscribe(MlChangedPayload, _on_ml)
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
