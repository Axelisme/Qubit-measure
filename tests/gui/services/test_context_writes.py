"""Tests for ContextService ml/md writes — the single write authority (ADR-0006).

ml writes go through ``apply_ml_writes``, which registers the entries (lowered by
the app-injected ``lower_module`` / ``lower_waveform`` callbacks — here the real
``cfg_lowering`` ones), bumps "context", and emits at most one MD/ML_CHANGED per
batch. The CfgSchema lowering itself is experiment-coupled and lives app-side
(``cfg_lowering`` / the Controller's ContextWritePort façade).
"""

from __future__ import annotations

from zcu_tools.gui.app.main.adapter import CfgSchema, ContextReadiness
from zcu_tools.gui.app.main.cfg_schemas import (
    module_cfg_to_value,
    waveform_cfg_to_value,
)
from zcu_tools.gui.app.main.event_bus import EventBus
from zcu_tools.gui.app.main.services.cfg_lowering import lower_module, lower_waveform
from zcu_tools.gui.app.main.state import ExpContext, State
from zcu_tools.gui.session.services.context import ContextService
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
