"""Tests for ContextService.set_ml_{module,waveform}_from_raw."""

from __future__ import annotations

import pytest
from zcu_tools.gui.adapter import ContextReadiness
from zcu_tools.gui.event_bus import EventBus
from zcu_tools.gui.io_manager import IOManager
from zcu_tools.gui.services.context import ContextService, MlEntryValidationError
from zcu_tools.gui.state import ExpContext, State
from zcu_tools.meta_tool import MetaDict, ModuleLibrary


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


def test_set_ml_module_from_raw_registers_module():
    svc = _make_svc()
    svc.set_ml_module_from_raw(
        "readout_rf",
        {
            "type": "readout/direct",
            "ro_ch": 0,
            "ro_freq": 6000.0,
            "ro_length": 1.0,
            "trig_offset": 0.0,
        },
    )
    ml = svc.get_current_ml()
    assert "readout_rf" in ml.modules


def test_set_ml_module_from_raw_invalid_wraps_as_validation_error():
    svc = _make_svc()
    with pytest.raises(MlEntryValidationError, match="Invalid module"):
        svc.set_ml_module_from_raw("bad", {"type": "unknown/junk"})


def test_set_ml_waveform_from_raw_registers_waveform():
    svc = _make_svc()
    svc.set_ml_waveform_from_raw(
        "drive_wav",
        {"style": "gauss", "length": 0.1, "sigma": 0.02},
    )
    ml = svc.get_current_ml()
    assert "drive_wav" in ml.waveforms


def test_set_ml_waveform_from_raw_invalid_wraps_as_validation_error():
    svc = _make_svc()
    with pytest.raises(MlEntryValidationError, match="Invalid waveform"):
        svc.set_ml_waveform_from_raw("bad", {"style": "no_such_style"})


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
    svc.set_ml_waveform_from_raw(
        "drive_wav",
        {"style": "gauss", "length": 0.1, "sigma": 0.02},
    )
    assert state.version.get("context") == before + 1
    svc.del_ml_waveform("drive_wav")
    assert state.version.get("context") == before + 2
