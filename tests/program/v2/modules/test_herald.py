"""Herald reset contracts at native cfg validation and timeline build."""

import pytest
from pydantic import TypeAdapter
from qick.asm_v2 import QickParam, QickSweep1D
from zcu_tools.cfg_model import CircleClassifier, LinearClassifier, ReadoutClassifier
from zcu_tools.program.v2 import TimelineBuilder
from zcu_tools.program.v2.modules import ModuleCfgFactory, Reset
from zcu_tools.program.v2.modules.readout import PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg

from tests.program._profiles import load_soccfg
from tests.program.v2._mux_support import mux_soccfg, native_mux


def pulse(classifier: ReadoutClassifier | None = None) -> PulseReadoutCfg:
    return PulseReadoutCfg(
        gen_ch=0,
        nqz=1,
        freq=100.0,
        gain=0.1,
        ro_ch=0,
        ro_length=0.1,
        post_delay=0.1,
        waveform=ConstWaveformCfg(length=0.1),
        classifier=classifier,
    )


@pytest.mark.parametrize(
    ("kind", "readout", "keep"),
    [("reset/herald", pulse(), "g"), ("reset/mux_herald", native_mux(), {"0": "g"})],
)
def test_plain_reset_cfg_rejects_herald(kind: str, readout: object, keep: object):
    with pytest.raises(ValueError, match="union_tag_invalid"):
        TypeAdapter(Reset.Cfg).validate_python(
            {"type": kind, "readout": readout, "keep": keep}
        )


def test_herald_cfg_rejects_herald_preparation():
    nested = {"type": "reset/herald", "readout": pulse(), "keep": "g"}
    with pytest.raises(ValueError, match=r"prepare.*\n.*union_tag_invalid"):
        ModuleCfgFactory.from_raw({**nested, "prepare": nested})


@pytest.mark.parametrize(
    ("classifier", "keep", "message"),
    [
        (None, "g", "classifier missing"),
        (CircleClassifier(g_center=0j, e_center=3 + 0j), "g", "circle radius missing"),
        (LinearClassifier(angle=0.0, threshold=1.0), "other", "state 'other'.*linear"),
    ],
)
def test_build_names_pulse_herald_calibration_error(
    classifier: ReadoutClassifier | None, keep: str, message: str
):
    readout = pulse(classifier)
    terminal = readout.build("terminal")
    cfg = ModuleCfgFactory.from_raw(
        {"type": "reset/herald", "readout": readout, "keep": keep}
    )
    assert cfg.type == "reset/herald"
    builder = TimelineBuilder(load_soccfg(), reps=1)
    builder.emit(cfg.build("herald", terminal=terminal))
    builder.emit(terminal)
    with pytest.raises(ValueError, match=message):
        builder.build()


@pytest.mark.parametrize(
    "change_cfg", [False, True], ids=["different-declaration", "different-cfg"]
)
def test_build_rejects_different_terminal_readout(change_cfg: bool):
    readout = pulse(LinearClassifier(angle=0.0, threshold=0.0))
    cfg = ModuleCfgFactory.from_raw(
        {"type": "reset/herald", "readout": readout, "keep": "g"}
    )
    assert cfg.type == "reset/herald"
    terminal = (
        readout.with_updates(gain=0.2).build("terminal")
        if change_cfg
        else readout.build("terminal")
    )
    builder = TimelineBuilder(load_soccfg(), reps=1)
    herald = (
        cfg.build("herald", terminal=terminal) if change_cfg else cfg.build("herald")
    )
    builder.emit(herald)
    builder.emit(terminal)
    with pytest.raises(ValueError, match="herald.*terminal readout"):
        builder.build()


@pytest.mark.parametrize("field", ["freq", "gain", "waveform.length", "ro_length"])
def test_build_rejects_herald_readout_sweeps(field: str):
    readout = pulse(LinearClassifier(angle=0.0, threshold=0.0))
    sweep: QickParam = QickSweep1D("sweep", 0.1, 0.2)
    if field == "waveform.length":
        readout.waveform = ConstWaveformCfg(length=sweep)
    else:
        setattr(readout, field, sweep)
    cfg = ModuleCfgFactory.from_raw(
        {"type": "reset/herald", "readout": readout, "keep": "g"}
    )
    assert cfg.type == "reset/herald"
    herald = cfg.build("herald")
    builder = TimelineBuilder(load_soccfg(), reps=1)
    builder.add_loop("sweep", 2)
    builder.emit(herald)
    builder.emit(herald.readout)
    with pytest.raises(ValueError, match=rf"readout\.{field}.*sweeps"):
        builder.build()


@pytest.mark.parametrize("key", ["ground", "-1", "1.0", "+1", " 1", True])
def test_mux_herald_cfg_names_an_invalid_tone_position(key: str | bool):
    with pytest.raises(ValueError, match="non-negative decimal string"):
        ModuleCfgFactory.from_raw(
            {"type": "reset/mux_herald", "readout": native_mux(), "keep": {key: "g"}}
        )


@pytest.mark.parametrize(
    ("tone", "classifier", "keep", "message"),
    [
        (2, None, "g", "tone 2.*does not exist"),
        (1, None, "g", "tone 1.*classifier missing"),
        (
            1,
            CircleClassifier(g_center=0j, e_center=3 + 0j),
            "g",
            "tone 1.*radius missing",
        ),
        (
            1,
            LinearClassifier(angle=0.0, threshold=1.0),
            "other",
            "tone 1.*state 'other'.*linear",
        ),
    ],
)
def test_build_names_mux_herald_tone_error(
    tone: int, classifier: ReadoutClassifier | None, keep: str, message: str
):
    readout = native_mux()
    readout.tones[1].classifier = classifier
    terminal = readout.build("terminal")
    cfg = ModuleCfgFactory.from_raw(
        {"type": "reset/mux_herald", "readout": readout, "keep": {tone: keep}}
    )
    assert cfg.type == "reset/mux_herald"
    builder = TimelineBuilder(mux_soccfg(), reps=1)
    builder.emit(cfg.build("herald", terminal=terminal))
    builder.emit(terminal)
    with pytest.raises(ValueError, match=message):
        builder.build()
