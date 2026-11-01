import pytest
from pydantic import BaseModel
from zcu_tools.program.v2.modules.readout import (
    AbsReadoutCfg,
    PulseReadout,
    PulseReadoutCfg,
    Readout,
    TablePulseReadout,
)
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pulse_ro_cfg(ch=3, ro_ch=None):
    return PulseReadoutCfg.model_validate(
        {
            "gen_ch": ch,
            "nqz": 1,
            "freq": 700.0,
            "gain": 0.8,
            "waveform": ConstWaveformCfg(length=0.5),
            "ro_ch": ro_ch if ro_ch is not None else ch,
            "ro_length": 1.0,
        }
    )


def test_readout_cfg_namespace_dispatches_nested_pulse():
    class Modules(BaseModel):
        readout: Readout.Cfg
        pulse: PulseReadout.Cfg

    cfg = _make_pulse_ro_cfg()
    modules = Modules.model_validate({"readout": cfg.to_dict(), "pulse": cfg.to_dict()})
    assert modules.readout == cfg
    assert modules.pulse == cfg
    assert isinstance(Readout("readout", modules.readout), PulseReadout)


@pytest.mark.parametrize(
    "length,ro_length,expected_us",
    [(0.5, 2.0, 2.65), (3.0, 0.5, 3.5), (0.0, 0.0, 0.65)],
)
def test_nominal_duration_covers_adc_and_generator_without_clock_rounding(
    length: float, ro_length: float, expected_us: float
) -> None:
    cfg = _make_pulse_ro_cfg(ch=2, ro_ch=1)
    cfg.set_param("length", length)
    cfg.ro_length = ro_length
    cfg.pre_delay = 0.25
    cfg.post_delay = 0.25
    cfg.trig_offset = 0.4

    assert cfg.nominal_duration_us() == pytest.approx(expected_us)


class TestReadoutFactory:
    def test_dispatch_pulse(self):
        ro = Readout("ro", _make_pulse_ro_cfg())
        assert isinstance(ro, PulseReadout)

    def test_unknown_cfg_raises(self):
        class UnknownCfg(AbsReadoutCfg):
            type: str = "unknown"

            def build(self, name):
                raise NotImplementedError

            def set_param(self, name, value):
                raise NotImplementedError

        with pytest.raises(NotImplementedError):
            Readout("ro", UnknownCfg())

    def test_name_set(self):
        ro = Readout("myro", _make_pulse_ro_cfg())
        assert ro.name == "myro"


class TestTablePulseReadout:
    def test_rejects_empty_tables(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            TablePulseReadout(
                "ro",
                _make_pulse_ro_cfg(),
                idx_reg="freq",
                freq_words=[],
                ro_freq_words=[],
            )

    def test_rejects_mismatched_table_lengths(self) -> None:
        with pytest.raises(ValueError, match="equal length"):
            TablePulseReadout(
                "ro",
                _make_pulse_ro_cfg(),
                idx_reg="freq",
                freq_words=[1, 2],
                ro_freq_words=[3],
            )


@pytest.mark.parametrize("flat_top", [False, True])
def test_native_readout_conversion_binds_arb_to_explicit_capture(tmp_path, flat_top):
    import numpy as np
    from zcu_tools import cfg_model as core
    from zcu_tools.program.v2.modules import ModuleCfgFactory
    from zcu_tools.program.v2.modules.waveform import ArbWaveformCfg, FlatTopWaveformCfg
    from zcu_tools.resources.waveform_assets import ArbWaveformDatabase

    first = tmp_path / "first"
    first.mkdir()
    np.savez(first / "shape.npz", time=[0.0, 0.5], idata=[0.0, 1.0], qdata=[0.0, 0.0])
    context = core.CfgValidationContext(assets=ArbWaveformDatabase.info_at(first))
    raw = {
        "type": "readout/pulse",
        "gen_ch": 1,
        "nqz": 1,
        "freq": 700.0,
        "gain": 0.4,
        "waveform": {
            "style": "flat_top",
            "length": 2.0,
            "raise_waveform": {"style": "arb", "data": "shape"},
        }
        if flat_top
        else {"style": "arb", "data": "shape"},
        "ro_ch": 0,
        "ro_length": 3.0,
    }
    source = core.ModuleCfgFactory.from_raw(raw, context=context)
    native = ModuleCfgFactory.from_core(source, context=context)
    assert isinstance(native, PulseReadoutCfg)
    waveform = (
        native.waveform.raise_waveform
        if isinstance(native.waveform, FlatTopWaveformCfg)
        else native.waveform
    )
    assert isinstance(waveform, ArbWaveformCfg)
    assert waveform.path == (first / "shape.npz").resolve().as_posix()
    assert waveform.length == 0.5
    with pytest.raises(ValueError, match="assets"):
        ModuleCfgFactory.from_core(source, context=None)
