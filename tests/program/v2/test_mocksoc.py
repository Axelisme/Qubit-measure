import pytest
from zcu_tools.program.v2.mocksoc import (
    make_mock_soccfg,
    make_mux_mock_soccfg,
    pfb_channel,
)


def test_make_mock_soccfg():
    cfg = make_mock_soccfg(n_gens=2, n_readouts=1)
    assert cfg is not None
    assert len(cfg["gens"]) == 2
    assert len(cfg["readouts"]) == 1


@pytest.mark.parametrize(
    ("frequency_mhz", "expected"),
    [
        (6000.0, 60),
        (6009.599, 60),
        (6009.6, 60),
        (6009.601, 61),
        (6080.0, 62),
        (7190.0, 37),
        (2450.0, 32),
        (2458.0, 32),
    ],
)
def test_pfb_channel_matches_qick_physical_frequency_assignment(
    frequency_mhz: float, expected: int
):
    cfg = make_mux_mock_soccfg(2, 5, gen_ch=1, ro_chs=(4, 1))
    rocfg = cfg["readouts"][4]
    registers = cfg.calc_ro_regs(rocfg, phase=0.0, sel="product")
    cfg.calc_ro_freq(
        rocfg, {"freq": frequency_mhz, "gen_ch": None}, registers, True, 0.0
    )
    assert pfb_channel(cfg, 4, frequency_mhz) == registers["pfb_ch"] == expected


@pytest.mark.parametrize("ro_ch", [-1, 1, 2, True])
def test_pfb_channel_rejects_missing_or_non_pfb_adc(ro_ch: int):
    cfg = make_mux_mock_soccfg(1, 2, gen_ch=0, ro_chs=(0,))
    with pytest.raises(ValueError, match="PFB"):
        pfb_channel(cfg, ro_ch, 6000.0)


@pytest.mark.parametrize("frequency_mhz", [float("nan"), float("inf"), -float("inf")])
def test_pfb_channel_rejects_nonfinite_frequency(frequency_mhz: float):
    cfg = make_mux_mock_soccfg(1, 1, gen_ch=0, ro_chs=(0,))
    with pytest.raises(ValueError, match="finite MHz"):
        pfb_channel(cfg, 0, frequency_mhz)
