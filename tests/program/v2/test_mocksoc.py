import pytest
from zcu_tools.program.v2.mocksoc import pfb_channel

from tests.program._profiles import load_mux_soccfg


@pytest.mark.parametrize(
    "frequency_mhz",
    [6000.0, 6009.599, 6009.6, 6009.601, 6080.0, 7190.0, 2450.0, 2458.0],
)
def test_pfb_channel_matches_qick_physical_frequency_assignment(
    frequency_mhz: float,
):
    cfg = load_mux_soccfg(2, 5, gen_ch=1, ro_chs=(4, 1))
    rocfg = cfg["readouts"][4]
    registers = cfg.calc_ro_regs(rocfg, phase=0.0, sel="product")
    cfg.calc_ro_freq(
        rocfg, {"freq": frequency_mhz, "gen_ch": None}, registers, True, 0.0
    )
    assert pfb_channel(cfg, 4, frequency_mhz) == registers["pfb_ch"]


@pytest.mark.parametrize("ro_ch", [-1, 1, 2, True])
def test_pfb_channel_rejects_missing_or_non_pfb_adc(ro_ch: int):
    cfg = load_mux_soccfg(1, 2, gen_ch=0, ro_chs=(0,))
    with pytest.raises(ValueError, match="PFB"):
        pfb_channel(cfg, ro_ch, 6000.0)


@pytest.mark.parametrize("frequency_mhz", [float("nan"), float("inf"), -float("inf")])
def test_pfb_channel_rejects_nonfinite_frequency(frequency_mhz: float):
    cfg = load_mux_soccfg(1, 1, gen_ch=0, ro_chs=(0,))
    with pytest.raises(ValueError, match="finite MHz"):
        pfb_channel(cfg, 0, frequency_mhz)
