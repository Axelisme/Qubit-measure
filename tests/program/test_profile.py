"""Profile-to-QICK clock, validation and real-compile contracts."""

from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from zcu_tools import program
from zcu_tools.program.v2 import TimelineProgramV2


@pytest.fixture
def profile(request):
    with (
        Path(__file__).parent / "profiles" / f"{request.param}.yaml"
    ).open() as stream:
        return yaml.safe_load(stream)


@pytest.mark.parametrize("profile", ["derivation"], indirect=True)
def test_independent_clocks_and_channel_groups_are_derived(profile):
    # This dedicated YAML is the worked example, not an ordinary board profile.
    profile["mux"] = []
    gen = profile["generators"][0]
    other = deepcopy(gen)
    other.update(count=1, fs_mult=18)
    profile["generators"].append(other)
    profile["readouts"][0]["count"] = 2
    cfg = program.build_soccfg(profile)
    assert [g["f_dds"] for g in cfg["gens"]] == [400.0, 400.0, 600.0]
    assert [g["f_fabric"] for g in cfg["gens"]] == [200.0, 200.0, 300.0]
    assert cfg["gens"][0]["fs"] == 800.0
    assert cfg["gens"][0]["fdds_div"] == 6
    assert cfg["rf"]["dacs"]["00"]["f_fabric"] == 100.0
    assert cfg["gens"][2]["dac"] == "02"
    assert cfg["gens"][2]["tproc_ch"] == 2
    assert [r["tproc_ctrl"] for r in cfg["readouts"]] == [3, 4]
    assert [r["trigger_bit"] for r in cfg["readouts"]] == [0, 1]
    assert cfg["readouts"][1]["fs"] == 900.0
    assert cfg["readouts"][1]["f_dds"] == 300.0
    assert cfg["readouts"][1]["f_output"] == 60.0
    assert cfg["readouts"][1]["fdds_div"] == 6
    assert cfg["rf"]["adcs"]["01"]["f_fabric"] == 60.0


@pytest.mark.parametrize("profile", ["derivation"], indirect=True)
def test_mux_clocks_follow_refclk_and_channelization(profile):
    # 200*12/3/8 = 100 MHz; 200*9/2/3/(64/2) = 9.375 MHz.
    # RF fabric precedes PFB output and uses 900/(3*8) = 37.5 MHz.
    profile["readouts"][0]["samps_per_clk"] = 8
    cfg = program.build_soccfg(profile)
    assert cfg["gens"][1]["f_dds"] == 100.0
    assert cfg["gens"][1]["fdds_div"] == 24
    assert cfg["readouts"][4]["f_output"] == 9.375
    assert cfg["readouts"][4]["f_dds"] == 9.375
    assert cfg["readouts"][4]["fdds_div"] == 192
    assert cfg["rf"]["adcs"]["04"]["f_fabric"] == 37.5


@pytest.mark.parametrize("profile", ["ordinary", "mux"], indirect=True)
def test_missing_fields_report_their_profile_path(profile):
    del profile["generators"][0]["fs_mult"]
    del profile["tprocs"][0]["f_core"]
    with pytest.raises(ValueError, match="Field required") as failure:
        program.build_soccfg(profile)
    assert "generators.0.fs_mult" in str(failure.value)
    assert "tprocs.0.f_core" in str(failure.value)


@pytest.mark.parametrize("profile", ["ordinary", "mux"], indirect=True)
@pytest.mark.parametrize("field", ["fs", "f_dds", "f_fabric", "fdds_div"])
def test_redundant_generator_clocks_cannot_override_derivation(profile, field):
    profile["generators"][0][field] = 123.0
    with pytest.raises(ValueError, match=rf"generators\.0\.{field}"):
        program.build_soccfg(profile)


@pytest.mark.parametrize("profile", ["ordinary"], indirect=True)
def test_redundant_readout_output_cannot_override_derivation(profile):
    profile["readouts"][0]["f_output"] = 123.0
    with pytest.raises(ValueError, match=r"readouts\.0\.f_output"):
        program.build_soccfg(profile)


@pytest.mark.parametrize("profile", ["ordinary"], indirect=True)
@pytest.mark.parametrize("invalid", [0, -1, True, 1.5])
def test_invalid_clock_divider_reports_path(profile, invalid):
    profile["generators"][0]["fs_div"] = invalid
    with pytest.raises(ValueError, match=r"generators\.0\.fs_div"):
        program.build_soccfg(profile)


@pytest.mark.parametrize("profile", ["mux"], indirect=True)
@pytest.mark.parametrize("channels", [[], [1, 1], [5], [-1], [True]])
def test_invalid_mux_routes_report_path(profile, channels):
    profile["mux"][0]["ro_chs"] = channels
    with pytest.raises(ValueError, match=r"mux\.0\.ro_chs"):
        program.build_soccfg(profile)


@pytest.mark.parametrize("profile", ["mux"], indirect=True)
@pytest.mark.parametrize("channels", [1, 63])
def test_pfb_channelization_requires_an_integer_clock_divider(profile, channels):
    profile["mux"][0]["readout"]["pfb_nch"] = channels
    with pytest.raises(ValueError, match=r"mux\.0\.readout\.pfb_nch"):
        program.build_soccfg(profile)


@pytest.mark.parametrize("profile", ["mux"], indirect=True)
def test_mux_generator_outside_profile_reports_path(profile):
    profile["mux"][0]["gen_ch"] = sum(
        group["count"] for group in profile["generators"]
    )
    with pytest.raises(ValueError, match=r"mux\.0\.gen_ch"):
        program.build_soccfg(profile)


@pytest.mark.parametrize("profile", ["ordinary", "mux"], indirect=True)
def test_profile_and_successive_results_do_not_share_mutable_data(profile):
    original = deepcopy(profile)
    first = program.build_soccfg(profile)
    first["gens"][0]["fs_mult"] = -1
    first["tprocs"][0]["output_pins"].append(["changed"])
    assert profile == original
    assert program.build_soccfg(profile).get_cfg() != first.get_cfg()


@pytest.mark.parametrize("profile", ["ordinary", "mux"], indirect=True)
def test_yaml_profile_compiles_real_qick_pulse_and_readout(profile):
    mux = bool(profile["mux"])
    gen_ch, ro_ch = (1, 4) if mux else (0, 0)

    class ReadoutProgram(TimelineProgramV2):
        def _initialize(self, cfg):
            if mux:
                self.declare_gen(
                    gen_ch,
                    nqz=1,
                    mixer_freq=6000.0,
                    mux_freqs=[6000.0],
                    mux_gains=[0.1],
                    mux_phases=[0.0],
                    ro_ch=ro_ch,
                )
                self.declare_readout(ro_ch, length=0.2, freq=6000.0, gen_ch=gen_ch)
                self.add_pulse(gen_ch, "probe", style="const", mask=[0], length=0.2)
            else:
                self.declare_gen(gen_ch, nqz=1)
                self.declare_readout(ro_ch, length=0.2)
                self.add_readoutconfig(ro_ch, "adc", freq=1000.0, gen_ch=gen_ch)
                self.add_pulse(
                    gen_ch,
                    "probe",
                    style="const",
                    freq=1000.0,
                    phase=0,
                    gain=0.1,
                    length=0.2,
                )

        def _body(self, cfg):
            if not mux:
                self.send_readoutconfig(ro_ch, "adc", t=0)
            self.pulse(gen_ch, "probe", t=0)
            self.trigger(ros=[ro_ch], t=0)

    compiled = ReadoutProgram(
        program.build_soccfg(profile), reps=1, initial_delay=0, relax_delay=0
    )
    assert len(compiled.binprog["pmem"]) > 0
    assert compiled.ro_chs[ro_ch]["length"] > 0
