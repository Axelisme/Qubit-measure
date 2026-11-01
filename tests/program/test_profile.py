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


@pytest.mark.parametrize("profile", ["wiring"], indirect=True)
def test_explicit_wiring_and_optional_connections_reach_qick(profile):
    cfg = program.build_soccfg(profile)
    for source, output, fields in (
        (profile["generators"][0], cfg["gens"][0], ("dac", "tproc_ch", "tmux_ch")),
        *[
            (source, output, ("adc", "tproc_ctrl", "tmux_ch", "trigger_type",
                              "trigger_port", "trigger_bit", "tproc_ch", "b_phase"))
            for source, output in zip(profile["readouts"], cfg["readouts"], strict=True)
        ],
    ):
        for field in fields:
            if source[field] is None:
                assert field not in output
            else:
                assert output[field] == source[field]
    assert set(cfg["rf"]["dacs"]) == {profile["generators"][0]["dac"]}
    assert set(cfg["rf"]["adcs"]) == {profile["readouts"][0]["adc"]}
    # Independent worked clocks: 200*9/2 = 900; 900/(3*5) = 60 MHz.
    assert cfg["rf"]["adcs"]["21"]["fs"] == 900.0
    assert cfg["rf"]["adcs"]["21"]["f_fabric"] == 60.0


@pytest.mark.parametrize("profile", ["wiring"], indirect=True)
def test_duplicate_dac_reports_the_later_generator(profile):
    other = deepcopy(profile["generators"][0])
    other["tproc_ch"] = 19
    profile["generators"].append(other)
    with pytest.raises(ValueError, match=r"generators\.1\.dac"):
        program.build_soccfg(profile)


@pytest.mark.parametrize("profile", ["wiring"], indirect=True)
@pytest.mark.parametrize("field,value", [("fs_mult", 18), ("fs_div", 4),
                                         ("decimation", 6), ("samps_per_clk", 10),
                                         ("coupling", "DC")])
def test_shared_adc_requires_identical_rf_settings(profile, field, value):
    profile["readouts"][1][field] = value
    with pytest.raises(ValueError, match=r"readouts\.1\.adc"):
        program.build_soccfg(profile)


@pytest.mark.parametrize("profile", ["wiring"], indirect=True)
@pytest.mark.parametrize("left,right", [(None, None), (None, 1), (1, None), (1, 1)])
@pytest.mark.parametrize("users", ["generators", "readouts", "cross_kind"])
def test_ambiguous_wave_port_reports_the_later_user(profile, left, right, users):
    gen, readout = profile["generators"][0], profile["readouts"][0]
    if users == "generators":
        readout["tproc_ctrl"] = None
        gen["tmux_ch"] = left
        profile["generators"].append({**gen, "dac": "31", "tmux_ch": right})
        path = r"generators\.1\.tproc_ch"
    elif users == "readouts":
        gen["tproc_ch"] = 19
        readout["tmux_ch"] = left
        profile["readouts"][1].update(tproc_ctrl=readout["tproc_ctrl"], tmux_ch=right)
        path = r"readouts\.1\.tproc_ctrl"
    else:
        gen["tmux_ch"], readout["tmux_ch"] = left, right
        path = r"readouts\.0\.tproc_ctrl"
    with pytest.raises(ValueError, match=path):
        program.build_soccfg(profile)


@pytest.mark.parametrize("profile", ["wiring"], indirect=True)
def test_three_wave_users_can_share_a_port_with_distinct_tmux(profile):
    profile["readouts"][1].update(
        tproc_ctrl=profile["generators"][0]["tproc_ch"], tmux_ch=2
    )
    cfg = program.build_soccfg(profile)
    assert [cfg["gens"][0]["tmux_ch"], *[r["tmux_ch"] for r in cfg["readouts"]]] == [0, 1, 2]


@pytest.mark.parametrize("profile", ["derivation"], indirect=True)
@pytest.mark.parametrize("control_port", [19, 0])
def test_mux_rejects_a_readout_with_tproc_control(profile, control_port):
    ch = profile["mux"][0]["ro_chs"][0]
    profile["readouts"][ch]["tproc_ctrl"] = control_port
    with pytest.raises(ValueError, match=r"mux\.0\.ro_chs"):
        program.build_soccfg(profile)


@pytest.mark.parametrize("profile", ["derivation"], indirect=True)
def test_mux_preserves_explicit_trigger_wiring_and_readout_phase_bits(profile):
    channels = profile["mux"][0]["ro_chs"]
    for port, ch in enumerate(channels):
        profile["readouts"][ch].update(trigger_type="tport", trigger_port=11 + port,
                                        trigger_bit=9 + port, b_phase=16 + port)
    cfg = program.build_soccfg(profile)
    for port, ch in enumerate(channels):
        output = cfg["readouts"][ch]
        assert output["pfb_port"] == port
        for field in ("trigger_type", "trigger_port", "trigger_bit", "b_phase"):
            assert output[field] == profile["readouts"][ch][field]
        assert "tproc_ctrl" not in output


@pytest.mark.parametrize("profile", ["derivation"], indirect=True)
def test_independent_channel_clocks_are_derived(profile):
    # This dedicated YAML is the worked example, not an ordinary board profile.
    profile["mux"] = []
    gen = profile["generators"][0]
    other = deepcopy(gen)
    other.update(dac="35", tproc_ch=9, fs_mult=18)
    profile["generators"].append(other)
    cfg = program.build_soccfg(profile)
    assert [g["f_dds"] for g in cfg["gens"]] == [400.0, 400.0, 600.0]
    assert [g["f_fabric"] for g in cfg["gens"]] == [200.0, 200.0, 300.0]
    assert cfg["gens"][0]["fs"] == 800.0
    assert cfg["gens"][0]["fdds_div"] == 6
    assert cfg["rf"]["dacs"]["00"]["f_fabric"] == 100.0
    assert cfg["readouts"][1]["fs"] == 900.0
    assert cfg["readouts"][1]["f_dds"] == 300.0
    assert cfg["readouts"][1]["f_output"] == 60.0
    assert cfg["readouts"][1]["fdds_div"] == 6
    assert cfg["rf"]["adcs"]["01"]["f_fabric"] == 60.0


@pytest.mark.parametrize("profile", ["derivation"], indirect=True)
def test_mux_clocks_follow_refclk_and_channelization(profile):
    # 200*12/3/8 = 100 MHz; 200*9/2/3/(64/2) = 9.375 MHz.
    # RF fabric precedes PFB output and uses 900/(3*8) = 37.5 MHz.
    profile["readouts"][4]["samps_per_clk"] = 8
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
@pytest.mark.parametrize("channels", [[], [1, 1], "outside_profile", [-1], [True]])
def test_invalid_mux_routes_report_path(profile, channels):
    if channels == "outside_profile":
        channels = [len(profile["readouts"])]
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
    profile["mux"][0]["gen_ch"] = len(profile["generators"])
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
