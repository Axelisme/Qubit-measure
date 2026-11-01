from __future__ import annotations

import pytest
from qick import QickConfig
from zcu_tools.program import describe_soc

from tests.program._profiles import load_soccfg


def test_describe_soc_sections_and_counts() -> None:
    cfg = load_soccfg(n_gens=2, n_readouts=1)
    text = describe_soc(cfg)

    assert f"QICK running on {cfg['board']}" in text
    assert f"software version {cfg['sw_version']}" in text
    assert f"Firmware built {cfg['fw_timestamp']}" in text
    for key, label in (("gens", "Generators"), ("readouts", "Readouts")):
        channels = cfg[key]
        assert f"{label} ({len(channels)})" in text
        for channel_type in {channel["type"] for channel in channels}:
            assert text.count(channel_type) == sum(
                channel["type"] == channel_type for channel in channels
            )


def test_describe_soc_columns_present() -> None:
    cfg = load_soccfg(n_gens=2, n_readouts=1)
    text = describe_soc(cfg)

    # ZCU216 port labels: dac "00"->"0_228", "01"->"1_228"; adc "00"->"0_224"
    assert "0_228" in text
    assert "1_228" in text
    assert "0_224" in text

    # sample rates (Msps) from rf.dacs / rf.adcs
    gen, ro = cfg["gens"][0], cfg["readouts"][0]
    assert f"{cfg['rf']['dacs'][gen['dac']]['fs']:.3f}" in text
    assert f"{cfg['rf']['adcs'][ro['adc']]['fs']:.3f}" in text

    # One sample spans 1/(fabric * samples_per_clock) us on the generator.
    gen_us = gen["maxlen"] / (gen["samps_per_clk"] * gen["f_fabric"])
    ro_us = ro["buf_maxlen"] / ro["f_output"]
    assert f"{gen['maxlen']} smp ({gen_us:.3f} us)" in text
    assert f"{ro['buf_maxlen']} smp ({ro_us:.3f} us)" in text


def test_describe_soc_rejects_non_zcu216() -> None:
    cfg = load_soccfg()
    cfg["board"] = "ZCU111"
    with pytest.raises(NotImplementedError, match="ZCU216"):
        describe_soc(cfg)


def test_describe_soc_handles_gen_without_envelope() -> None:
    cfg = load_soccfg(n_gens=1, n_readouts=1)
    gen = cfg["gens"][0]
    del gen["maxlen"]
    text = describe_soc(cfg)

    # The generator row still renders a missing envelope as "-".
    row = next(line for line in text.splitlines() if gen["type"] in line)
    assert row.split()[-1] == "-"


def test_describe_soc_accepts_qickconfig_from_dict() -> None:
    # describe_soc must work on a plain QickConfig, not only MockQickSoc
    cfg = QickConfig(load_soccfg().get_cfg())
    assert "Generators" in describe_soc(cfg)


def test_describe_soc_accepts_plain_dict() -> None:
    # SocCfgLike is structural (__getitem__ only): a raw cfg dict — like the
    # GUI's thinner SocCfgProtocol handle — must work without a QickConfig wrapper
    cfg = load_soccfg().get_cfg()
    assert "Generators" in describe_soc(cfg)


def test_describe_soc_handles_gen_without_envelope_or_clocks() -> None:
    cfg = load_soccfg(n_gens=1, n_readouts=1)
    gen = cfg["gens"][0]
    gen["type"] = "axis_sg_int4_v1"
    gen["dac"] = "21"
    cfg["rf"]["dacs"]["21"] = {"fs": 8192.0}
    ro = cfg["readouts"][0]
    ro["ro_type"] = "axis_readout_v3"
    ro["adc"] = "12"
    ro["buf_maxlen"] = 4096
    ro["f_output"] = 192.0
    cfg["rf"]["adcs"]["12"] = {"fs": 1536.0}
    del gen["maxlen"]
    del gen["samps_per_clk"]
    del gen["f_fabric"]

    text = describe_soc(cfg)

    assert f"Generators ({len(cfg['gens'])})" in text
    assert f"Readouts ({len(cfg['readouts'])})" in text
    assert "max pulse len" in text
    assert "buf maxlen" in text
    gen_row = next(line for line in text.splitlines() if "axis_sg_int4_v1" in line)
    ro_row = next(line for line in text.splitlines() if "axis_readout_v3" in line)
    assert gen_row.split() == [
        "0",
        "axis_sg_int4_v1",
        "1_230",
        f"{cfg['rf']['dacs'][gen['dac']]['fs']:.3f}",
        "-",
    ]
    assert ro_row.split() == [
        "0",
        "axis_readout_v3",
        "2_225",
        f"{cfg['rf']['adcs'][ro['adc']]['fs']:.3f}",
        str(ro["buf_maxlen"]),
        "smp",
        f"({ro['buf_maxlen'] / ro['f_output']:.3f}",
        "us)",
    ]
