from __future__ import annotations

import pytest
from qick import QickConfig
from qick.qick_asm import get_version
from zcu_tools.program import describe_soc
from zcu_tools.program.v2 import make_mock_soccfg


def test_describe_soc_sections_and_counts() -> None:
    text = describe_soc(make_mock_soccfg(n_gens=2, n_readouts=1))

    assert "QICK running on ZCU216" in text
    assert f"software version {get_version()}" in text
    assert "Firmware built 2025-01-01" in text  # mock fw_timestamp
    assert "Generators (2)" in text
    assert "Readouts (1)" in text
    # one header row + one row per channel for each table
    assert text.count("axis_signal_gen_v6") == 2
    assert text.count("axis_readout_v2") == 1


def test_describe_soc_columns_present() -> None:
    text = describe_soc(make_mock_soccfg(n_gens=2, n_readouts=1))

    # ZCU216 port labels: dac "00"->"0_228", "01"->"1_228"; adc "00"->"0_224"
    assert "0_228" in text
    assert "1_228" in text
    assert "0_224" in text

    # sample rates (Msps) from rf.dacs / rf.adcs
    assert "12288.000" in text  # generator fs
    assert "2457.600" in text  # readout fs

    # max lengths in samples + us
    # gen: 65536 / (16 * 384.0) = 10.667 us ; readout: 8192 / 307.2 = 26.667 us
    assert "65536 smp (10.667 us)" in text
    assert "8192 smp (26.667 us)" in text


def test_describe_soc_rejects_non_zcu216() -> None:
    cfg = make_mock_soccfg()
    cfg["board"] = "ZCU111"
    with pytest.raises(NotImplementedError, match="ZCU216"):
        describe_soc(cfg)


def test_describe_soc_handles_gen_without_envelope() -> None:
    cfg = make_mock_soccfg(n_gens=1, n_readouts=1)
    del cfg["gens"][0]["maxlen"]
    text = describe_soc(cfg)

    # generator row still renders, with "-" for the missing envelope length
    assert "axis_signal_gen_v6" in text
    assert "65536 smp" not in text


def test_describe_soc_accepts_qickconfig_from_dict() -> None:
    # describe_soc must work on a plain QickConfig, not only MockQickSoc
    cfg = QickConfig(make_mock_soccfg()._cfg)
    assert "Generators" in describe_soc(cfg)


def test_describe_soc_accepts_plain_dict() -> None:
    # SocCfgLike is structural (__getitem__ only): a raw cfg dict — like the
    # GUI's thinner SocCfgProtocol handle — must work without a QickConfig wrapper
    cfg = make_mock_soccfg()._cfg
    assert "Generators" in describe_soc(cfg)
