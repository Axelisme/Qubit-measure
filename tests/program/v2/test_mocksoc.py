import pytest
from zcu_tools.program.v2.mocksoc import make_mock_soc, make_mock_soccfg


def test_make_mock_soccfg():
    cfg = make_mock_soccfg(n_gens=2, n_readouts=1)
    assert cfg is not None
    assert (
        getattr(cfg, "gens", None) is None
        or getattr(cfg, "gens") is not None
        or "gens" in cfg._cfg
    )
    assert len(cfg["gens"]) == 2
    assert "readouts" in cfg._cfg
    assert len(cfg["readouts"]) == 1


def test_mock_soc_methods():
    soc = make_mock_soc(n_gens=1, n_readouts=1)

    # Test getting tproc counter returns huge number
    assert soc.get_tproc_counter(0) == soc._BIG_COUNT

    # Test readout start and poll
    soc.start_readout(total_shots=100, counter_addr=0, ch_list=[0], reads_per_shot=[1])
    assert soc._readout_state == (100, [0], [1])
    assert soc._poll_done is False

    # Test poll_data with fake data
    data = soc.poll_data()
    assert len(data) == 1
    total_shots, (arr_list, _) = data[0]
    assert total_shots == 100
    assert len(arr_list) == 1
    assert arr_list[0].shape == (100, 2)
    assert soc._poll_done is True

    # Second poll should return empty
    assert soc.poll_data() == []


def test_mock_soc_decimated_accumulated():
    soc = make_mock_soc()

    dec = soc.get_decimated(0, 0, 50)
    assert dec.shape == (50, 2)

    acc = soc.get_accumulated(0, 0, 20)
    assert acc.shape == (20, 2)
