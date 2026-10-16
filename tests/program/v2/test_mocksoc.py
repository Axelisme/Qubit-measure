from zcu_tools.program.v2.mocksoc import make_mock_soccfg


def test_make_mock_soccfg():
    cfg = make_mock_soccfg(n_gens=2, n_readouts=1)
    assert cfg is not None
    assert len(cfg["gens"]) == 2
    assert len(cfg["readouts"]) == 1
