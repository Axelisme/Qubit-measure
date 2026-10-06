from zcu_tools.program.v2.modules.pulse import Pulse


def test_pulse_none_cfg_is_no_op(mock_prog):
    p = Pulse("p", None)
    assert p.run(mock_prog, t=0.5) == 0.5
    assert p.total_length(mock_prog) == 0.0
    assert p.allow_rerun() is True
