from typing import Any

from qick.asm_v2 import QickParam
from zcu_tools.program.v2.macro.debug import PrintTimeStamp


class MockProg:
    def __init__(self):
        self._gen_ts = [0.0, 1.2, 0.0]
        self._ro_ts = [0.0, 3.4]


def test_print_timestamp_expand():
    macro = PrintTimeStamp("test", t=1.0)
    assert macro.expand(MockProg()) == []


def test_print_timestamp_preprocess(caplog):
    import logging

    caplog.set_level(logging.DEBUG)

    prog = MockProg()
    macro = PrintTimeStamp("test_name", t=1.0, prefix=">> ")
    macro.preprocess(prog)

    log_text = caplog.text
    assert ">> test_name" in log_text
    assert "global time: 1.0" in log_text
    assert "gen[1] 1.2" in log_text
    assert "ro[1] 3.4" in log_text
    assert "gen[0]" not in log_text
    assert "ro[0]" not in log_text


def test_print_timestamp_preprocess_explicit_chs(caplog):
    import logging

    caplog.set_level(logging.DEBUG)

    prog = MockProg()
    macro = PrintTimeStamp("test_name", t=1.0, gen_chs=[0], ro_chs=[0])
    macro.preprocess(prog)

    log_text = caplog.text
    assert "gen[0] 0.0" in log_text
    assert "ro[0] 0.0" in log_text
    assert "gen[1]" not in log_text
    assert "ro[1]" not in log_text


def test_print_timestamp_qick_param(caplog):
    import logging

    caplog.set_level(logging.DEBUG)

    prog = MockProg()
    param = QickParam(start=1.0, spans={})
    macro = PrintTimeStamp("test_name", t=param)
    macro.preprocess(prog)

    assert "global time:" in caplog.text
