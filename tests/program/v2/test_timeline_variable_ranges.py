"""Correlated timeline bounds through public compile and linked timing."""

import pytest
from qick import QickConfig
from qick.asm_v2 import Macro, QickSweep1D
from zcu_tools.program.v2.mocksoc import make_mock_soccfg
from zcu_tools.program.v2.timeline import TimelineProgramV2

from .timing_support import shots, waves


def v6_profile() -> QickConfig:
    """Use a clock-consistent v6 generator with tProc/fabric ratio 28/39."""
    cfg = make_mock_soccfg(1, 1).get_cfg()
    for clock in (cfg["gens"][0], cfg["rf"]["dacs"]["00"]):
        clock.update(fs=9584.64, fs_mult=39, fs_div=1, interpolation=1, f_fabric=599.04)
    cfg["gens"][0]["f_dds"] = 9584.64
    return QickConfig(cfg)


def test_same_loop_fabric_end_is_dominated_without_runtime_maximum():
    class Program(TimelineProgramV2):
        def _initialize(self, cfg):
            self.add_loop("i", 50)
            self.declare_gen(0, nqz=1)
            self.add_pulse(
                0, "p", style="const", freq=1000, phase=0, gain=0.5,
                length=QickSweep1D("i", 100 / 599.04, 492 / 599.04),
            )

        def _body(self, cfg):
            self.fork("default", into="pulse")
            self.pulse(0, "p", timeline="pulse")
            self.delay_timeline(QickSweep1D("i", 80 / 430.08, 374 / 430.08))
            self.join("pulse", into="default")

    prog = Program(v6_profile(), initial_delay=0, relax_delay=0)
    result = shots(prog)
    assert [waves(s) for s in result] == [[0]] * 50
    assert [s.body_end_cycle - s.start_cycle for s in result] == list(range(80, 375, 6))
    assert not any("ABS" in inst.get("OP", "") for inst in prog.prog_list)


class AllocateDuration(Macro):
    def preprocess(self, prog):
        prog.add_reg("duration")

    def translate(self, prog):
        pass


@pytest.mark.parametrize("hint", [None, (80, 374)])
def test_author_register_does_not_inherit_loop_correlation(hint):
    class Program(TimelineProgramV2):
        def _initialize(self, cfg):
            self.add_loop("i", 50)
            self.append_macro(AllocateDuration())

        def _body(self, cfg):
            self.write_reg_op("duration", "i", "+", 80)
            self.fork("default", into="other")
            self.delay_timeline("duration", range_hint=hint, tag="author")
            self.delay_timeline(QickSweep1D("i", 72 / 430.08, 121 / 430.08), timeline="other")
            self.join("other", into="default")

    prog = Program(v6_profile(), initial_delay=0, relax_delay=0)
    assert any("ABS" in inst.get("OP", "") for inst in prog.prog_list)
    with pytest.raises(RuntimeError, match="runtime"):
        prog.get_time_param("author", "t")
    result = shots(prog)
    assert [s.body_end_cycle - s.start_cycle for s in result] == [
        80 + i for i in range(50)
    ]
