"""Correlated timeline bounds through public compile and linked timing."""

import pytest
from qick import QickConfig
from qick.asm_v2 import Macro, QickSweep1D
from zcu_tools.program.v2.mocksoc import make_mock_soccfg
from zcu_tools.program.v2.modules import LoadValue
from zcu_tools.program.v2.timeline import CycleBounds, TimelineProgramV2

from .timing_support import shots, waves


def v6_profile() -> QickConfig:
    """Use a clock-consistent v6 generator with tProc/fabric ratio 28/39."""
    cfg = make_mock_soccfg(1, 1).get_cfg()
    for clock in (cfg["gens"][0], cfg["rf"]["dacs"]["00"]):
        clock.update(fs=9584.64, fs_mult=39, fs_div=1, interpolation=1, f_fabric=599.04)
    cfg["gens"][0]["f_dds"] = 9584.64
    return QickConfig(cfg)


def test_complementary_tproc_pulses_leave_a_literal_constant_end():
    cfg = make_mock_soccfg(1, 1).get_cfg()
    for clock in (cfg["gens"][0], cfg["rf"]["dacs"]["00"]):
        clock.update(
            fs=6881.28, fs_mult=28, fs_div=1, interpolation=4, f_fabric=430.08
        )
    cfg["gens"][0].update(
        type="axis_sg_int4_v1",
        has_mixer=True,
        f_dds=1720.32,
        fdds_div=4,
        samps_per_clk=4,
    )

    class Program(TimelineProgramV2):
        def _initialize(self, cfg):
            self.add_loop("x", 9)
            self.declare_gen(0, nqz=1, mixer_freq=0)
            x = QickSweep1D("x", 1, 9)
            for name, length in (("x", x), ("complement", 10 - x), ("after", 0.1)):
                self.add_pulse(
                    0, name, style="const", freq=100, phase=0, gain=0.5, length=length
                )

        def _body(self, cfg):
            self.pulse(0, "x", timeline="default")
            self.pulse(0, "complement", timeline="default")
            self.pulse(0, "after", timeline="default", tag="after")

    prog = Program(QickConfig(cfg), reps=2, initial_delay=0, relax_delay=0)
    # Native QICK steps are +/-430 ticks. Rounded starts sum to 4301.
    assert [waves(s) for s in shots(prog)] == [
        [0, 430 + 430 * i, 4301] for _ in range(2) for i in range(9)
    ]
    assert prog.get_time_param("after", "t") * 430.08 == pytest.approx(4301)
    assert not any(name.startswith("_timeline_end_") for name in prog.reg_dict)
    assert any(
        inst.get("CMD") == "WPORT_WR" and inst.get("TIME") == "@4301"
        for inst in prog.prog_list
    )


def test_complementary_fabric_pulses_keep_pointwise_ceil_differences():
    class Program(TimelineProgramV2):
        def _initialize(self, cfg):
            self.add_loop("x", 9)
            self.declare_gen(0, nqz=1)
            x = QickSweep1D("x", 1, 9)
            for name, length in (("x", x), ("complement", 10 - x), ("after", 0.1)):
                self.add_pulse(
                    0, name, style="const", freq=100, phase=0, gain=0.5, length=length
                )

        def _body(self, cfg):
            self.pulse(0, "x", timeline="default")
            self.pulse(0, "complement", timeline="default")
            self.pulse(0, "after", timeline="default", tag="after")

    prog = Program(make_mock_soccfg(1, 1), reps=2, initial_delay=0, relax_delay=0)
    # At 384 MHz the k=8 upper conversion is 287/256. Separate ceils
    # of complementary lengths alternate between 4306 and 4305, not one end.
    expected = [
        [0, 431, 4306],
        [0, 861, 4305],
        [0, 1292, 4306],
        [0, 1722, 4305],
        [0, 2153, 4306],
        [0, 2583, 4305],
        [0, 3014, 4306],
        [0, 3444, 4305],
        [0, 3875, 4306],
    ]
    assert [waves(s) for s in shots(prog)] == expected * 2
    with pytest.raises(RuntimeError, match="non-affine"):
        prog.get_time_param("after", "t")
    assert any(name.startswith("_timeline_end_") for name in prog.reg_dict)
    assert prog.compile_datamem() is None


def test_same_loop_fabric_end_is_dominated_without_runtime_maximum():
    class Program(TimelineProgramV2):
        def _initialize(self, cfg):
            self.add_loop("i", 50)
            self.declare_gen(0, nqz=1)
            self.add_pulse(
                0,
                "p",
                style="const",
                freq=1000,
                phase=0,
                gain=0.5,
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
    assert not any("ABS" in str(inst.get("OP", "")) for inst in prog.prog_list)


def test_three_way_join_prunes_partial_dominance_and_reconstructs_shared_length_once():
    class Program(TimelineProgramV2):
        def _initialize(self, cfg):
            self.add_loop("i", 50)
            self.add_loop("j", 3)
            self.add_loop("case", 3)

        def _body(self, cfg):
            length = QickSweep1D("i", 60 / 430.08, 256 / 430.08)
            other = QickSweep1D("j", 0, 80 / 430.08)
            self.delay_timeline(7 / 430.08)
            with self.branch(by="case", align_timeline=True) as branch:
                with branch.case(0):
                    self.delay_timeline(length + 40 / 430.08)
                with branch.case(1):
                    self.delay_timeline(length + other)
                with branch.case(2):
                    self.delay_timeline(2 * length)

    prog = Program(v6_profile(), initial_delay=0, relax_delay=0)
    assert [s.body_end_cycle - s.start_cycle for s in shots(prog)] == [
        7 + max(2 * length, length + other)
        for length in range(60, 257, 4)
        for other in (0, 40, 80)
        for _ in range(3)
    ]
    assert sum(inst.get("IF") == "S" for inst in prog.prog_list) == 1
    # Only the shared 4*i is rebuilt as a native sweep register.
    assert (
        sum(
            register.init is not None and register.init.spans == {"i": 196}
            for register in prog.reg_dict.values()
        )
        == 1
    )


@pytest.mark.parametrize("extra", [False, True])
def test_join_eliminates_every_provably_dominated_candidate(extra):
    class Program(TimelineProgramV2):
        def _initialize(self, cfg):
            self.add_loop("i", 50)
            self.add_loop("case", 3 if extra else 2)

        def _body(self, cfg):
            length = QickSweep1D("i", 60 / 430.08, 256 / 430.08)
            self.delay_timeline(7 / 430.08)
            with self.branch(by="case", align_timeline=True) as branch:
                with branch.case(0):
                    self.delay_timeline(length + 50 / 430.08)
                with branch.case(1):
                    self.delay_timeline(2 * length)
                if extra:
                    with branch.case(2):
                        self.delay_timeline(length + 10 / 430.08)

    prog = Program(v6_profile(), initial_delay=0, relax_delay=0)
    assert [s.body_end_cycle - s.start_cycle for s in shots(prog)] == [
        7 + 2 * length for length in range(60, 257, 4) for _ in range(3 if extra else 2)
    ]
    assert not any(
        inst.get("CMD") == "REG_WR" and inst.get("IF") in ("S", "NS")
        for inst in prog.prog_list
    )


def test_join_compares_negative_residuals_with_a_proven_signed32_difference():
    class Program(TimelineProgramV2):
        def _initialize(self, cfg):
            self.add_loop("i", 2)
            self.add_loop("j", 2)
            self.add_loop("case", 5)

        def _body(self, cfg):
            left = QickSweep1D("i", 0, 500_000_000 / 430.08)
            right = QickSweep1D("j", 0, 500_000_000 / 430.08)
            with self.branch(by="case", align_timeline=True) as branch:
                for index, duration in enumerate(
                    [
                        2 * left,
                        2 * right,
                        left + right,
                        left + right + 1 / 430.08,
                        left + right + 2 / 430.08,
                    ]
                ):
                    with branch.case(index):
                        self.delay_timeline(duration)

    prog = Program(v6_profile(), initial_delay=0, relax_delay=0)
    assert [s.body_end_cycle - s.start_cycle for s in shots(prog)] == [
        end
        for end in (2, 1_000_000_000, 1_000_000_000, 1_000_000_002)
        for _ in range(5)
    ]
    # S=L+M leaves L-M and M-L, with negative legal values. Their difference
    # stays in [-1e9, 1e9], so two flag comparisons are safe without a mask.
    assert (
        sum(
            inst.get("CMD") == "REG_WR" and inst.get("IF") == "S"
            for inst in prog.prog_list
        )
        == 2
    )
    assert not any("^" in str(inst.get("OP", "")) for inst in prog.prog_list)


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
            self.delay_timeline(
                QickSweep1D("i", 72 / 430.08, 121 / 430.08), timeline="other"
            )
            self.join("other", into="default")

    prog = Program(v6_profile(), initial_delay=0, relax_delay=0)
    assert any(
        inst.get("CMD") == "TEST" or "^" in str(inst.get("OP", ""))
        for inst in prog.prog_list
    )
    with pytest.raises(RuntimeError, match="runtime"):
        prog.get_time_param("author", "t")
    result = shots(prog)
    assert [s.body_end_cycle - s.start_cycle for s in result] == [
        80 + i for i in range(50)
    ]


def test_measurement_selected_table_is_independent_of_loop_time():
    loader = LoadValue("measured", [80, 129], "duration", "selected")

    class Program(TimelineProgramV2):
        def _initialize(self, cfg):
            self.add_loop("i", 50)
            self.declare_readout(0, length=0.1)
            self.append_macro(AllocateDuration())
            self.declare(loader)

        def _body(self, cfg):
            self.read_input(0)
            with self.time_producer("duration", CycleBounds(0, 1)):
                self.write_reg_op("duration", "s_port_l", "AND", 1)
            self.emit(loader)
            self.fork("default", into="other")
            self.delay_timeline("selected", tag="selected")
            self.delay_timeline(
                QickSweep1D("i", 72 / 430.08, 121 / 430.08), timeline="other"
            )
            self.join("other", into="default")

    prog = Program(v6_profile(), initial_delay=0, relax_delay=0)
    assert any(
        inst.get("CMD") == "TEST" or "^" in str(inst.get("OP", ""))
        for inst in prog.prog_list
    )
    with pytest.raises(RuntimeError, match="runtime"):
        prog.get_time_param("selected", "t")


def test_two_loop_fabric_conversion_keeps_linear_part_for_dominance():
    class Program(TimelineProgramV2):
        def _initialize(self, cfg):
            self.add_loop("i", 3)
            self.add_loop("j", 3)
            self.declare_gen(0, nqz=1)
            self.add_pulse(
                0,
                "p",
                style="const",
                freq=1000,
                phase=0,
                gain=0.5,
                length=(
                    QickSweep1D("i", 100 / 599.04, 116 / 599.04)
                    + QickSweep1D("j", 0, 16 / 599.04)
                ),
            )

        def _body(self, cfg):
            self.fork("default", into="pulse")
            self.pulse(0, "p", timeline="pulse")
            self.delay_timeline(
                QickSweep1D("i", 80 / 430.08, 92 / 430.08)
                + QickSweep1D("j", 0, 12 / 430.08)
            )
            self.join("pulse", into="default")

    prog = Program(v6_profile(), initial_delay=0, relax_delay=0)
    assert not any("ABS" in str(inst.get("OP", "")) for inst in prog.prog_list)
    assert [s.body_end_cycle - s.start_cycle for s in shots(prog)] == [
        80,
        86,
        92,
        86,
        92,
        98,
        92,
        98,
        104,
    ]
