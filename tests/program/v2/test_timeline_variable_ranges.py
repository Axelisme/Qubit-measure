"""Correlated timeline bounds through public compile and linked timing."""

from math import ceil

import pytest
from qick import QickConfig
from qick.asm_v2 import Macro, QickSweep1D
from zcu_tools.program.v2.modules import LoadValue
from zcu_tools.program.v2.timeline import CycleBounds, TimelineProgramV2

from tests.program._profiles import TPROC_MHZ, load_soccfg
from tests.program._timing_expectations import native_sweep, runtime_cycles

from .timing_support import shots, waves


def v6_profile() -> QickConfig:
    """Use a clock-consistent v6 generator with tProc/fabric ratio 28/39."""
    cfg = load_soccfg(1, 1).get_cfg()
    for clock in (cfg["gens"][0], cfg["rf"]["dacs"]["00"]):
        clock.update(fs=9584.64, fs_mult=39, fs_div=1, interpolation=1, f_fabric=599.04)
    cfg["gens"][0]["f_dds"] = 9584.64
    return QickConfig(cfg)


def test_complementary_tproc_pulses_leave_a_literal_constant_end():
    cfg = load_soccfg(1, 1).get_cfg()
    for clock in (cfg["gens"][0], cfg["rf"]["dacs"]["00"]):
        clock.update(
            fs=6881.28, fs_mult=28, fs_div=1, interpolation=4, f_fabric=TPROC_MHZ
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
    forward = native_sweep(prog.soccfg, 1, 9, 9, gen_ch=0)
    backward = native_sweep(prog.soccfg, 9, 1, 9, gen_ch=0)
    constant_end = forward[0] + backward[0]
    assert [waves(s) for s in shots(prog)] == [
        [0, a, a + b] for _ in range(2) for a, b in zip(forward, backward, strict=True)
    ]
    assert prog.get_time_param("after", "t") * TPROC_MHZ == pytest.approx(constant_end)
    assert not any(name.startswith("_timeline_end_") for name in prog.reg_dict)
    assert any(
        inst.get("CMD") == "WPORT_WR" and inst.get("TIME") == f"@{constant_end}"
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

    prog = Program(load_soccfg(1, 1), reps=2, initial_delay=0, relax_delay=0)
    # Each complementary operation ceils separately; the sum need not be constant.
    forward = native_sweep(prog.soccfg, 1, 9, 9, gen_ch=0)
    backward = native_sweep(prog.soccfg, 9, 1, 9, gen_ch=0)
    expected = [
        [0, ceil(runtime_cycles(prog.soccfg, a)), ceil(runtime_cycles(prog.soccfg, a)) + ceil(runtime_cycles(prog.soccfg, b))]
        for a, b in zip(forward, backward, strict=True)
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
            self.timeline_inc(QickSweep1D("i", 80 / TPROC_MHZ, 374 / TPROC_MHZ))
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
            length = QickSweep1D("i", 60 / TPROC_MHZ, 256 / TPROC_MHZ)
            other = QickSweep1D("j", 0, 80 / TPROC_MHZ)
            self.timeline_inc(7 / TPROC_MHZ)
            with self.branch(by="case", align_timeline=True) as branch:
                with branch.case(0):
                    self.timeline_inc(length + 40 / TPROC_MHZ)
                with branch.case(1):
                    self.timeline_inc(length + other)
                with branch.case(2):
                    self.timeline_inc(2 * length)

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
            length = QickSweep1D("i", 60 / TPROC_MHZ, 256 / TPROC_MHZ)
            self.timeline_inc(7 / TPROC_MHZ)
            with self.branch(by="case", align_timeline=True) as branch:
                with branch.case(0):
                    self.timeline_inc(length + 50 / TPROC_MHZ)
                with branch.case(1):
                    self.timeline_inc(2 * length)
                if extra:
                    with branch.case(2):
                        self.timeline_inc(length + 10 / TPROC_MHZ)

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
            left = QickSweep1D("i", 0, 500_000_000 / TPROC_MHZ)
            right = QickSweep1D("j", 0, 500_000_000 / TPROC_MHZ)
            with self.branch(by="case", align_timeline=True) as branch:
                for index, duration in enumerate(
                    [
                        2 * left,
                        2 * right,
                        left + right,
                        left + right + 1 / TPROC_MHZ,
                        left + right + 2 / TPROC_MHZ,
                    ]
                ):
                    with branch.case(index):
                        self.timeline_inc(duration)

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
            self.timeline_inc("duration", range_hint=hint, tag="author")
            self.timeline_inc(
                QickSweep1D("i", 72 / TPROC_MHZ, 121 / TPROC_MHZ), timeline="other"
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
            self.timeline_inc("selected", tag="selected")
            self.timeline_inc(
                QickSweep1D("i", 72 / TPROC_MHZ, 121 / TPROC_MHZ), timeline="other"
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
            self.timeline_inc(
                QickSweep1D("i", 80 / TPROC_MHZ, 92 / TPROC_MHZ)
                + QickSweep1D("j", 0, 12 / TPROC_MHZ)
            )
            self.join("pulse", into="default")

    prog = Program(v6_profile(), initial_delay=0, relax_delay=0)
    assert not any("ABS" in str(inst.get("OP", "")) for inst in prog.prog_list)
    assert [s.body_end_cycle - s.start_cycle for s in shots(prog)] == [
        80 + 6 * i + 6 * j for i in range(3) for j in range(3)
    ]
