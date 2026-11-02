"""Lookup parameter conversion at the QICK scalar and waveform seams."""

import copy
import operator
from collections.abc import Callable
from fractions import Fraction
from typing import Any

import numpy as np
import pytest
from qick import QickConfig
from qick.asm_v2 import QickParam, QickProgramV2, QickRawParam, TimedMacro, Waveform
from qick.helpers import to_int
from zcu_tools.program import v2


@pytest.mark.parametrize("quantize,trunc", [(1, False), (4, False), (4, True)])
def test_table_conversion_rounds_each_point_like_qick(
    mock_soccfg: QickConfig, quantize: int, trunc: bool
) -> None:
    values = [0.017, -0.129, 0.013, 0.017]
    scale = mock_soccfg.us2cycles(1.0)
    table = v2.TableParam("point", values)
    raw = to_int(table, scale, quantize=quantize, parname="length", trunc=trunc)
    expected = [
        to_int(value, scale, quantize=quantize, parname="length", trunc=trunc)
        for value in values
    ]
    assert isinstance(raw, v2.TableRawParam)
    assert raw.loop == "point"
    assert list(raw.values) == expected
    assert raw.start == expected[0]
    assert raw.minval() == min(expected)
    assert raw.maxval() == max(expected)
    np.testing.assert_array_equal(table.to_array(), values)
    np.testing.assert_allclose(
        table.get_rounded({"point": len(values)}).to_array(),
        np.asarray(expected) / scale,
    )


@pytest.mark.parametrize(
    "operation", [operator.add, operator.sub, operator.mul, operator.truediv]
)
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("scalar", [3.0, QickParam(3.0)], ids=["float", "qick_scalar"])
def test_scalar_arithmetic_preserves_loop_and_rejects_sweep_mixing(
    operation: Callable[..., Any], reverse: bool, scalar: float | QickParam
) -> None:
    values = [2.0, -4.0, 8.0, 2.0]
    table = v2.TableParam("point", values)
    result = operation(scalar, table) if reverse else operation(table, scalar)
    expected = [operation(3.0, v) if reverse else operation(v, 3.0) for v in values]
    assert result.loop == "point"
    assert result.is_sweep()
    assert result.minval() == min(expected)
    assert result.maxval() == max(expected)
    np.testing.assert_allclose(result.to_array(), expected)
    np.testing.assert_array_equal(table.to_array(), values)
    assert table < 0
    assert table > 0
    assert table <= -4
    assert table >= 8
    assert not table < -4
    assert not table > 8
    with pytest.raises((TypeError, RuntimeError)):
        float(table)
    for other in [v2.TableParam("other", values), QickParam(2.0, {"other": 1.0})]:
        with pytest.raises(TypeError):
            operation(other, table) if reverse else operation(table, other)


def test_raw_manager_transforms_preserve_all_points_and_rounding() -> None:
    table = v2.TableParam("point", [2.0, 4.0, -6.0])
    raw = table.to_int(2.0, parname="gain")
    original = copy.copy(raw)
    assert isinstance(original, v2.TableRawParam)
    raw *= Fraction(3, 2)
    raw += 5
    raw *= -2
    assert raw.values == (-22, -34, 26)
    assert raw.start == -22
    assert raw.minval() == -34
    assert raw.maxval() == 26
    assert original.values == (4, 8, -12)
    assert raw % 16 is raw
    raw.to_steps({"point": 3})
    assert raw.spans == {}
    assert raw.steps == {}
    np.testing.assert_array_equal((raw / 2.0).to_array(), [-11, -17, 13])
    np.testing.assert_array_equal(
        table.get_rounded({"point": 3}).to_array(), table.values
    )
    cloned = copy.copy(raw)
    np.testing.assert_array_equal(cloned.to_rounded().to_array(), [4, 8, -12])
    uneven = v2.TableRawParam("point", [4, 3], par="gain")
    with pytest.raises((ValueError, RuntimeError), match="even|divis"):
        uneven *= Fraction(1, 2)
    assert uneven.values == (4, 3)


@pytest.mark.parametrize("storage", ["wave", "register"])
def test_qick_waveform_and_close_loop_leave_table_words_unstepped(
    mock_soccfg: QickConfig,
    storage: str,
) -> None:
    scale = mock_soccfg.us2cycles(1.0)
    raw = v2.TableParam("point", [0.15, 0.31, 0.22]).to_int(scale, parname="length")
    wave = Waveform(freq=0, phase=0, env=0, gain=0, length=raw, conf=0)
    scalar_wave = Waveform(freq=0, phase=0, env=0, gain=0, length=raw.start, conf=0)
    np.testing.assert_array_equal(wave.compile(), scalar_wave.compile())
    assert wave.sweeps() == [raw]
    programs = []
    for candidate in [wave, scalar_wave]:
        prog = QickProgramV2(mock_soccfg)
        if storage == "wave":
            prog.add_raw_pulse("test", [candidate])
        else:
            init = raw if candidate is wave else QickRawParam("length", raw.start, {})
            init.to_steps({"point": 3})
            prog.add_reg(name="lookup", init=init)
        prog.open_loop(3, name="point")
        prog.close_loop()
        prog.end()
        prog.compile()
        programs.append(prog)
    assert programs[0].prog_list == programs[1].prog_list
    np.testing.assert_array_equal(
        programs[0].binprog["wmem"], programs[1].binprog["wmem"]
    )


def test_qick_parameter_copy_and_convert_time_preserve_table_type(
    mock_soccfg: QickConfig,
) -> None:
    values = [0.15, 0.31, 0.22]
    table = v2.TableParam("point", values)
    prog = QickProgramV2(mock_soccfg)
    prog.declare_gen(0, nqz=1)
    prog.add_pulse(
        ch=0, name="test", style="const", freq=table, phase=0, gain=0.2, length=0.1
    )
    prog.open_loop(len(values), name="point")
    prog.close_loop()
    prog.end()
    prog.compile()
    assert isinstance(prog.get_pulse_param("test", "freq"), v2.TableParam)
    expected = []
    for value in values:
        scalar = QickProgramV2(mock_soccfg)
        scalar.declare_gen(0, nqz=1)
        scalar.add_pulse(
            ch=0, name="test", style="const", freq=value, phase=0, gain=0.2, length=0.1
        )
        scalar.end()
        scalar.compile()
        expected.append(scalar.get_pulse_param("test", "freq"))
    np.testing.assert_allclose(
        prog.get_pulse_param("test", "freq", as_array=True), expected
    )
    macro = TimedMacro()
    rounded = macro.convert_time(prog, table, "t")
    assert isinstance(macro.t_params["t"], v2.TableParam)
    assert isinstance(rounded, v2.TableParam)
    assert macro.t_regs["t"] in prog.reg_dict, "table time must use a swept register"
    time_reg = prog.reg_dict[macro.t_regs["t"]]
    assert isinstance(time_reg.init, v2.TableRawParam)
    assert list(time_reg.init.values) == [mock_soccfg.us2cycles(v) for v in values]
    with pytest.raises(RuntimeError):
        int(time_reg.init)
    expected_times = [mock_soccfg.cycles2us(mock_soccfg.us2cycles(v)) for v in values]
    np.testing.assert_allclose(rounded.to_array(), expected_times)
    assert "point" in repr(table)
    assert "point" in str(table)


@pytest.mark.parametrize("attribute", ["start", "spans", "raw_param", "derived_param"])
def test_affine_access_rejects_tables_with_loop_diagnostic(attribute: str) -> None:
    table = v2.TableParam("point", [1.0, 3.0])
    with pytest.raises(TypeError, match="point"):
        getattr(table, attribute)


@pytest.mark.parametrize(
    "operation", [operator.add, operator.sub, operator.mul, operator.truediv]
)
def test_rounding_follows_qick_copies_and_scalar_derivations(
    mock_soccfg: QickConfig, operation: Callable[..., Any]
) -> None:
    values = [0.017, -0.129, 0.013]
    table = v2.TableParam("point", values)
    scale = mock_soccfg.us2cycles(1.0)
    derived = copy.copy(operation(table, 3.0))
    derived.to_int(scale, parname="length")
    expected = []
    for value in values:
        scalar = QickParam(value)
        scalar_derived = copy.copy(operation(scalar, 3.0))
        scalar_derived.to_int(scale, quantize=1, parname="length")
        expected.append(scalar.get_rounded({"point": len(values)}).start)
    np.testing.assert_allclose(
        table.get_rounded({"point": len(values)}).to_array(), expected
    )


@pytest.mark.parametrize("frontend", ["builder", "native"])
def test_add_table_sweep_preserves_mixed_loop_order_and_rejects_invalid_values(
    mock_soccfg: QickConfig,
    frontend: str,
) -> None:
    def declare(program: v2.TimelineBuilder | v2.TimelineProgramV2) -> None:
        for values in [[], [float("nan")], [1.0, float("inf")], [float("-inf")]]:
            with pytest.raises(ValueError, match="empty|finite"):
                program.add_table_sweep("table", values)
        table = program.add_table_sweep("table", [2.0, 1.0, 2.0])
        np.testing.assert_array_equal(table.to_array(), [2.0, 1.0, 2.0])
        program.add_sweep("linear", v2.SweepCfg(start=1.0, stop=2.0, step=1.0, expts=2))
        program.add_table_sweep("inner", [4.0])
        with pytest.raises(ValueError, match="already used"):
            program.add_table_sweep("linear", [1.0])
        with pytest.raises(ValueError, match="already used"):
            program.add_sweep(
                "table", v2.SweepCfg(start=1.0, stop=1.0, step=0.0, expts=1)
            )
        with pytest.raises(ValueError, match="already used"):
            program.add_table_sweep("reps", [1.0])

    if frontend == "builder":
        builder = v2.TimelineBuilder(mock_soccfg)
        declare(builder)
        prog = builder.build()
        with pytest.raises(ValueError, match="built|frozen"):
            builder.add_table_sweep("later", [1.0])
    else:

        class Native(v2.TimelineProgramV2):
            def _initialize(self, cfg):
                declare(self)

        prog = Native(mock_soccfg)
    assert [(loop.name, loop.count) for loop in prog.sweep_loops] == [
        ("reps", 1),
        ("table", 3),
        ("linear", 2),
        ("inner", 1),
    ]


@pytest.mark.parametrize("mismatch", [False, True])
@pytest.mark.parametrize("storage", ["wave", "register"])
def test_independent_tables_on_one_loop_validate_point_counts_at_build(
    mock_soccfg: QickConfig,
    mismatch: bool,
    storage: str,
) -> None:
    values = [0.15, 0.31, 0.22]
    other_values = [0.29, 0.11] if mismatch else [0.29, 0.11, 0.14]
    raws: list[v2.TableRawParam] = []
    scale = mock_soccfg.us2cycles(1.0)

    class Native(v2.TimelineProgramV2):
        def _initialize(self, cfg):
            self.add_loop("point", len(values))
            for name, points in [("first", values), ("second", other_values)]:
                raw = v2.TableParam("point", points).to_int(scale, parname="length")
                raws.append(raw)
                if storage == "wave":
                    self.add_raw_pulse(name, [Waveform(0, 0, 0, 0, raw, 0)])
                else:
                    self.add_reg(name=name, init=raw)

    if mismatch:
        with pytest.raises(ValueError, match="point"):
            Native(mock_soccfg)
    else:
        prog = Native(mock_soccfg)
        assert prog.loop_dict["point"] == len(values)
        for raw, points in zip(raws, [values, other_values], strict=True):
            assert list(raw.values) == [
                to_int(v, scale, parname="length") for v in points
            ]
