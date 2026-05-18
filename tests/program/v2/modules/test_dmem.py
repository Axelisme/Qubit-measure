from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest
from zcu_tools.program.v2.modules.dmem import _COMPRESS_MIN_VALUES, LoadValue, ScanWith


def _make_dmem_prog(temp_regs=("r10", "r11")):
    """Prog mock where acquire_temp_reg yields temp_regs as a context manager."""
    prog = MagicMock()
    prog.add_dmem.return_value = 0

    @contextmanager
    def _acq(n):
        yield list(temp_regs[:n])

    prog.acquire_temp_reg.side_effect = _acq
    return prog


def test_small_values_not_compressed():
    lv = LoadValue("x", [1, 2, 3], idx_reg="i", val_reg="v")
    assert lv._is_compressed is False
    assert lv._packed_values == [1, 2, 3]
    assert lv.allow_rerun() is True


def test_large_small_range_gets_compressed():
    values = [i % 4 for i in range(64)]  # 2 bits, 64 values
    lv = LoadValue("x", values, idx_reg="i", val_reg="v")
    assert lv._is_compressed is True
    assert lv._bits_per_value == 2
    assert lv._values_per_word == 16
    assert len(lv._packed_values) == 64 // 16
    assert lv.allow_rerun() is True


def test_negative_values_rejected():
    with pytest.raises(ValueError, match=r"\[0,"):
        LoadValue("x", [1, -1, 2], idx_reg="i", val_reg="v")


def test_value_exceeding_int32_max_rejected():
    with pytest.raises(ValueError, match=r"\[0,"):
        LoadValue("x", [0, 2**31], idx_reg="i", val_reg="v")


def test_int32_max_value_accepted():
    lv = LoadValue("x", [2**31 - 1], idx_reg="i", val_reg="v")
    assert lv.values == [2**31 - 1]


def test_auto_compress_off_keeps_values():
    values = [i % 4 for i in range(64)]
    lv = LoadValue("x", values, idx_reg="i", val_reg="v", auto_compress=False)
    assert lv._is_compressed is False
    assert lv._packed_values == values


def test_empty_values_short_circuit():
    lv = LoadValue("x", [], idx_reg="i", val_reg="v")
    prog = _make_dmem_prog()

    lv.init(prog)
    out = lv.run(prog, t=1.25)

    assert out == 1.25
    prog.add_dmem.assert_not_called()
    prog.read_dmem.assert_not_called()
    prog.write_reg.assert_not_called()
    prog.write_reg_op.assert_not_called()


def test_bits_needed_unsigned_boundaries():
    assert LoadValue._bits_needed(0) == 1
    assert LoadValue._bits_needed(1) == 1
    assert LoadValue._bits_needed(2) == 2
    assert LoadValue._bits_needed(255) == 8


# ---------------------------------------------------------------------------
# LoadValue.run — uncompressed path
# ---------------------------------------------------------------------------


def test_load_value_run_uncompressed_zero_offset_uses_write_reg():
    lv = LoadValue("x", [10, 20, 30], idx_reg="i", val_reg="v")
    prog = _make_dmem_prog()
    lv.init(prog)
    lv.offset = 0
    lv.run(prog)

    prog.write_reg.assert_called_once_with("r10", "i")
    prog.read_dmem.assert_called_once_with(dst="v", addr="r10")


def test_load_value_run_uncompressed_nonzero_offset_uses_write_reg_op():
    lv = LoadValue("x", [10, 20, 30], idx_reg="i", val_reg="v")
    prog = _make_dmem_prog()
    lv.init(prog)
    lv.offset = 5
    lv.run(prog)

    prog.write_reg_op.assert_any_call("r10", "i", "+", 5)
    prog.read_dmem.assert_called_once_with(dst="v", addr="r10")


def test_load_value_run_uncompressed_returns_t():
    lv = LoadValue("x", [1, 2, 3], idx_reg="i", val_reg="v")
    prog = _make_dmem_prog()
    lv.init(prog)
    lv.offset = 0
    out = lv.run(prog, t=1.5)
    assert out == 1.5


# ---------------------------------------------------------------------------
# LoadValue.run — compressed path
# ---------------------------------------------------------------------------


def test_load_value_run_compressed_uses_asr_word_shift():
    values = [i % 4 for i in range(64)]
    lv = LoadValue("x", values, idx_reg="i", val_reg="v")
    assert lv._is_compressed is True
    prog = _make_dmem_prog()
    lv.init(prog)
    lv.run(prog)

    # addr = idx ASR word_shift
    prog.write_reg_op.assert_any_call("r10", "i", "ASR", lv._word_shift)
    prog.read_dmem.assert_called_once()


def test_load_value_run_compressed_nonzero_offset_calls_inc_reg():
    values = [i % 4 for i in range(64)]
    lv = LoadValue("x", values, idx_reg="i", val_reg="v")
    prog = _make_dmem_prog()
    lv.init(prog)
    lv.offset = 3
    lv.run(prog)
    prog.inc_reg.assert_called_with("r10", 3)


# ---------------------------------------------------------------------------
# ScanWith
# ---------------------------------------------------------------------------


def test_scan_with_empty_values_rejected():
    with pytest.raises(ValueError):
        ScanWith("s", [], val_reg="v")


def test_scan_with_allow_rerun_returns_false():
    s = ScanWith("s", [1, 2, 3], val_reg="v")
    assert s.allow_rerun() is False


def test_scan_with_init_delegates_to_repeat():
    s = ScanWith("s", [1, 2, 3], val_reg="v")
    prog = _make_dmem_prog()
    prog.add_reg = MagicMock()
    s.init(prog)
    # Repeat.init calls prog.add_reg for counter_reg
    prog.add_reg.assert_called()


def test_scan_with_run_delegates_to_repeat():
    s = ScanWith("s", [1, 2, 3], val_reg="v")
    # replace inner repeat with a mock
    inner = MagicMock()
    inner.run.return_value = 0.0
    s.repeat_mod = inner
    prog = _make_dmem_prog()
    s.run(prog, t=0.5)
    inner.run.assert_called_once_with(prog, 0.5)


def test_scan_with_add_content_delegates_to_repeat():
    from zcu_tools.program.v2.modules.delay import SoftDelay

    s = ScanWith("s", [1, 2, 3], val_reg="v")
    child = SoftDelay("extra", 0.0)
    s.add_content(child)
    # LoadValue is first, child is appended after it
    assert child in s.repeat_mod.sub_modules
    assert len(s.repeat_mod.sub_modules) == 2


# ---------------------------------------------------------------------------
# _plan_compression — edge cases
# ---------------------------------------------------------------------------


def test_large_value_max_17bits_not_compressed():
    # max value = 65536 (17-bit) → bits rounds to 32 → values_per_word = 1 < 2 → no compression
    values = [65536] * _COMPRESS_MIN_VALUES
    lv = LoadValue("x", values, idx_reg="i", val_reg="v")
    assert lv._is_compressed is False


def test_large_value_max_16bits_gets_compressed():
    # max value = 65535 (16-bit) → bits = 16 → values_per_word = 2 → compressed
    values = [65535] * _COMPRESS_MIN_VALUES
    lv = LoadValue("x", values, idx_reg="i", val_reg="v")
    assert lv._is_compressed is True
    assert lv._bits_per_value == 16
    assert lv._values_per_word == 2


def test_bits_needed_rejects_negative():
    with pytest.raises(ValueError, match="non-negative"):
        LoadValue._bits_needed(-1)
