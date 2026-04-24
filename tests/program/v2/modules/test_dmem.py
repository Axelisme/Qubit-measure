import pytest
from zcu_tools.program.v2.modules.dmem import LoadValue


def test_small_values_not_compressed():
    lv = LoadValue("x", [1, 2, 3], idx_reg="i", val_reg="v")
    assert lv._is_compressed is False
    assert lv._packed_values == [1, 2, 3]
    assert lv.allow_rerun() is True


def test_large_small_range_gets_compressed():
    values = [i % 4 for i in range(64)]  # 2 bits, 64 values
    lv = LoadValue("x", values, idx_reg="i", val_reg="v")
    assert lv._is_compressed is True
    assert lv._signed_mode is False
    assert lv._bits_per_value == 2
    assert lv._values_per_word == 16
    assert len(lv._packed_values) == 64 // 16
    # unsigned + compressed is allowed to rerun
    assert lv.allow_rerun() is True


def test_signed_compression_blocks_rerun():
    values = [(-1 if i % 2 else 1) for i in range(64)]
    lv = LoadValue("x", values, idx_reg="i", val_reg="v")
    assert lv._is_compressed is True
    assert lv._signed_mode is True
    assert lv.allow_rerun() is False


def test_auto_compress_off_keeps_values():
    values = [i % 4 for i in range(64)]
    lv = LoadValue("x", values, idx_reg="i", val_reg="v", auto_compress=False)
    assert lv._is_compressed is False
    assert lv._packed_values == values


def test_empty_values_rejected():
    with pytest.raises(ValueError):
        LoadValue("x", [], idx_reg="i", val_reg="v")


def test_bits_needed_unsigned_boundaries():
    assert LoadValue._bits_needed_unsigned(0) == 1
    assert LoadValue._bits_needed_unsigned(1) == 1
    assert LoadValue._bits_needed_unsigned(2) == 2
    assert LoadValue._bits_needed_unsigned(255) == 8


def test_bits_needed_signed_boundaries():
    # [-2,1] fits in 2 bits signed
    assert LoadValue._bits_needed_signed(-2, 1) == 2
    # [-128, 127] fits in 8 bits
    assert LoadValue._bits_needed_signed(-128, 127) == 8
