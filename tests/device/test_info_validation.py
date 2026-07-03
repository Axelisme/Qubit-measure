from __future__ import annotations

from typing import Any, cast

import pytest
from pydantic import ValidationError
from zcu_tools.device.fake import FakeDeviceInfo
from zcu_tools.device.yoko import YOKOGS200Info


def test_fake_device_info_rejects_bool_value() -> None:
    with pytest.raises(ValidationError, match="real numeric scalar"):
        FakeDeviceInfo(address="none", value=cast(Any, True))


def test_fake_device_info_rejects_string_value() -> None:
    with pytest.raises(ValidationError, match="real numeric scalar"):
        FakeDeviceInfo(address="none", value=cast(Any, "1.23"))


def test_yoko_info_rejects_bool_value() -> None:
    with pytest.raises(ValidationError, match="real numeric scalar"):
        YOKOGS200Info(address="GPIB::1", value=cast(Any, True))


def test_yoko_info_rejects_string_value() -> None:
    with pytest.raises(ValidationError, match="real numeric scalar"):
        YOKOGS200Info(address="GPIB::1", value=cast(Any, "1.23"))


def test_fake_device_info_accepts_int_and_float_values() -> None:
    assert FakeDeviceInfo(address="none", value=1).value == pytest.approx(1.0)
    assert FakeDeviceInfo(address="none", value=1.23).value == pytest.approx(1.23)


def test_yoko_info_accepts_int_and_float_values() -> None:
    assert YOKOGS200Info(address="GPIB::1", value=1).value == pytest.approx(1.0)
    assert YOKOGS200Info(address="GPIB::1", value=1.23).value == pytest.approx(1.23)
