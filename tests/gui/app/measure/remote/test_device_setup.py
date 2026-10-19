"""Complete device.setup requests at the public dispatch/model seam."""

from collections.abc import Mapping
from dataclasses import dataclass, field

import pytest
from zcu_tools.device.base import BaseDeviceInfo
from zcu_tools.device.fake import FakeDeviceInfo
from zcu_tools.device.yoko import YOKOGS200Info
from zcu_tools.gui.app.measure.remote.dispatch import METHOD_REGISTRY
from zcu_tools.gui.remote.errors import ErrorCode, RemoteError
from zcu_tools.gui.remote.param_spec import validate_params
from zcu_tools.gui.session.device_models import SetupDeviceRequest


@dataclass
class RecordingDeviceControl:
    info: BaseDeviceInfo
    requests: list[SetupDeviceRequest] = field(default_factory=list)

    def get_device_info(self, name: str) -> BaseDeviceInfo:
        return self.info

    def start_setup_device(self, req: SetupDeviceRequest) -> int:
        self.requests.append(req)
        return 71


@dataclass
class SetupAdapter:
    device_control: RecordingDeviceControl


def submit(
    dev: RecordingDeviceControl, cfg: Mapping[str, object]
) -> Mapping[str, object]:
    method = METHOD_REGISTRY["device.setup"]
    params = validate_params(method.params, {"name": "bias", "cfg": dict(cfg)})
    return method.handler(SetupAdapter(dev), params)


def accept(
    dev: RecordingDeviceControl, cfg: Mapping[str, object]
) -> Mapping[str, object]:
    try:
        return submit(dev, cfg)
    except RemoteError as exc:
        pytest.fail(f"valid complete cfg must be accepted: {exc}")


def yoko_control() -> RecordingDeviceControl:
    return RecordingDeviceControl(
        YOKOGS200Info(address="cached", mode="current", value=0.002)
    )


def valid_current_cfg() -> dict[str, object]:
    return {"address": "requested", "mode": "current", "value": 0.001}


def reject_after_witness(
    dev: RecordingDeviceControl,
    valid: dict[str, object],
    invalid: dict[str, object],
    diagnostic: str,
) -> None:
    assert accept(dev, valid) == {"operation_id": 71}
    assert len(dev.requests) == 1
    with pytest.raises(RemoteError, match=diagnostic) as error:
        submit(dev, invalid)
    assert error.value.code == ErrorCode.INVALID_PARAMS
    assert len(dev.requests) == 1


@pytest.mark.parametrize(
    "kind",
    ["fake-defaults", "yoko-current-implicit-unit", "yoko-voltage-explicit-unit"],
)
def test_complete_cfg_returns_operation_without_merging_cached_values(
    kind: str,
) -> None:
    if kind == "fake-defaults":
        dev = RecordingDeviceControl(FakeDeviceInfo(address="cached", value=9.0))
        assert accept(dev, {"address": "requested"}) == {"operation_id": 71}
        assert dev.requests[0].info.model_dump()["value"] == 0.0
        assert accept(dev, {"address": "requested", "value": 1.5}) == {
            "operation_id": 71
        }
        assert dev.requests[1].info.model_dump()["value"] == 1.5
    else:
        dev = yoko_control()
        cfg = (
            valid_current_cfg()
            if kind == "yoko-current-implicit-unit"
            else {
                "address": "requested",
                "mode": "voltage",
                "value_unit": "V",
                "value": 1.5,
            }
        )
        assert accept(dev, cfg) == {"operation_id": 71}
        info = dev.requests[0].info.model_dump()
        assert info["value"] == cfg["value"]
        assert info["value_unit"] == ("A" if cfg["mode"] == "current" else "V")
        assert info["rampstep"] == (1e-6 if cfg["mode"] == "current" else 1e-3)
    assert all(
        req.name == "bias" and req.info.address == "requested" for req in dev.requests
    )
    assert dev.info.address == "cached"


@pytest.mark.parametrize(
    "missing", ["address", "mode"], ids=["missing-address", "missing-yoko-mode"]
)
def test_incomplete_cfg_rejects_before_setup(missing: str) -> None:
    cfg = valid_current_cfg()
    invalid = dict(cfg)
    del invalid[missing]
    reject_after_witness(yoko_control(), cfg, invalid, missing)


@pytest.mark.parametrize(
    "patch,diagnostic",
    [({"extra": 2}, "extra"), ({"type": "YOKOGS200"}, "FakeDevice")],
    ids=["fake-extra", "wrong-model-type"],
)
def test_invalid_cfg_field_rejects_before_setup(
    patch: dict[str, object], diagnostic: str
) -> None:
    valid: dict[str, object] = {"address": "none", "value": 1.0}
    reject_after_witness(
        RecordingDeviceControl(FakeDeviceInfo(address="none")),
        valid,
        {**valid, **patch},
        diagnostic,
    )


@pytest.mark.parametrize(
    "mode,unit", [("current", "V"), ("voltage", "A")], ids=["current-V", "voltage-A"]
)
def test_conflicting_cfg_unit_rejects_before_setup(mode: str, unit: str) -> None:
    valid: dict[str, object] = {"address": "requested", "mode": mode, "value": 0.001}
    reject_after_witness(
        yoko_control(), valid, {**valid, "value_unit": unit}, "value unit.*mode"
    )


@pytest.mark.parametrize(
    "invalid",
    [True, "1", float("inf"), float("nan"), 1.5],
    ids=["bool", "string", "infinity", "nan", "current-over-limit"],
)
def test_invalid_cfg_value_rejects_before_setup(invalid: object) -> None:
    valid = valid_current_cfg()
    reject_after_witness(yoko_control(), valid, {**valid, "value": invalid}, "value")
