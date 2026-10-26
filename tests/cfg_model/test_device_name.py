"""Instrument-name validation at the shared configuration boundary."""

import pytest
from pydantic import TypeAdapter, ValidationError
from zcu_tools.cfg_model import (
    ConfigBase,
    DeviceName,
    is_device_name,
)


class DeviceCfg(ConfigBase):
    required: DeviceName
    optional: DeviceName | None = None


@pytest.mark.parametrize(
    "name", ["coil.left", ".coil", "coil.", ".", "", " ", "\t\n", "\u00a0\u3000"]
)
@pytest.mark.parametrize("field", ["required", "optional"])
def test_device_name_rejects_invalid_names_in_required_and_nullable_fields(
    name: str, field: str
) -> None:
    with pytest.raises(ValidationError) as failure:
        DeviceCfg.model_validate({"required": "coil", field: name})

    assert repr(name) in str(failure.value)
    assert failure.value.errors()[0]["loc"] == (field,)


@pytest.mark.parametrize("name", ["coil", " Coil/A-1 ", "線圈", "coil．left"])
def test_device_name_keeps_other_strings_nullable_marker_and_serialization(
    name: str,
) -> None:
    cfg = DeviceCfg(required=name, optional=name)
    assert cfg.model_dump(mode="json") == {"required": name, "optional": name}
    assert DeviceCfg.model_validate_json(cfg.model_dump_json()) == cfg
    assert DeviceCfg(required=name).optional is None
    assert is_device_name(DeviceCfg.model_fields["required"])
    assert is_device_name(DeviceCfg.model_fields["optional"])
    adapter = TypeAdapter(DeviceName)
    assert adapter.validate_json(adapter.dump_json(name)) == name
