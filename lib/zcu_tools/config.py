from __future__ import annotations

from copy import deepcopy

from pydantic import BaseModel, ConfigDict
from typing_extensions import Any, Self


class ConfigBase(BaseModel):
    model_config = ConfigDict(
        extra="forbid", arbitrary_types_allowed=True, validate_assignment=True
    )

    def with_updates(self, **kwargs) -> Self:
        cfg = deepcopy(self)

        def deepupdate_cfg(cfg: ConfigBase, d: dict[str, Any]) -> None:
            for key, value in d.items():
                if isinstance(value, dict):
                    deepupdate_cfg(getattr(cfg, key), value)
                else:
                    setattr(cfg, key, value)

        deepupdate_cfg(cfg, kwargs)
        return cfg.model_validate(cfg)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="python", by_alias=True, exclude_none=True)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(\n{self.to_dict()}\n)"

    def __str__(self) -> str:
        def format_value(value: Any, indent: int) -> str:
            pad = "    " * indent
            inner_pad = "    " * (indent + 1)
            if isinstance(value, ConfigBase):
                fields = list(value.__class__.model_fields.keys())
                if not fields:
                    return f"{value.__class__.__name__}()"
                lines = [f"{value.__class__.__name__}("]
                for name in fields:
                    attr = getattr(value, name)
                    lines.append(
                        f"{inner_pad}{name} = {format_value(attr, indent + 1)},"
                    )
                lines.append(f"{pad})")
                return "\n".join(lines)
            if isinstance(value, dict):
                if not value:
                    return "{}"
                lines = ["{"]
                for k, v in value.items():
                    lines.append(
                        f"{inner_pad}{k!r}: {format_value(v, indent + 1)},"
                    )
                lines.append(f"{pad}}}")
                return "\n".join(lines)
            if isinstance(value, list):
                if not value:
                    return "[]"
                if all(
                    not isinstance(v, (ConfigBase, dict, list)) for v in value
                ):
                    return repr(value)
                lines = ["["]
                for v in value:
                    lines.append(f"{inner_pad}{format_value(v, indent + 1)},")
                lines.append(f"{pad}]")
                return "\n".join(lines)
            return repr(value)

        return format_value(self, 0)
