from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile, gettempdir
from typing import Any, Optional, Union

from platformdirs import user_cache_dir

from zcu_tools.gui.adapter import (
    CfgNodeSpec,
    CfgNodeValue,
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    DeviceRefSpec,
    DirectValue,
    EvalValue,
    LiteralSpec,
    ModuleRefSpec,
    ModuleRefValue,
    SavePaths,
    ScalarSpec,
    SweepSpec,
    SweepValue,
    WaveformRefSpec,
    WaveformRefValue,
    make_default_value,
)

logger = logging.getLogger(__name__)


# Retain the existing filename so an old payload can be rejected explicitly.
SESSION_VERSION = 2
_SESSION_FILENAME = "tab_session_v1.json"


class SessionPersistenceError(RuntimeError):
    """Expected failure while reading, writing, or restoring a GUI session."""


@dataclass(frozen=True)
class PersistedTab:
    adapter_name: str
    cfg_raw: dict[str, object]
    save_paths_override: Optional[SavePaths]


@dataclass(frozen=True)
class PersistedSession:
    version: int
    tabs: list[PersistedTab]
    active_tab_index: Optional[int]


def _safe_cache_root() -> Path:
    try:
        return Path(user_cache_dir("zcu_tools", "zcu_tools")) / "gui"
    except Exception:
        return Path(gettempdir()) / ".zcu_tools_gui"


def _to_json_compatible(value: object) -> object:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_to_json_compatible(item) for item in value]
    if isinstance(value, tuple):
        return [_to_json_compatible(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _to_json_compatible(v) for k, v in value.items()}
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _to_json_compatible(to_dict())
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _to_json_compatible(model_dump())
    return str(value)


class SessionPersistenceService:
    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        base_dir = cache_dir if cache_dir is not None else _safe_cache_root()
        self._session_path = base_dir / _SESSION_FILENAME

    @property
    def session_path(self) -> Path:
        return self._session_path

    def save_session(self, session: PersistedSession) -> None:
        if session.version != SESSION_VERSION:
            raise SessionPersistenceError(
                f"Unsupported session version for save: {session.version!r}; "
                f"expected {SESSION_VERSION}"
            )
        payload = {
            "version": session.version,
            "active_tab_index": session.active_tab_index,
            "tabs": [
                {
                    "adapter_name": tab.adapter_name,
                    "cfg_raw": _to_json_compatible(tab.cfg_raw),
                    "save_paths_override": (
                        {
                            "data_path": tab.save_paths_override.data_path,
                            "image_path": tab.save_paths_override.image_path,
                        }
                        if tab.save_paths_override is not None
                        else None
                    ),
                }
                for tab in session.tabs
            ],
        }
        self._write_payload(payload)

    def load_session(self) -> Optional[PersistedSession]:
        if not self._session_path.exists():
            return None
        try:
            data = json.loads(self._session_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SessionPersistenceError(
                f"Failed to read session settings: {exc}"
            ) from exc
        if not isinstance(data, dict):
            raise SessionPersistenceError("Session file payload must be a JSON object")
        if data.get("version") != SESSION_VERSION:
            raise SessionPersistenceError(
                f"Unsupported session version: {data.get('version')!r}; "
                f"expected {SESSION_VERSION}"
            )
        raw_tabs = data.get("tabs")
        if not isinstance(raw_tabs, list):
            raise SessionPersistenceError("Session tabs must be a list")
        tabs: list[PersistedTab] = []
        for item in raw_tabs:
            if not isinstance(item, dict):
                raise SessionPersistenceError("Session tab entry must be a JSON object")
            adapter_name = item.get("adapter_name")
            cfg_raw = item.get("cfg_raw")
            raw_override = item.get("save_paths_override")
            if not isinstance(adapter_name, str) or not adapter_name:
                raise SessionPersistenceError(
                    "Session tab adapter_name must be non-empty"
                )
            if not isinstance(cfg_raw, dict):
                raise SessionPersistenceError(
                    "Session tab cfg_raw must be a JSON object"
                )
            override: Optional[SavePaths]
            if raw_override is None:
                override = None
            elif isinstance(raw_override, dict):
                data_path = raw_override.get("data_path")
                image_path = raw_override.get("image_path")
                if not isinstance(data_path, str) or not isinstance(image_path, str):
                    raise SessionPersistenceError(
                        "Session save_paths_override must contain string paths"
                    )
                override = SavePaths(data_path=data_path, image_path=image_path)
            else:
                raise SessionPersistenceError(
                    "Session save_paths_override must be an object"
                )
            tabs.append(
                PersistedTab(
                    adapter_name=adapter_name,
                    cfg_raw=cfg_raw,
                    save_paths_override=override,
                )
            )
        active_tab_index = data.get("active_tab_index")
        if active_tab_index is not None and not isinstance(active_tab_index, int):
            raise SessionPersistenceError("Session active_tab_index must be an integer")
        return PersistedSession(
            version=SESSION_VERSION,
            tabs=tabs,
            active_tab_index=active_tab_index,
        )

    def _write_payload(self, payload: dict[str, object]) -> None:
        temp_path: Optional[Path] = None
        try:
            self._session_path.parent.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=self._session_path.parent,
                delete=False,
            ) as file:
                file.write(json.dumps(payload, ensure_ascii=True, indent=2))
                temp_path = Path(file.name)
            temp_path.replace(self._session_path)
        except (OSError, TypeError, ValueError) as exc:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
            raise SessionPersistenceError(
                f"Failed to save session settings: {exc}"
            ) from exc

    def schema_to_raw(self, schema: CfgSchema, *, ml: Any) -> dict[str, object]:
        del ml
        return self._section_value_to_raw(schema.spec, schema.value)

    def raw_to_schema(
        self, base_schema: CfgSchema, raw_cfg: dict[str, object]
    ) -> CfgSchema:
        try:
            value = self._section_value_from_raw(base_schema.spec, raw_cfg)
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            raise SessionPersistenceError(
                f"Invalid session cfg payload: {exc}"
            ) from exc
        return CfgSchema(spec=base_schema.spec, value=value)

    def _section_value_to_raw(
        self, spec: CfgSectionSpec, value: CfgSectionValue
    ) -> dict[str, object]:
        payload: dict[str, object] = {}
        for key, node_spec in spec.fields.items():
            node_val = value.fields.get(key)
            if node_val is None:
                continue
            payload[key] = self._node_value_to_raw(node_spec, node_val)
        return payload

    def _node_value_to_raw(self, spec: CfgNodeSpec, value: CfgNodeValue) -> object:
        if isinstance(spec, LiteralSpec):
            # Fixed-value field; the literal value is canonical.
            return {
                "__kind": "direct",
                "value": _to_json_compatible(spec.value),
                "is_unset": False,
            }
        if isinstance(spec, ScalarSpec):
            assert isinstance(value, (DirectValue, EvalValue))
            if isinstance(value, EvalValue):
                return {
                    "__kind": "eval",
                    "expr": value.expr,
                }
            return {
                "__kind": "direct",
                "value": _to_json_compatible(value.value),
                "is_unset": value.is_unset,
            }
        if isinstance(spec, SweepSpec):
            assert isinstance(value, SweepValue)
            return {
                "start": self._sweep_edge_to_raw(value.start),
                "stop": self._sweep_edge_to_raw(value.stop),
                "expts": value.expts,
                "step": value.step,
            }
        if isinstance(spec, DeviceRefSpec):
            assert isinstance(value, DirectValue)
            return {
                "__kind": "direct",
                "value": _to_json_compatible(value.value),
                "is_unset": value.is_unset,
            }
        if isinstance(spec, CfgSectionSpec):
            assert isinstance(value, CfgSectionValue)
            return self._section_value_to_raw(spec, value)
        if isinstance(spec, ModuleRefSpec):
            assert isinstance(value, ModuleRefValue)
            return {
                "__kind": "module_ref",
                "chosen_key": value.chosen_key,
                "is_overridden": value.is_overridden,
                "value": self._section_value_to_raw(
                    self._select_allowed_spec_for_restore(spec, value.chosen_key),
                    value.value,
                ),
            }
        if isinstance(spec, WaveformRefSpec):
            assert isinstance(value, WaveformRefValue)
            return {
                "__kind": "waveform_ref",
                "chosen_key": value.chosen_key,
                "is_overridden": value.is_overridden,
                "value": self._section_value_to_raw(
                    self._select_allowed_spec_for_restore(spec, value.chosen_key),
                    value.value,
                ),
            }
        return _to_json_compatible(value)

    def _sweep_edge_to_raw(self, value: Union[float, EvalValue]) -> object:
        if isinstance(value, EvalValue):
            return {"__kind": "eval", "expr": value.expr}
        return float(value)

    def _section_value_from_raw(
        self,
        spec: CfgSectionSpec,
        raw: dict[str, object],
    ) -> CfgSectionValue:
        value = make_default_value(spec)
        for key, node_spec in spec.fields.items():
            if key not in raw:
                continue
            parsed = self._node_value_from_raw(node_spec, raw[key])
            if parsed is not None:
                value.fields[key] = parsed
        return value

    def _node_value_from_raw(
        self,
        spec: CfgNodeSpec,
        raw: object,
    ) -> Optional[CfgNodeValue]:
        if isinstance(spec, ScalarSpec):
            if (
                isinstance(raw, dict)
                and raw.get("__kind") == "eval"
                and isinstance(raw.get("expr"), str)
            ):
                return EvalValue(expr=raw["expr"], resolved=None, error=None)
            if isinstance(raw, dict) and raw.get("__kind") == "direct":
                value = raw.get("value")
                is_unset = bool(raw.get("is_unset", False))
                return DirectValue(value=value, is_unset=is_unset)
            if isinstance(raw, str) and raw.strip().startswith("="):
                raise RuntimeError("Legacy scalar '=expr' payload is unsupported")
            return DirectValue(raw)
        if isinstance(spec, SweepSpec):
            if isinstance(raw, dict):
                start = self._parse_sweep_edge(raw["start"])
                stop = self._parse_sweep_edge(raw["stop"])
                expts = int(raw["expts"])
                step_raw = raw.get("step")
                if step_raw is None:
                    raise RuntimeError("Sweep step is required in session payload")
                step = float(step_raw)
                return SweepValue(start=start, stop=stop, expts=expts, step=step)
            raise RuntimeError("Sweep payload must be an object")
        if isinstance(spec, DeviceRefSpec):
            if isinstance(raw, dict) and raw.get("__kind") == "direct":
                value = raw.get("value")
                if not isinstance(value, str):
                    raise RuntimeError("Device reference value must be string")
                return DirectValue(value, is_unset=bool(raw.get("is_unset", False)))
            raise RuntimeError("Device reference must use direct payload encoding")
        if isinstance(spec, CfgSectionSpec):
            if not isinstance(raw, dict):
                raise RuntimeError("Section payload must be an object")
            return self._section_value_from_raw(spec, raw)
        if isinstance(spec, ModuleRefSpec):
            return self._ref_value_from_raw(spec, raw)
        if isinstance(spec, WaveformRefSpec):
            return self._waveform_ref_value_from_raw(spec, raw)
        return None

    def _parse_sweep_edge(self, raw: object) -> Union[float, EvalValue]:
        if (
            isinstance(raw, dict)
            and raw.get("__kind") == "eval"
            and isinstance(raw.get("expr"), str)
        ):
            return EvalValue(expr=raw["expr"], resolved=None, error=None)
        if isinstance(raw, str) and raw.strip().startswith("="):
            raise RuntimeError("Legacy sweep '=expr' payload is unsupported")
        if isinstance(raw, (int, float)):
            return float(raw)
        raise RuntimeError("Sweep edge must be numeric or '=expr'")

    def _ref_value_from_raw(
        self,
        spec: ModuleRefSpec,
        raw: object,
    ) -> ModuleRefValue:
        if (
            isinstance(raw, dict)
            and raw.get("__kind") == "module_ref"
            and isinstance(raw.get("chosen_key"), str)
            and isinstance(raw.get("value"), dict)
        ):
            chosen_key = raw["chosen_key"]
            value_spec = self._select_allowed_spec_for_restore(spec, chosen_key)
            nested = self._section_value_from_raw(value_spec, raw["value"])
            return ModuleRefValue(
                chosen_key=chosen_key,
                value=nested,
                is_overridden=bool(raw.get("is_overridden", False)),
            )
        raise RuntimeError("Module reference must use module_ref payload encoding")

    def _waveform_ref_value_from_raw(
        self,
        spec: WaveformRefSpec,
        raw: object,
    ) -> WaveformRefValue:
        if (
            isinstance(raw, dict)
            and raw.get("__kind") == "waveform_ref"
            and isinstance(raw.get("chosen_key"), str)
            and isinstance(raw.get("value"), dict)
        ):
            chosen_key = raw["chosen_key"]
            value_spec = self._select_allowed_spec_for_restore(spec, chosen_key)
            nested = self._section_value_from_raw(value_spec, raw["value"])
            return WaveformRefValue(
                chosen_key=chosen_key,
                value=nested,
                is_overridden=bool(raw.get("is_overridden", False)),
            )
        raise RuntimeError("Waveform reference must use waveform_ref payload encoding")

    def _select_allowed_spec_for_restore(
        self, spec: Union[ModuleRefSpec, WaveformRefSpec], chosen_key: str
    ) -> CfgSectionSpec:
        if chosen_key.startswith("<Custom:") and chosen_key.endswith(">"):
            label = chosen_key[len("<Custom:") : -1]
            for allowed_spec in spec.allowed:
                if allowed_spec.label == label:
                    return allowed_spec
        return spec.allowed[0]


__all__ = [
    "PersistedSession",
    "PersistedTab",
    "SESSION_VERSION",
    "SessionPersistenceError",
    "SessionPersistenceService",
]
