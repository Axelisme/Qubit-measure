from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from tempfile import gettempdir
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
    ModuleRefSpec,
    ModuleRefValue,
    MultiSweepSpec,
    MultiSweepValue,
    SavePaths,
    ScalarSpec,
    SweepSpec,
    SweepValue,
    WaveformRefSpec,
    WaveformRefValue,
    make_default_value,
    schema_to_dict,
)

logger = logging.getLogger(__name__)


_SESSION_VERSION = 1
_SESSION_FILENAME = "tab_session_v1.json"


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
        self._session_path.parent.mkdir(parents=True, exist_ok=True)
        self._session_path.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )

    def load_session(self) -> Optional[PersistedSession]:
        if not self._session_path.exists():
            return None
        data = json.loads(self._session_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise RuntimeError("Session file payload must be a JSON object")
        if data.get("version") != _SESSION_VERSION:
            raise RuntimeError("Unsupported session version")
        raw_tabs = data.get("tabs")
        if not isinstance(raw_tabs, list):
            raise RuntimeError("Session tabs must be a list")
        tabs: list[PersistedTab] = []
        for item in raw_tabs:
            if not isinstance(item, dict):
                raise RuntimeError("Session tab entry must be a JSON object")
            adapter_name = item.get("adapter_name")
            cfg_raw = item.get("cfg_raw")
            raw_override = item.get("save_paths_override")
            if not isinstance(adapter_name, str) or not adapter_name:
                raise RuntimeError("Session tab adapter_name must be non-empty")
            if not isinstance(cfg_raw, dict):
                raise RuntimeError("Session tab cfg_raw must be a JSON object")
            override: Optional[SavePaths]
            if raw_override is None:
                override = None
            elif isinstance(raw_override, dict):
                data_path = raw_override.get("data_path")
                image_path = raw_override.get("image_path")
                if not isinstance(data_path, str) or not isinstance(image_path, str):
                    raise RuntimeError(
                        "Session save_paths_override must contain string paths"
                    )
                override = SavePaths(data_path=data_path, image_path=image_path)
            else:
                raise RuntimeError("Session save_paths_override must be an object")
            tabs.append(
                PersistedTab(
                    adapter_name=adapter_name,
                    cfg_raw=cfg_raw,
                    save_paths_override=override,
                )
            )
        active_tab_index = data.get("active_tab_index")
        if active_tab_index is not None and not isinstance(active_tab_index, int):
            raise RuntimeError("Session active_tab_index must be an integer")
        return PersistedSession(
            version=_SESSION_VERSION,
            tabs=tabs,
            active_tab_index=active_tab_index,
        )

    def schema_to_raw(self, schema: CfgSchema, *, ml: Any) -> dict[str, object]:
        return schema_to_dict(schema, ml)

    def raw_to_schema(
        self, base_schema: CfgSchema, raw_cfg: dict[str, object]
    ) -> CfgSchema:
        value = self._section_value_from_raw(base_schema.spec, raw_cfg)
        return CfgSchema(spec=base_schema.spec, value=value)

    def _section_value_from_raw(
        self,
        spec: CfgSectionSpec,
        raw: dict[str, object],
    ) -> CfgSectionValue:
        value = make_default_value(spec)
        for key, node_spec in spec.fields.items():
            if key not in raw:
                continue
            try:
                parsed = self._node_value_from_raw(node_spec, raw[key])
            except Exception:
                logger.warning(
                    "restore schema field failed: key=%r spec=%s",
                    key,
                    type(node_spec).__name__,
                    exc_info=True,
                )
                continue
            if parsed is not None:
                value.fields[key] = parsed
        return value

    def _node_value_from_raw(
        self,
        spec: CfgNodeSpec,
        raw: object,
    ) -> Optional[CfgNodeValue]:
        if isinstance(spec, ScalarSpec):
            if isinstance(raw, str) and raw.strip().startswith("="):
                return EvalValue(expr=raw.strip(), resolved=None, error=None)
            return DirectValue(raw)
        if isinstance(spec, SweepSpec):
            if isinstance(raw, dict):
                start = self._parse_sweep_edge(raw["start"])
                stop = self._parse_sweep_edge(raw["stop"])
                expts = int(raw["expts"])
                step_raw = raw.get("step")
                if step_raw is None:
                    if expts == 1:
                        step = 0.0
                    else:
                        start_f = self._sweep_edge_resolved_float(start, "start")
                        stop_f = self._sweep_edge_resolved_float(stop, "stop")
                        step = (stop_f - start_f) / (expts - 1)
                else:
                    step = float(step_raw)
                return SweepValue(start=start, stop=stop, expts=expts, step=step)
            raise RuntimeError("Sweep payload must be an object")
        if isinstance(spec, MultiSweepSpec):
            if not isinstance(raw, dict):
                raise RuntimeError("MultiSweep payload must be an object")
            axes: dict[str, SweepValue] = {}
            for axis, axis_spec in spec.axes.items():
                axis_raw = raw.get(axis)
                if not isinstance(axis_raw, dict):
                    continue
                parsed = self._node_value_from_raw(axis_spec, axis_raw)
                if isinstance(parsed, SweepValue):
                    axes[axis] = parsed
            return MultiSweepValue(axes=axes)
        if isinstance(spec, DeviceRefSpec):
            if not isinstance(raw, str):
                raise RuntimeError("Device reference must be string")
            return DirectValue(raw)
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
        if isinstance(raw, str) and raw.strip().startswith("="):
            return EvalValue(expr=raw.strip(), resolved=None, error=None)
        if isinstance(raw, (int, float)):
            return float(raw)
        raise RuntimeError("Sweep edge must be numeric or '=expr'")

    def _sweep_edge_resolved_float(self, value: object, edge_name: str) -> float:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, EvalValue):
            if isinstance(value.resolved, (int, float)):
                return float(value.resolved)
            raise RuntimeError(f"Sweep {edge_name} expression is unresolved")
        raise RuntimeError(f"Sweep {edge_name} must be numeric")

    def _ref_value_from_raw(
        self,
        spec: ModuleRefSpec,
        raw: object,
    ) -> ModuleRefValue:
        if not isinstance(raw, dict):
            raise RuntimeError("Module reference payload must be an object")
        for allowed_spec in spec.allowed:
            try:
                nested = self._section_value_from_raw(allowed_spec, raw)
                return ModuleRefValue(
                    chosen_key=f"<Custom:{allowed_spec.label}>",
                    value=nested,
                )
            except Exception:
                continue
        fallback_spec = spec.allowed[0]
        return ModuleRefValue(
            chosen_key=f"<Custom:{fallback_spec.label}>",
            value=make_default_value(fallback_spec),
        )

    def _waveform_ref_value_from_raw(
        self,
        spec: WaveformRefSpec,
        raw: object,
    ) -> WaveformRefValue:
        if not isinstance(raw, dict):
            raise RuntimeError("Waveform reference payload must be an object")
        for allowed_spec in spec.allowed:
            try:
                nested = self._section_value_from_raw(allowed_spec, raw)
                return WaveformRefValue(
                    chosen_key=f"<Custom:{allowed_spec.label}>",
                    value=nested,
                )
            except Exception:
                continue
        fallback_spec = spec.allowed[0]
        return WaveformRefValue(
            chosen_key=f"<Custom:{fallback_spec.label}>",
            value=make_default_value(fallback_spec),
        )


__all__ = [
    "PersistedSession",
    "PersistedTab",
    "SessionPersistenceService",
]
