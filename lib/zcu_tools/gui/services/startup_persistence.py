"""Startup settings persistence — remembers chip/qub/res names, IP/port, and devices."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import gettempdir
from typing import Optional

from platformdirs import user_cache_dir

logger = logging.getLogger(__name__)

_STARTUP_VERSION = 2
_STARTUP_FILENAME = "startup_v2.json"

_DEFAULT_LEFT_PANEL_WIDTH = 500


@dataclass(frozen=True)
class PersistedDeviceEntry:
    type_name: str
    name: str
    address: str


@dataclass(frozen=True)
class PersistedStartup:
    version: int
    chip_name: str
    qub_name: str
    res_name: str
    result_dir: str
    database_path: str
    ip: str
    port: int
    devices: list[PersistedDeviceEntry]
    left_panel_width: int = _DEFAULT_LEFT_PANEL_WIDTH


def _safe_cache_root() -> Path:
    try:
        return Path(user_cache_dir("zcu_tools", "zcu_tools")) / "gui"
    except Exception:
        return Path(gettempdir()) / ".zcu_tools_gui"


def _make_default() -> PersistedStartup:
    return PersistedStartup(
        version=_STARTUP_VERSION,
        chip_name="",
        qub_name="",
        res_name="",
        result_dir="",
        database_path="",
        ip="192.168.10.1",
        port=8887,
        devices=[],
        left_panel_width=_DEFAULT_LEFT_PANEL_WIDTH,
    )


class StartupPersistenceService:
    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        base_dir = cache_dir if cache_dir is not None else _safe_cache_root()
        self._path = base_dir / _STARTUP_FILENAME
        self._current: PersistedStartup = _make_default()

    def load(self) -> Optional[PersistedStartup]:
        if not self._path.exists():
            return None
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("payload must be a JSON object")
            version = raw.get("version")
            if version not in (1, _STARTUP_VERSION):
                raise ValueError(f"unsupported version: {version!r}")
            devices: list[PersistedDeviceEntry] = []
            for entry in raw.get("devices", []):
                if not isinstance(entry, dict):
                    continue
                type_name = entry.get("type_name", "")
                name = entry.get("name", "")
                address = entry.get("address", "")
                if not isinstance(type_name, str) or not isinstance(name, str):
                    continue
                if not type_name or not name:
                    continue
                devices.append(
                    PersistedDeviceEntry(
                        type_name=type_name,
                        name=name,
                        address=str(address),
                    )
                )
            result = PersistedStartup(
                version=_STARTUP_VERSION,
                chip_name=str(raw.get("chip_name", "")),
                qub_name=str(raw.get("qub_name", "")),
                res_name=str(raw.get("res_name", "")),
                result_dir=str(raw.get("result_dir", "")),
                database_path=str(raw.get("database_path", "")),
                ip=str(raw.get("ip", "192.168.10.1")),
                port=int(raw.get("port", 8887)),
                devices=devices,
                left_panel_width=int(
                    raw.get("left_panel_width", _DEFAULT_LEFT_PANEL_WIDTH)
                ),
            )
            self._current = result
            return result
        except Exception:
            logger.warning(
                "Failed to load startup settings from %s", self._path, exc_info=True
            )
            return None

    def save(self, data: PersistedStartup) -> None:
        self._current = data
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": data.version,
                "chip_name": data.chip_name,
                "qub_name": data.qub_name,
                "res_name": data.res_name,
                "result_dir": data.result_dir,
                "database_path": data.database_path,
                "ip": data.ip,
                "port": data.port,
                "devices": [asdict(d) for d in data.devices],
                "left_panel_width": data.left_panel_width,
            }
            self._path.write_text(
                json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8"
            )
        except Exception:
            logger.warning(
                "Failed to save startup settings to %s", self._path, exc_info=True
            )

    def get_current(self) -> PersistedStartup:
        return self._current

    def update_project(
        self,
        *,
        chip_name: str,
        qub_name: str,
        res_name: str,
        result_dir: str,
        database_path: str,
    ) -> None:
        updated = PersistedStartup(
            version=_STARTUP_VERSION,
            chip_name=chip_name,
            qub_name=qub_name,
            res_name=res_name,
            result_dir=result_dir,
            database_path=database_path,
            ip=self._current.ip,
            port=self._current.port,
            devices=self._current.devices,
            left_panel_width=self._current.left_panel_width,
        )
        self.save(updated)

    def update_connection(self, *, ip: str, port: int) -> None:
        updated = PersistedStartup(
            version=_STARTUP_VERSION,
            chip_name=self._current.chip_name,
            qub_name=self._current.qub_name,
            res_name=self._current.res_name,
            result_dir=self._current.result_dir,
            database_path=self._current.database_path,
            ip=ip,
            port=port,
            devices=self._current.devices,
            left_panel_width=self._current.left_panel_width,
        )
        self.save(updated)

    def add_device(self, entry: PersistedDeviceEntry) -> None:
        existing = [d for d in self._current.devices if d.name != entry.name]
        updated = PersistedStartup(
            version=_STARTUP_VERSION,
            chip_name=self._current.chip_name,
            qub_name=self._current.qub_name,
            res_name=self._current.res_name,
            result_dir=self._current.result_dir,
            database_path=self._current.database_path,
            ip=self._current.ip,
            port=self._current.port,
            devices=[*existing, entry],
            left_panel_width=self._current.left_panel_width,
        )
        self.save(updated)

    def remove_device(self, name: str) -> None:
        updated = PersistedStartup(
            version=_STARTUP_VERSION,
            chip_name=self._current.chip_name,
            qub_name=self._current.qub_name,
            res_name=self._current.res_name,
            result_dir=self._current.result_dir,
            database_path=self._current.database_path,
            ip=self._current.ip,
            port=self._current.port,
            devices=[d for d in self._current.devices if d.name != name],
            left_panel_width=self._current.left_panel_width,
        )
        self.save(updated)

    def update_left_panel_width(self, width: int) -> None:
        updated = PersistedStartup(
            version=_STARTUP_VERSION,
            chip_name=self._current.chip_name,
            qub_name=self._current.qub_name,
            res_name=self._current.res_name,
            result_dir=self._current.result_dir,
            database_path=self._current.database_path,
            ip=self._current.ip,
            port=self._current.port,
            devices=self._current.devices,
            left_panel_width=width,
        )
        self.save(updated)


__all__ = [
    "PersistedDeviceEntry",
    "PersistedStartup",
    "StartupPersistenceService",
]
