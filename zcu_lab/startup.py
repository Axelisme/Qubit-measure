"""Default lab session policy, independent of the framework's layout cache.

Only this script owns measure-session.json under ctx.cache_root. Aliases remain
entry-owned in aliases.yaml. Authored tab inputs are copied as opaque Form JSON.
No hardware settings, raw measurement data or component bindings are saved here.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from functools import partial
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from zcu_tools.gui.app.measure.services.tab_input import TabInput
from zcu_tools.gui.app.measure.startup_context import StartupContext

logger = logging.getLogger(__name__)


class _Soc(BaseModel):
    address: str
    port: int


class _Device(BaseModel):
    name: str
    type_name: str
    address: str


class _Tab(BaseModel):
    adapter: str
    input: str


class _Session(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal[1] = 1
    soc: _Soc | None = None
    devices: tuple[_Device, ...] = ()
    entry: str | None = None
    point: str | None = None
    tabs: tuple[_Tab, ...] = ()
    placeholders: dict[str, str] = Field(default_factory=dict)


def _save(ctx: StartupContext, path: Path) -> None:
    # on_close runs on-owner after drain, before managed drivers/SoC are removed.
    soc = ctx.soc.current()
    selection = ctx.setup.current()
    snapshot = _Session(
        soc=_Soc(address=soc.ip, port=soc.port) if soc is not None else None,
        devices=tuple(
            _Device(name=row.name, type_name=row.type_name, address=row.address)
            for row in ctx.devices.current()
        ),
        placeholders=ctx.placeholders.current(),
        entry=selection.entry.name if selection.entry is not None else None,
        point=selection.point_label,
        tabs=tuple(
            _Tab(adapter=tab.adapter_name, input=tab.input.form_json)
            for tab in ctx.tabs.current()
        ),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, delete=False
        ) as file:
            temporary = Path(file.name)
            file.write(snapshot.model_dump_json(indent=2))
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class _Restore:
    """Count and isolate the script's repeated resource restore attempts."""

    def __init__(self, ctx: StartupContext) -> None:
        self.ctx = ctx
        self.restored = 0
        self.failed = 0
        self.skipped = 0

    def attempt(self, label: str, action: Callable[[], object]) -> bool:
        self.ctx.progress(f"Restoring {label}")
        try:
            action()
        except Exception:
            # This script, not the framework, owns per-item failure isolation.
            self.failed += 1
            logger.exception("Startup could not restore %s", label)
            return False
        self.restored += 1
        return True

    def finish(self) -> None:
        self.ctx.summary(
            f"Startup: {self.restored} restored, {self.failed} failed, {self.skipped} skipped"
        )


def _restore_devices(ctx: StartupContext, saved: _Session, restore: _Restore) -> None:
    for device in saved.devices:
        connected = restore.attempt(
            f"device {device.name}",
            partial(ctx.devices.connect, device.name, device.type_name, device.address),
        )
        if not connected:
            try:
                ctx.devices.remember(device.name, device.type_name, device.address)
            except Exception:
                restore.failed += 1
                logger.exception("Startup could not remember device %s", device.name)


def _restore_tabs(ctx: StartupContext, saved: _Session, restore: _Restore) -> None:
    entry = saved.entry
    point = saved.point
    entry_ready = entry is not None and restore.attempt(
        f"entry {entry}", partial(ctx.setup.open_entry, entry)
    )
    point_ready = (
        entry_ready
        and point is not None
        and restore.attempt(f"point {point}", partial(ctx.setup.use_point, point))
    )
    if not point_ready:
        restore.skipped += len(saved.tabs)
        if saved.tabs:
            logger.warning(
                "Startup skipped %s tabs: entry/point not ready", len(saved.tabs)
            )
        return
    for tab in saved.tabs:
        restore.attempt(
            f"tab {tab.adapter}",
            partial(ctx.tabs.new_tab, tab.adapter, input=TabInput(tab.input)),
        )


def startup(ctx: StartupContext) -> None:
    """Restore the script's own cache in order, then save live state on normal close.

    Called by the trusted script runner on a worker. SoC and each device failure
    is logged without dialogs; failed devices remain remembered for Reconnect.
    Tabs open only after this run opens the saved entry and selects its point.
    No later source selection retries skipped tabs. On-close reads the then-live
    state, not this startup input. Force kill cannot guarantee a save.
    """
    path = ctx.cache_root / "measure-session.json"
    ctx.on_close(lambda: _save(ctx, path))
    restore = _Restore(ctx)
    try:
        saved = _Session.model_validate_json(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        restore.finish()
        return
    except (OSError, ValueError):
        restore.failed += 1
        logger.exception("Startup could not read session cache %s", path)
        restore.finish()
        return
    for name, binding in saved.placeholders.items():
        restore.attempt(
            f"placeholder {name}", partial(ctx.placeholders.register, name, binding)
        )
    soc = saved.soc
    if soc is not None:
        restore.attempt("SoC", partial(ctx.soc.connect, soc.address, soc.port))
    _restore_devices(ctx, saved, restore)
    _restore_tabs(ctx, saved, restore)
    restore.finish()
