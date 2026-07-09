"""Shared process runtime for standalone GUI apps.

The runtime owns process-level mechanics shared by GUI entry points:
logging, matplotlib backend policy, QApplication setup, remote-control socket
start/stop, and exit-code handling. App modules provide only their fixed
runtime contract plus app-specific assembly/lifecycle behavior.
"""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Protocol, TypeVar, cast

from zcu_tools.gui.logging_setup import setup_gui_logging
from zcu_tools.gui.remote.rpc_endpoint import ControlOptions


class PlotPolicy(Enum):
    """Matplotlib process policy for a GUI app."""

    EMBEDDED_BACKEND = "embedded_backend"
    AGG_ONLY = "agg_only"
    NONE = "none"


@dataclass(frozen=True)
class GuiRuntimeSpec:
    """Fixed process contract for one GUI app type."""

    app_name: str
    app_slug: str
    plot_policy: PlotPolicy
    default_control_port: int
    logging_group: str = "gui"
    logging_extra_namespaces: tuple[str, ...] = ()


@dataclass(frozen=True)
class GuiLaunchOptions:
    """Launch-time process options supplied by the CLI edge."""

    log_root: Path
    to_file: bool = True
    log_file: Path | None = None
    control_port: int | None = None
    control_token: str | None = None
    control_allow_external: bool = False
    no_control: bool = False


class GuiWindow(Protocol):
    def show(self) -> None: ...


class ControlAdapter(Protocol):
    def start(self) -> int: ...

    def stop(self) -> None: ...


class SignalLike(Protocol):
    def connect(self, callback: object) -> None: ...


class GuiApplication(Protocol):
    aboutToQuit: SignalLike

    def exec(self) -> int: ...


@dataclass
class GuiAssembly:
    """Objects the runtime needs after app-specific assembly."""

    controller: object
    window: GuiWindow
    control_adapter: ControlAdapter | None = None


class GuiRuntimeBehavior(ABC):
    """App-specific behavior behind the shared runtime interface."""

    spec: ClassVar[GuiRuntimeSpec]

    @abstractmethod
    def assemble(self, control: ControlOptions | None) -> GuiAssembly:
        """Build controller/window and optionally the app-local control adapter."""
        raise NotImplementedError

    def before_show(self, assembly: GuiAssembly) -> None:
        """Run app-specific setup after assembly and before the window is shown."""
        del assembly

    def after_show(self, assembly: GuiAssembly) -> None:
        """Run app-specific setup after the window is shown and control is started."""
        del assembly


BehaviorT = TypeVar("BehaviorT", bound=GuiRuntimeBehavior)


def build_control_options(
    spec: GuiRuntimeSpec, options: GuiLaunchOptions
) -> ControlOptions | None:
    """Build shared remote-control options from app contract + launch options."""
    if options.no_control:
        return None
    explicit_port = options.control_port is not None
    port = options.control_port if explicit_port else spec.default_control_port
    if port is None:
        raise ValueError("control port is required when remote control is enabled")
    return ControlOptions(
        port=port,
        token=options.control_token,
        allow_external=options.control_allow_external,
        allow_port_fallback=not explicit_port,
        app_slug=spec.app_slug,
    )


def launch_gui_runtime(
    behavior_cls: type[BehaviorT],
    options: GuiLaunchOptions,
    *args: Any,
    **kwargs: Any,
) -> int:
    """Configure process-level policy, instantiate behavior, and run the GUI."""
    spec = behavior_cls.spec
    setup_gui_logging(
        app_name=spec.app_name,
        log_root=options.log_root,
        to_file=options.to_file,
        log_file=options.log_file,
        extra_namespaces=spec.logging_extra_namespaces,
        group=spec.logging_group,
    )
    _configure_pre_qt_plot_policy(spec.plot_policy)
    control = build_control_options(spec, options)
    behavior = behavior_cls(*args, **kwargs)
    return run_gui_runtime(behavior, control)


def run_gui_runtime(
    behavior: GuiRuntimeBehavior, control: ControlOptions | None
) -> int:
    """Run an already-instantiated GUI behavior on the Qt event loop."""
    app = _get_or_create_qapplication()
    _configure_post_qt_plot_policy(behavior.spec.plot_policy, app)

    assembly = behavior.assemble(control)
    _validate_control_assembly(control, assembly)
    behavior.before_show(assembly)
    assembly.window.show()

    if assembly.control_adapter is not None:
        if not _start_control_adapter(behavior.spec, control, assembly.control_adapter):
            return 1
        # stop() unsubscribes app-side listeners synchronously, so keep it on the
        # Qt main thread; aboutToQuit is emitted there.
        app.aboutToQuit.connect(assembly.control_adapter.stop)

    behavior.after_show(assembly)
    return int(app.exec())


def _validate_control_assembly(
    control: ControlOptions | None, assembly: GuiAssembly
) -> None:
    if control is None and assembly.control_adapter is not None:
        raise RuntimeError("control adapter was built while control is disabled")
    if control is not None and assembly.control_adapter is None:
        raise RuntimeError("control options were provided but no adapter was built")


def _start_control_adapter(
    spec: GuiRuntimeSpec, control: ControlOptions | None, adapter: ControlAdapter
) -> bool:
    try:
        adapter.start()
    except RuntimeError as exc:
        port = getattr(control, "port", "?")
        print(
            f"\nERROR: cannot open control socket for {spec.app_name!r} on port {port}.\n"
            f"  {exc}\n\n"
            f"  That port is pinned and already in use.\n"
            f"  Pass a different --control-port <N>, omit it to auto-pick a\n"
            f"  free port, or --no-control to disable the remote-control socket.\n",
            file=sys.stderr,
        )
        return False
    return True


def _configure_pre_qt_plot_policy(policy: PlotPolicy) -> None:
    if policy is PlotPolicy.EMBEDDED_BACKEND:
        from zcu_tools.gui.plotting.setup import configure_matplotlib_backend

        configure_matplotlib_backend()
    elif policy is PlotPolicy.AGG_ONLY:
        import matplotlib

        matplotlib.use("Agg")
    elif policy is PlotPolicy.NONE:
        return
    else:  # pragma: no cover - Enum exhaustiveness guard.
        raise ValueError(f"unknown plot policy: {policy!r}")


def _configure_post_qt_plot_policy(policy: PlotPolicy, app: GuiApplication) -> None:
    if policy is PlotPolicy.NONE:
        return

    from zcu_tools.gui.plotting import install_mathtext_lock, prewarm_mathtext

    if policy is PlotPolicy.EMBEDDED_BACKEND:
        from zcu_tools.gui.plotting import ensure_host, set_shutting_down

        ensure_host()
        app.aboutToQuit.connect(lambda: set_shutting_down(True))
    elif policy is not PlotPolicy.AGG_ONLY:
        raise ValueError(f"unknown plot policy: {policy!r}")

    install_mathtext_lock()
    prewarm_mathtext()


def _get_or_create_qapplication() -> GuiApplication:
    from qtpy.QtWidgets import QApplication  # type: ignore[attr-defined]

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    if not isinstance(app, QApplication):
        raise RuntimeError("existing Qt application is not a QApplication")
    return cast(GuiApplication, app)


__all__ = [
    "ControlAdapter",
    "GuiAssembly",
    "GuiLaunchOptions",
    "GuiRuntimeBehavior",
    "GuiRuntimeSpec",
    "GuiWindow",
    "PlotPolicy",
    "GuiApplication",
    "SignalLike",
    "build_control_options",
    "launch_gui_runtime",
    "run_gui_runtime",
]
