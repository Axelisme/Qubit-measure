"""Shared process runtime for standalone GUI apps.

The runtime owns process-level mechanics shared by GUI entry points:
logging, rendering initialization, QApplication setup, remote-control socket
start/stop, and exit-code handling. App modules provide only their fixed
runtime contract plus app-specific assembly/lifecycle behavior.
"""

from __future__ import annotations

import faulthandler
import logging
import sys
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, TextIO, TypeVar

from zcu_tools.gui.logging_setup import setup_gui_logging
from zcu_tools.gui.remote.rpc_endpoint import ControlOptions

if TYPE_CHECKING:
    from qtpy.QtWidgets import QApplication

    from zcu_tools.gui.lifetime import LifetimeScope


@dataclass(frozen=True)
class GuiRuntimeSpec:
    """Fixed process contract for one GUI app type."""

    app_name: str
    app_slug: str
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


class ExitWatchdog(Protocol):
    """Faulthandler-compatible watchdog retaining its output until process exit."""

    def dump_traceback_later(
        self, timeout: float, *, file: TextIO, exit: bool = False
    ) -> None:
        """Schedule one all-thread dump, keeping file referenced until finalization."""
        ...


class GuiExitDiagnostics:
    """Append a post-teardown watchdog header and delayed stacks to the GUI log.

    log_file is the GUI log path, or None to use stderr. timeout is positive
    seconds. watchdog defaults to the process-global faulthandler and retains
    the separately opened output until process finalization. Do not close that
    output or install another delayed dump while exit diagnostics are active.
    Diagnostic I/O failures are logged without changing the GUI exit result.
    """

    def __init__(
        self,
        log_file: Path | None,
        *,
        timeout: float = 5,
        watchdog: ExitWatchdog = faulthandler,
    ) -> None:
        if timeout <= 0:
            raise ValueError("exit diagnostic timeout must be positive")
        self._log_file = log_file
        self._timeout = timeout
        self._watchdog = watchdog
        self._armed = False

    def arm(self) -> None:
        """Arm once, leaving the watchdog active through interpreter teardown.

        threading._shutdown runs before atexit, then module/GC finalization may
        block in object destructors. Do not cancel at atexit. CPython's
        faulthandler retains the file object until its own finalization, so the
        independent append fd survives logging.shutdown closing its handlers.
        """
        if self._armed:
            return
        output = sys.stderr
        try:
            if self._log_file is not None:
                output = self._log_file.open("a", encoding="utf-8")
            output.write(
                "\n=== GUI exit watchdog armed; Python stacks follow if still "
                f"running after {self._timeout:g} seconds ===\n"
            )
            output.flush()
            self._watchdog.dump_traceback_later(self._timeout, file=output, exit=False)
        except (OSError, ValueError, RuntimeError):
            if output is not sys.stderr:
                output.close()
            logging.getLogger(__name__).warning(
                "Could not arm GUI exit-stack watchdog", exc_info=True
            )
            return
        self._armed = True


class GuiWindow(Protocol):
    def show(self) -> None: ...


class ControlAdapter(Protocol):
    def start(self) -> int: ...

    def stop(self) -> None: ...


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

    def close(self, assembly: GuiAssembly | None) -> None:
        """Release app-owned resources after drain, including failed startup."""
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
    extra_logging_namespaces: tuple[str, ...] = (),
    **kwargs: Any,
) -> int:
    """Configure process policy, instantiate behavior, and return its exit code.

    behavior_cls declares the app's fixed runtime spec. options supplies CLI
    process settings; args and kwargs go to the behavior constructor. The
    caller-provided extra_logging_namespaces are appended to the app's fixed
    logging namespaces before construction, without changing the shared spec.
    Logging, construction and runtime failures propagate to the caller.
    """
    spec = behavior_cls.spec
    log_file = setup_gui_logging(
        app_name=spec.app_name,
        log_root=options.log_root,
        to_file=options.to_file,
        log_file=options.log_file,
        extra_namespaces=spec.logging_extra_namespaces + extra_logging_namespaces,
        group=spec.logging_group,
    )
    control = build_control_options(spec, options)
    behavior = behavior_cls(*args, **kwargs)
    return run_gui_runtime(
        behavior, control, exit_diagnostics=GuiExitDiagnostics(log_file)
    )


def run_gui_runtime(
    behavior: GuiRuntimeBehavior,
    control: ControlOptions | None,
    *,
    exit_diagnostics: GuiExitDiagnostics | None = None,
) -> int:
    """Run behavior on Qt, then log live threads after app-owned resources close.

    exit_diagnostics optionally arms a process-exit watchdog after close;
    None records only the thread summary. The launcher supplies diagnostics
    using the resolved GUI log path. Quit callbacks are disconnected after
    behavior.close, including failure paths. Plotting stays in shutdown mode
    after quit to protect canvas destructors during process finalization.
    """
    from zcu_tools.gui.lifetime import LifetimeScope

    app = _get_or_create_qapplication()
    scope = LifetimeScope()
    assembly = None
    try:
        _initialize_rendering(app, scope)
        assembly = behavior.assemble(control)
        _validate_control_assembly(control, assembly)
        behavior.before_show(assembly)
        assembly.window.show()

        if assembly.control_adapter is not None:
            if not _start_control_adapter(
                behavior.spec, control, assembly.control_adapter
            ):
                return 1
            # stop() unsubscribes app-side listeners synchronously, so keep it on the
            # Qt main thread; aboutToQuit is emitted there.
            scope.connect(app.aboutToQuit, assembly.control_adapter.stop)

        behavior.after_show(assembly)
        return int(app.exec())
    finally:
        # Runs after admission/drain on success and on every startup failure.
        # A close failure retains the original exception as its context.
        try:
            behavior.close(assembly)
        finally:
            scope.close()
        threads = "; ".join(
            f"name={thread.name!r} daemon={thread.daemon} ident={thread.ident}"
            for thread in threading.enumerate()
        )
        logging.getLogger(__name__).info(
            "GUI teardown complete; live Python threads: %s", threads
        )
        if exit_diagnostics is not None:
            exit_diagnostics.arm()


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


def _initialize_rendering(app: QApplication, scope: LifetimeScope) -> None:
    from zcu_tools.gui.plotting import (
        ensure_host,
        install_mathtext_lock,
        prewarm_mathtext,
        set_shutting_down,
    )

    ensure_host()
    scope.connect(app.aboutToQuit, lambda: set_shutting_down(value=True))
    install_mathtext_lock()
    prewarm_mathtext()


def _get_or_create_qapplication() -> QApplication:
    from qtpy.QtWidgets import QApplication

    from zcu_tools.gui.widgets.selectable_text import install_selectable_message_boxes

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    if not isinstance(app, QApplication):
        raise RuntimeError("existing Qt application is not a QApplication")
    # Process-wide Qt behavior shared by every launcher that runs on this runtime.
    install_selectable_message_boxes(app)
    return app


__all__ = [
    "ControlAdapter",
    "ExitWatchdog",
    "GuiExitDiagnostics",
    "GuiAssembly",
    "GuiLaunchOptions",
    "GuiRuntimeBehavior",
    "GuiRuntimeSpec",
    "GuiWindow",
    "build_control_options",
    "launch_gui_runtime",
    "run_gui_runtime",
]
