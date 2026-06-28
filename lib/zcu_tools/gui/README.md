# `zcu_tools.gui` — GUI framework cheat-sheet

**Last updated:** 2026-06-29 (Qt typing invariants)

High-level map of the shared GUI layer. App-specific detail lives in each app's
own README under `app/<name>/`; cross-cutting subpackages (`event_bus`,
`plotting`, `remote`, `session`, `widgets`) are shared by every app.

## Dialogs — always non-blocking

Every app embeds a control socket whose RPC handler runs on the Qt event loop,
so **all dialogs (and message boxes) launch with `open()`, never `exec()`**.
A blocking `exec()` would freeze the event loop until the user dismisses the
dialog, stalling the control socket (and, for measure, deadlocking cross-thread
marshalling). Read a dialog's outcome from its `accepted` / `finished` signal
instead of `exec()`'s return value, set `WA_DeleteOnClose`, and hold an instance
reference so `open()`'s immediate return does not let it be garbage-collected.
The measure registry path (`MainWindow.open_dialog` / `close_dialog`) is detailed
in `app/main/services/remote/README.md`. The sole intentional `exec()` is the
global unhandled-exception hook in `app/main/utils/error_handler.py`, where the
process is already crashing and the message must block.

Short-lived modal confirmations, error reports, and text prompts are the
deliberate exception. A synchronous `QMessageBox.question` / `.warning` /
`.critical` or `QInputDialog.getText` may block: it is raised by a direct,
synchronous user action while no operation is in flight, carries no
long-running flow, and never marshals a worker result back to the main
thread. Its nested modal loop still services the control socket's queued
notifier, so it neither stalls startup nor deadlocks cross-thread
marshalling. The `open()` rule still governs every dialog that hosts a
long-running flow or can surface while an operation is running.

## Logging (`logging_setup.py`)

`logging_setup.setup_gui_logging` is the single place that decides *how* every
GUI entry point configures logging. All four `script/run_*_gui.py` launchers and
the measure MCP server (`mcp/measure/server.py:main`) call it instead of each
rolling their own handler set.

Key invariants:

- **The file handler is attached at the whole `zcu_tools.gui` namespace** (plus
  any `extra_namespaces` an entry point needs — measure adds
  `zcu_tools.experiment.v2_gui`; the MCP server adds `zcu_tools.mcp`). Attaching
  at the package root, not an app sub-namespace, is deliberate: cross-cutting
  subpackages (`event_bus`, `plotting`, `remote`, `session`) are siblings of the
  app namespace, and a handler scoped to one app would silently miss them. That
  missed-sibling gap is the bug this scheme exists to prevent; the regression
  test is `tests/gui/test_logging_setup.py`.
- **Per-session timestamped files** under `<repo>/logs/<group>/<app>/` (`group`
  = `gui` for launchers, `mcp` for the server). Each launch writes its own file,
  so a previous session's evidence is never overwritten. On startup the helper
  purges all but the newest `retain` (default 10) files in that directory.
- **Levels:** file handler at DEBUG, stderr handler at WARNING. High-frequency
  bookkeeping (operation create, background submit) logs at DEBUG; lifecycle
  events (operation settle, connect/device result, persistence flush/restore,
  writeback apply) at INFO; failures at WARNING/ERROR (worker exceptions carry
  `exc_info` so the real traceback survives the cross-thread marshal).
- A `--log-file` CLI override (when a launcher exposes one) wins over the
  per-session scheme: the explicit path is used verbatim and no purge runs.

The MCP server uses stdout for its JSON-RPC transport, so its logging never
touches stdout — the helper only adds a stderr handler and the DEBUG file
handler.

## Qt Typing Invariants

Qt override signatures must match the binding stubs exactly, including nullable
event parameters and stub parameter names. `closeEvent` implementations use
`a0: QCloseEvent | None`; `eventFilter` implementations use
`a0: QObject | None, a1: QEvent | None` and immediately delegate nullable inputs
to `super()`.

Qt table header accessors are treated as optional by the stubs. Code that
configures `horizontalHeader()` must assert the header is not `None` before
calling resize APIs; the assert records the widget invariant and keeps the
runtime failure explicit if a binding ever violates it.
