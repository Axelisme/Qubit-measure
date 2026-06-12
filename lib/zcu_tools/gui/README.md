# `zcu_tools.gui` — GUI framework cheat-sheet

**Last updated:** 2026-06-12 (Phase 157: logging system)

High-level map of the shared GUI layer. App-specific detail lives in each app's
own README under `app/<name>/`; cross-cutting subpackages (`event_bus`,
`plotting`, `remote`, `session`, `widgets`) are shared by every app.

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
