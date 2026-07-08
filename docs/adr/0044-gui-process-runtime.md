# 0044 — GUI process runtime spec and behavior

**Status:** accepted.

## Context

Standalone GUI apps share process mechanics: logging setup, matplotlib backend
selection, `QApplication` creation, plot host lifecycle, remote-control socket
startup, adapter shutdown, and process exit codes. These concerns accumulated in
launcher scripts, `gui/run_app.py`, and app-local `run_app` functions. The
existing shared layers already cover lower-level mechanisms: `gui/remote` owns
transport, `gui/session` owns measurement-session primitives, and `gui/plotting`
owns figure routing. Process startup needs its own module without absorbing app
domain policy.

## Decision

`gui.runtime` is the single GUI process-runtime module.

Each app declares a fixed runtime contract as `GuiRuntimeBehavior.spec`, a class
variable containing `GuiRuntimeSpec`: app name, discovery slug, plot policy,
default control port, logging group, and extra logging namespaces. Launch-time
user options live in `GuiLaunchOptions` and behavior constructor arguments, not
in the static spec.

App-specific behavior is implemented by `GuiRuntimeBehavior`, an ABC. The
required method is `assemble(control)`, which builds the controller, window, and
the app-local remote-control adapter when `control` is not `None`. Optional
`before_show` and `after_show` hooks express app-local lifecycle work while the
runtime keeps ordering.

The runtime owns:

- logging setup before app behavior construction;
- pre-Qt plot policy (`EMBEDDED_BACKEND`, `AGG_ONLY`, or `NONE`);
- `QApplication` get-or-create;
- post-Qt plot setup (`ensure_host`, mathtext lock/prewarm, shutdown marker);
- `ControlOptions` construction from spec + launch options;
- remote adapter `start()` / `stop()` and bind-failure reporting;
- integer exit codes.

Launcher scripts parse CLI flags and call `sys.exit(main(argv))`. Runtime and app
composition functions return `int` and do not call `sys.exit`.

## Boundaries

Runtime does not own remote method sets, handler policy, operation guards,
session services, persistence schema, or app domain state. Remote/domain/session
policy remains in each app and in the existing shared layers.

## Consequences

`run_qt_app` is retired. New GUI apps implement `GuiRuntimeBehavior` instead of
passing callback factories to a helper. The first migration covers `fluxdep` and
`dispersive`; `main` and `autofluxdep` keep their app-local launch paths until a
later phase migrates their persistence and startup-dialog lifecycle.
