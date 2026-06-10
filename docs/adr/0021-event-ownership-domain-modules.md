# ADR-0021 — Event Ownership: Domain Modules Own Enum + Payloads

**Status:** Accepted (Phase 152, 2026-06-10)
**Related:** [[0004]] (three-questions dependency rule), [[0005]] (service roles DDD/Hexagonal)

---

## Context

Before Phase 152 every measure-gui experiment event (tab lifecycle + run lifecycle)
is defined in a single flat `app/main/event_bus.py` file under one `GuiEvent` enum.
The autofluxdep app has a similarly flat `event_bus.py` under a generic `EventType`
enum. Two problems arise:

1. **Mixed domain ownership**: tab events and run events belong to different domains
   but share one enum and one file.  Adding a new domain event forces every reader to
   touch (and mentally parse) unrelated definitions.

2. **Opaque name `EventType`**: the autofluxdep enum name gives no domain signal;
   reading code is harder than it has to be.

The `gui/session/events.py` module already demonstrates the target pattern — one
`SessionEvent` enum owned by the session domain — but the per-app experiment events
have not followed it.

---

## Decision

**A domain module owns its event enum + payload definitions.**

Concrete rules:

- Each domain gets its own module (e.g. `events/tab.py`, `events/run.py`) with a
  domain-specific enum (e.g. `TabEvent`, `RunEvent`).
- The enum name carries the domain signal (`TabEvent`, `RunEvent`,
  `WorkflowEvent`, `SessionEvent`, …) instead of the generic `GuiEvent`/`EventType`.
- The module-local base payload class is typed to its own enum
  (`EVENT: ClassVar[TabEvent]`), so a payload can never be paired with the wrong
  domain enum.
- **Wire names are enum values** (`"tab_added"`, `"run_started"`, …) and are
  byte-identical across refactors (enum value strings are the only public contract).
- **Ports are centralized per layer in `ports.py`** ([[0004]]/[[0005]] norm).  Event
  definitions in domain modules are not ports; they are value objects that both sides
  of a bus subscription import.

---

## Consequences

### App composition layer

Each app **wires** domains at two places:

1. **`EventBus` subscriptions** — the bus stays payload-type-keyed
   (`BaseEventBus`); each app subscribes the domain payload types it cares about.
2. **`EVENT_SERIALIZERS` dict** — the remote adapter keys serializers by payload
   type; `wire_event_name(pt) = pt.EVENT.value` remains the one source of truth for
   wire names.

Neither the bus mechanism nor the `EVENT_SERIALIZERS` registry changes; this ADR
only moves *where* enum + payload definitions live.

### No more flat event_bus.py at app level

`app/main/event_bus.py` and `app/autofluxdep/event_bus.py` are deleted. The
`EventBus` alias (`= BaseEventBus`) is no longer needed; callers import
`BaseEventBus` from `zcu_tools.gui.event_bus` directly.

### fluxdep / dispersive

Already single-domain; their event files are untouched.

---

## Non-decisions

- The bus *mechanism* (payload-type key, emit/subscribe on main thread) does not
  change — see [[0014]] for the shared transport.
- `State` and `EventBus` remain concrete (not ported): `State` is the ADR-0004
  Query SSOT; `BaseEventBus` is the shared abstract transport itself.
- No vertical slice reorganisation: domains are sub-modules of `events/`, not
  top-level packages ([[0005]] M5 decision).
