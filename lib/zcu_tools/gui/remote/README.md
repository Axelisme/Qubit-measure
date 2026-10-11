# `zcu_tools.gui.remote` — shared remote adapter

**Last updated:** 2026-10-11, single-holder dispatch

This package owns NDJSON framing, typed envelopes, authentication, socket queues,
owner-scheduler dispatch and lazy event delivery. Apps supply methods, serializers
and application authority; transport does not own experiment policy.

Measure and Fluxdep inject the same `ControlIdentity` used by their Qt frontend.
Business methods require holder by default; explicit readonly declarations allow
observers. Control management and catalog remain callable with their own admission
rules. Dispatch checks current holder on the owner loop before domain effects.
Fixed socket identity, trusted labels, lock/reclaim and structured refusals are
specified in [ADR-0070](../../../../docs/adr/0070-control-identity.md).

Reads return current projections, not mutation baselines. MCP does not keep a
second authority or version table. Authentication, exposure, domain validation,
busy checks and source-bound operations remain independent. Readonly apps do not
inject writable control authority.

Response failure or timeout can follow a committed command. Transport neither
rolls back domain effects nor retries requests. Event subscriptions keep their
connection-local lifecycle and lazy serialization/backpressure behavior.

Wire details live in the [measure remote README](../app/measure/remote/README.md).
App composition and error boundaries are in
[ADR-0068](../../../../docs/adr/0068-remote-transport.md).
