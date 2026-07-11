---
status: accepted
---

# ADR-0049 — Subscriber-aware lazy remote push

## Context

Remote EventBus bridges used to serialize and encode every event before checking
whether any connected client subscribed to its wire name.  Cfg-editor changes
also traversed the complete settable path tree on every edit even when nobody
subscribed to that editor.  A separate query-before-broadcast check would retain
a time-of-check/time-of-use race with subscribe, unsubscribe, and disconnect.

## Decision

`NdjsonRpcEndpoint` owns a two-phase lazy broadcast transaction.  Under its
client-registry lock it first selects matching registered, non-closing links in
registry order.  Zero matches return without calling the line factory.  The
factory runs once on the caller thread and outside the endpoint lock.  A final
locked phase revalidates only the original links and uses the existing
per-client enqueue path.

App-owned subscription mutation and listing use a narrow endpoint transaction
over `ClientLink.app_ctx`.  The endpoint never interprets that state.  This
linearizes selection, unsubscribe, and reply enqueue: a completed unsubscribe
or disconnect receives no late push; a push selected for final delivery before
unsubscribe is queued before the unsubscribe reply.  A client subscribing after
initial selection does not receive the old event.

Shared EventBus callbacks put serializer plus NDJSON encoding inside the line
factory.  CfgEditorService supplies an app-neutral payload factory; it still
bumps the editor version synchronously for every edit, while the remote adapter
materializes `current_paths()` only for a matching subscriber.  `editor_closed`
subscription cleanup is a delivery hook and occurs only after that close push is
accepted by a client queue.  Predicate and delivery callbacks are pure,
non-blocking client-state operations and never call back into the endpoint.

Diagnostics remain eager and independent of EventBus subscription.  Existing
wire names, payload shapes, per-client ordering, queue capacity, drop budget,
and sole-writer isolation do not change.  `WIRE_VERSION` and MCP policy therefore
remain unchanged; every GUI app bumps only its code revision.

## Consequences

- No matching subscriber means no serializer, path traversal, or encode cost.
- Multiple matching clients share one immutable encoded line while retaining
  independent queue/backpressure behavior.
- Serializer and payload failures remain contained by the app callback and do
  not partially enqueue a push.
- Endpoint synchronization remains domain-free and never holds its registry
  lock across application serialization or cfg traversal.

This decision preserves the editor ownership of [[0008]], the diagnostic and
second-view boundaries of [[0013]], the pure transport split of [[0014]], and
the producer-domain ownership of [[0021]].  It does not alter the internal event
facts or View reaction matrix established by [[0048]].
