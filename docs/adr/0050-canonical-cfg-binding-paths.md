---
status: accepted
---

# 0050 — Canonical cfg binding paths

## Context

Cfg mutation and listing previously used a second field-subtype grammar in the
measure remote adapter. The authorities drifted on scalar types, reference
keys, sweep aliases, suggestions, and batch path diffs.

## Decision

`gui.cfg.binding` owns nominal `SettableTarget` objects and one traversal for
listing, resolving, setting, validation, suggestions, and migration hints.
Every listed path resolves to the same kind, value type, current value, choices,
and `affects_path_shape` metadata; no unlisted alias executes.

Canonical scalar paths are dotted leaves. Sweep paths end directly in
`start|stop|expts|step`, centered sweeps in `center|span|expts|step`, reference
keys in `.ref`, and reference children descend directly. Removed `.sweep.*` and
`.value.*` spellings fail before mutation and report a verified replacement.
Unrepresentable or reserved field keys fail when the target tree is built.

`gui.cfg` owns strict custom-reference-key make/parse/query helpers. Binding
normalizes allowed bare labels; tags remain internal persistence representation.
Binding never imports measure session policy. `ValueRef` resolves at the app
service seam and only for scalar targets. Remote code only projects targets to
existing flat/tree wire views and applies prefix read policy.

Cfg edit batches are ordered, fail-fast, and non-atomic. Resolution remains
sequential so a reference switch can expose the next path. Only a shape target
materializes path sets, once before the first shape edit and once after success.
The sorted response is the final net diff, so A→B→A is empty. Failure preserves
successful prefix edits, re-raises the original typed failure, and performs no
after diff. Per-edit version bumps and [[0049]] lazy subscriber payloads remain;
this decision does not implement snapshot coalescing.

## Consequences

- Binding list and resolve acceptance sets cannot drift.
- Remote consumers depend on a deep nominal contract, not field classes.
- Agents copy listing paths verbatim and receive actionable legacy hints.
- Batch listing cost is independent of edit count unless shape changes.

This extends [[0008]], [[0013]], [[0014]], [[0045]], and [[0046]].
