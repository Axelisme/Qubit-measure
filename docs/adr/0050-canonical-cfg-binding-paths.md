---
status: accepted
---

# 0050 — Canonical cfg binding paths

> 現行定位：binding grammar 仍有效；batch 成功前綴是現況，不代表已落實核准的 atomic edit batch。目標見 [Cfg draft](draft/cfg-editing-boundaries.md)，跨 owner 分工見 [[0065]]。

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

These leaf sweep paths remain the GUI widget contract. Agent edits use a nominal
aggregate target at the sweep parent path and provide the complete sweep object.
Both entry points mutate the same binding model through SweepEditor or
CenteredSweepEditor validation and normalization. Remote code does not duplicate
field subtype grammar; an aggregate edit validates its inputs before committing.

`gui.cfg` owns strict custom-reference-key make/parse/query helpers. Binding
normalizes allowed bare labels; tags remain internal persistence representation.
Binding never imports measure session policy. `ValueRef` resolves at the app
service seam and only for scalar targets. Remote code projects targets to flat
mutation listings and path-change replies.

Complete cfg reads use the separate nominal `CfgDraft.observe()` contract.
`CfgNodeObservation` includes locked literals, cached options, active reference
children, validity, and value carriers with raw/resolved/error state. Binding
owns field traversal and returns detached data without resolving sources. A
settable listing or persistence encoding cannot substitute for this observation.
The remote adapter serializes these data nodes and applies prefix read policy;
it does not import field/editor classes or reconstruct their validation rules.
Only a successful full read without a prefix parameter reveals the whole cfg
revision. Unknown objects and nonfinite values fail serialization rather than
becoming display strings. Prefixes and bare versions never certify a full read.

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
