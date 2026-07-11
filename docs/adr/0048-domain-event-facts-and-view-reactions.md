---
status: accepted
---

# ADR-0048 — Domain event facts and coordinator-owned View reactions

## Context

Tab interaction/content envelopes historically carried only a tab id. Producers
used the same payload for local edits, operation lifecycle, result commits, and
writeback, so the Qt consumer could only rebuild most of the tab. Analyze start
also clears its canvases while State retains the previous figures; failed or
cancelled operations therefore need an explicit signal that retained content may
be shown again.

## Decision

`TabInteractionChangedPayload` and `TabContentChangedPayload` each carry a
mandatory closed domain fact. Producers describe the lifecycle outcome or
committed resource; they do not describe widgets, refresh flags, or masks.

`MainWindowEventCoordinator` owns the ordered fact-to-reaction matrices. It reads
at most one tab snapshot when a reaction needs State and reads none for local
form/path edits. Failed, cancelled, and start-rejected primary/post analysis
facts restore retained primary then post figures. Successful terminal facts do
not draw; the following content-commit fact draws the new result once. Run
failure never restores invalidated pre-run figures, and loaded-result commit
explicitly clears stale canvases.

Normal run lifecycle uses `RunStartedPayload` / `RunFinishedPayload` without a
duplicate tab-interaction event. ModuleLibrary changes refresh service-owned cfg
drafts directly; the main-window coordinator does not rebuild tab content for
them.

The remote serializer intentionally projects both internal fact payloads to the
existing coarse `{tab_id, requery:["tab.snapshot"]}` envelope. Event names,
serializer shapes, and remote subscriptions stay unchanged, so only the GUI code
revision changes; the wire contract version does not.

## Consequences

- Missing producer classification fails at payload construction/type checking.
- View cost is visible in one coordinator-owned matrix and payloads remain
  domain-semantic value objects.
- Retained State plus an explicit terminal fact is the figure restore contract.
- Operation terminal and content commit remain separate facts, preventing
  success-path double drawing.

This decision applies [[0004]] reaction wiring, preserves [[0013]]/[[0014]] wire
adapter boundaries, and extends [[0021]] domain ownership from event names to
their semantic facts.
