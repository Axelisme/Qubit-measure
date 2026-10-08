# gui/app/workflows — Workflow form submission

**Last updated:** 2026-10-08，tunables submission seed

TunablesHost owns the accepted expression input, GUI draft base revision and submission policy for one native run. GUI and JSON callers use the same FormState/Engine seam. Engine still owns numeric validation, revision and journal publication. A stale Apply retains draft and fails without retry. External edits preserve an edited or locally unparseable GUI draft until explicit Discard.

Concrete workflow attachments remain in zcu_lab and are supplied by the composition root. The T1 tuning attachment embeds the existing T1TunablesPanel. This seed does not implement app startup, Start capture, hardware limits, worker leases, RPC transport, PNG or source reload.
