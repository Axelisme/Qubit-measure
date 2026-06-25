---
status: accepted
---

# Arbitrary waveform MCP failures use tool errors

Agent-facing arbitrary waveform MCP tools follow the existing measure-gui remote/MCP error contract: success returns a normal result payload, while validation failures, name collisions, missing assets, and readiness failures surface as failed tool calls backed by the RPC `{ok:false,error:{code,message,reason?,data?}}` envelope. `set_arb_waveform` therefore returns `{"success":true,"status":"created"|"overwritten","preview_figure":...}` only on success; failures carry stable `reason` tags and optional structured `data` instead of normal `{success:false,...}` payloads. Arbitrary waveform mutations bump an `arb_waveforms` resource version, and agent-facing mutation uses the same stale-guard pattern as other guarded measure-gui edits so an agent does not silently overwrite a GUI-side asset change it has not observed. This keeps arb waveform tools consistent with the rest of measure-gui automation while still giving agents machine-readable failure causes.
