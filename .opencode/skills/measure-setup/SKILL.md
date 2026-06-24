---
name: measure-setup
description: Collect the stable, per-rig measurement setup (board connection, project identity, channel wiring, device addresses) into a single user-filled measure_setup.yaml at the repo root. Use before driving a measure-gui measurement when setup info is needed and not already in the file.
---

Read `.agents/skills/measure-setup/SKILL.md` and follow it as the canonical skill content.

This shim exists because opencode auto-loads project skills from `.opencode/skills/`, while the repo keeps cross-agent canonical skill content under `.agents/skills/`.
