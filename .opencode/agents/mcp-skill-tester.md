---
name: mcp-skill-tester
description: Use this agent when you need to exercise MCP tools and their associated skills end-to-end during development, validating that they behave as documented and gathering structured usability feedback.
mode: subagent
color: warning
options:
  source: .agents/agents/mcp-skill-tester.md
---

Before doing any MCP/skill dogfooding work, read `.agents/agents/mcp-skill-tester.md` and follow it as the canonical prompt for this agent.

This shim exists because opencode auto-loads project agents from `.opencode/agents/`, while the canonical cross-agent content for this repo lives under `.agents/`.
