**Last updated:** 2026-06-25 (MeasureMcpSession policy ownership)

# `zcu_tools/mcp/measure/`

measure-gui 的 MCP entry，負責把 MCP tool call 轉接到 live measure-gui
RemoteControlAdapter。此 package 是 app-local policy 層，不是共用 transport。

## 邊界

- `server.py` 保留 agent-facing tool declarations、override tool schema、stdio loop
  hooks，以及 lifecycle tool 的薄委派。
- `session.py` 擁有 measure-only policy state：diagnostics piggyback queue、
  optimistic-concurrency baseline、guarded send flow、operation handle capture，以及
  `gui_debug_operations` 使用的 latest-handle projection。
- `session_policy.py` 放不可變 policy table 與純 helper：version guard deps、
  read-reveal table、start-op semantic key mapping、stale key 語義化。
- `McpBridge` 只屬於 `zcu_tools.mcp.core` 的 transport adapter；measure-gui policy
  不下放到 bridge。

## 測試注意

Remote/MCP 測試會建立 loopback socket；受限 sandbox 可能需要 unsandboxed execution。
headless 測試環境通常需要 `QT_QPA_PLATFORM=offscreen`、
`QT_QPA_PLATFORMTHEME=`、`MPLBACKEND=Agg`。
