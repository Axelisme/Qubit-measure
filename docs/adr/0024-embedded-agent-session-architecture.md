# ADR-0024 — Agent launch UI 退役

> **狀態：已退役。** measure-gui 不再提供 toolbar「Agent」按鈕、
> `AgentLaunchDialog` 或 `services/agent_launcher.py`。本 ADR 保留為歷史決策
> 紀錄；目前仍生效的 user↔agent runtime feedback 設計見 [[0025]]。

## 脈絡

measure-gui 曾經提供一個「Agent」按鈕，讓使用者從 GUI 內啟動外部終端中的
互動式 agent，並把 loopback MCP 設定、允許工具、session id 與 bootstrap prompt
一併組好。這個設計取代了更早的 embedded stream-json agent session 嘗試：
`AgentRunner` / `AgentChatService` / `AgentSupervisor` /
`IndependentAgentSession` / `AgentSessionRegistry` / `AgentSessionPort` 等元件。

後續使用發現這個手動 launch 入口不是必要 workflow：agent session 可由外部
CLI/MCP 入口自行啟動，再透過既有 measure-gui control socket 連線。GUI 內維護
terminal spawn、resumable session 清單與 bootstrap prompt 只增加維護面，沒有對
實際量測 workflow 帶來足夠價值。

## 決策

measure-gui 移除手動 Agent launch surface：

- 移除 toolbar「Agent」按鈕。
- 移除 `ui/agent_launch_dialog.py` 與其專用 UI tests。
- 移除 `services/agent_launcher.py` 與其專用 service tests。
- 移除 `Controller.build_agent_bootstrap_prompt()`，因為不再由 GUI spawn agent。

Agent 仍可透過外部 MCP/CLI 流程連到 GUI control socket；lazy auto-connect 與
measure-gui MCP server 的 tool surface 不屬於本 ADR 的退役範圍。

## 保留

run / analysis / device operation 期間的 user↔agent runtime feedback 保留，並以
[[0025]] 為目前設計來源：

- GUI 端 `FeedbackPanel` 仍在有 live operation 且 MCP control client 連線時 mount。
- `Send` 仍送出 `Message`，operation 繼續跑，agent 收到 `user_feedback`。
- `Send & Stop` 仍送出單一 `Stop(reason)` 意圖，依 operation cancel hook gating。
- agent→user prompt 仍由 `gui_prompt_user` / `NotifyUserDialog` 處理。

## 後果

- GUI 不再負責 spawn terminal、追蹤 resumable sessions 或注入 bootstrap prompt。
- Agent 的啟動責任回到外部 CLI/MCP workflow，GUI 只保留已連線 client 的 runtime
  interaction surface。
- [[0025]] 成為 user↔agent 互動的唯一有效 ADR；本 ADR 僅說明已移除的 launch UI。
