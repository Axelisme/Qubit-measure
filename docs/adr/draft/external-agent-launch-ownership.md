# 外部 agent launch 責任（draft）

**狀態：** 已核准的責任邊界，待 Remote／Transport ADR 核實並轉正；不代表本文件是現行 ADR。

## 現況與依據

[退役的 ADR-0024](../retired/0024-embedded-agent-session-architecture.md) 記錄 GUI 移除 toolbar Agent launch、terminal spawn、resumable session 清單與 bootstrap prompt。現行 `gui/app/measure/app.py` 的 startup composition 與 `scripts/run_measure_gui.py` 的程序入口沒有 agent launch 接線；`mcp/measure/server.py` 提供 agent 使用的 MCP 入口。GUI 仍可接收 remote control client。GUI 啟動 agent 和外部 agent 連接 GUI 是不同方向。

## 待轉正的邊界

外部 CLI／MCP workflow 擁有 agent 啟動；GUI 不負責 spawn terminal、管理 resumable agent session 或注入 bootstrap prompt。這不要求取消 GUI 的 remote socket，也不改動 lazy auto-connect 或 MCP workflow。這是 launch 責任的分流，不為 process runtime 新增 agent lifecycle policy。

## 轉正條件

T21 對照 Remote／Transport 實作核實連線及 launch 責任，在該 domain ADR 承接上述有效邊界；確認當時的 CLI／MCP 入口和 lazy connect 行為後，移除本 draft。具體工具與連線流程由 remote owner 文件維護。
