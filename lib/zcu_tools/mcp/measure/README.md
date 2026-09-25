**Last updated:** 2026-09-26 — live catalog and reconnect-safe operations

# `zcu_tools/mcp/measure/`

這是 measure-gui 的 MCP driving adapter。它只透過 GUI 的 loopback remote socket 操作同一份 GUI 狀態；GUI core 不 import MCP。GUI remote method entries 擁有每個 wire method 的 exposure、guard、read-reveal、成功寫入刷新 baseline 與 operation policy。MCP 連線時載入 live `rpc.catalog`，不維護第二份 method/policy 表，也不根據 catalog 動態建立 tools。

## 連線與操作

- `assembly.py` 建立固定手寫工具表。`connect`、`status`、`wait`、`cancel` 與三個 `rpc_*` 是首批入口；其餘 domain tools 依各自 ticket 接入。`rpc_call` 只能呼叫 catalog 標為 `rpc` 的 method。
- `session.py` 擁有單一 MCP session 的 catalog、guard observation、bridge 與 opaque integer operation handles。明確重連或 socket 重建時清 catalog/observations/舊 handle 對應；下一個 GUI incarnation 可重用 wire operation ID，但不重用此 MCP session 曾向 agent 外露的 handle。GUI-origin operation 由 `status` 收錄，與 agent-started operation 使用同一映射；wait/cancel 先驗 handle，才把 GUI ID 送到 wire。這不是第二個 operation outcome store。
- 資源版本由 GUI owner bump。MCP 只在 read 完整揭露 catalog 指定的資源後更新相應 baseline；status 的 orientation reads 不吸收其他資源。成功寫入是否刷新 baseline 也由 GUI catalog 明確宣告。stale 拒絕不刷新 baseline，需重讀資源後才由呼叫者決定是否重試；transport timeout 不自動重送。
- `bridge` 只管 socket/GUI subprocess。`connect(launch=...)` 對已由此 bridge 啟動且仍活著的 GUI 不會在另一個空 port 假裝再次啟動；MCP 清理只斷線，不殺 GUI。所有硬體 gate、取消與 operation 結果都仍歸 GUI owners。

## 驗證

`tests/mcp/measure/` 以 public tools/session、recording transport 驗證 catalog、連線、guard、operation。GUI remote/service 測試驗證真 socket 與 GUI-origin path。離線選集只用 fake/mock，不啟動真儀器；測試路徑與 fixture 見 `tests/README.md`。
