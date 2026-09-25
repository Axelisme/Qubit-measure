**Last updated:** 2026-09-26 — write-version receipts and guarded observations

# `zcu_tools/mcp/measure/`

這是 measure-gui 的 MCP driving adapter。它只透過 GUI 的 loopback remote socket 操作同一份 GUI 狀態；GUI core 不 import MCP。GUI remote method entries 擁有每個 wire method 的 exposure、guard、read-reveal、成功寫入刷新 baseline 與 operation policy。MCP 連線時載入 live `rpc.catalog`，不維護第二份 method/policy 表，也不根據 catalog 動態建立 tools。

## 連線與操作

- `assembly.py` 建立固定手寫工具表。`connect`、`status`、`wait`、`cancel` 與三個 `rpc_*` 是首批入口；其餘 domain tools 依各自 ticket 接入。`rpc_call` 只能呼叫 catalog 標為 `rpc` 的 method。
- `session.py` 擁有單一 MCP session 的 catalog、guard observation、bridge 與 opaque integer operation handles。明確重連或非預期 EOF 後清 catalog/observations/舊 handle 對應；下一個 GUI incarnation 可重用 wire operation ID，但不重用此 MCP session 曾向 agent 外露的 handle。GUI-origin operation 由 `status` 收錄，與 agent-started operation 使用同一映射；wait/cancel 先驗 handle，才把 GUI ID 送到 wire。這不是第二個 operation outcome store。
- 資源版本由 GUI owner bump。MCP 在完整 read 前取保守版本，成功後只更新 catalog 指定的資源；`prefix` 局部讀取不揭露整份 cfg，status 的 orientation reads 不吸收其他資源。GUI catalog 宣告哪些寫入回傳 owner-thread 前後版本；MCP 只更新該次變更且寫入前版本符合既有觀察的資源，不吸收別的 GUI 編輯。stale 拒絕不刷新 baseline，需重讀資源後才由呼叫者決定是否重試；斷線或 transport timeout 不自動重送。
- `bridge` 只管 socket/GUI subprocess。`connect(launch=...)` 對已由此 bridge 啟動且仍活著的 GUI 不會在另一個空 port 假裝再次啟動；MCP 清理只斷線，不殺 GUI。所有硬體 gate、取消與 operation 結果都仍歸 GUI owners。

## 驗證

`tests/mcp/measure/` 以 public tools/session、recording transport 驗證 catalog、連線、guard、operation。GUI remote/service 測試驗證真 socket 與 GUI-origin path。離線選集只用 fake/mock，不啟動真儀器；測試路徑與 fixture 見 `tests/README.md`。
