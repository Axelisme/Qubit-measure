**Last updated:** 2026-09-27，operation handle reconnect safety

# `zcu_tools/mcp/measure/`

這是 measure-gui 的 MCP driving adapter。它只透過 GUI 的 loopback remote socket 操作同一份 GUI 狀態；GUI core 不 import MCP。GUI remote method entries 擁有每個 wire method 的 exposure、guard、read-reveal、成功寫入刷新 baseline 與 operation policy。MCP 連線時載入 live `rpc.catalog`，不維護第二份 method/policy 表，也不根據 catalog 動態建立 tools。

## 連線與操作

- `assembly.py` 建立固定手寫工具表。`connect`、`status`、`wait`、`cancel` 與三個 `rpc_*` 是首批入口；其餘 domain tools 依各自 ticket 接入。`rpc_call` 只能呼叫 catalog 標為 `rpc` 的 method。
- `session.py` 擁有單一 MCP session 的 catalog、guard observation、bridge 與 opaque integer operation handles。明確重連或非預期 EOF 後清 catalog/observations/舊 handle 對應；下一個 GUI incarnation 可重用 wire operation ID，但不重用此 MCP session 曾向 agent 外露的 handle。GUI-origin operation 由 `status` 收錄，與 agent-started operation 使用同一映射；wait/cancel/progress 在每次 wire 操作前確認連線，再把 opaque handle 解析成該 GUI 世代的 ID。送出前若斷線即失敗，不用舊 ID 向重啟後的 GUI 重送。這不是第二個 operation outcome store。
- 資源版本由 GUI owner bump。MCP 在完整 read 前取保守版本，成功後只更新 catalog 指定的資源；`prefix` 局部讀取不揭露整份 cfg，status 的 orientation reads 不吸收其他資源。GUI catalog 宣告哪些寫入回傳 owner-thread 前後版本；MCP 只更新該次變更且寫入前版本符合既有觀察的資源，不吸收別的 GUI 編輯。新建 `tab.new` 回執只確立新 tab 的存在版本，不把未讀 cfg 或其他資源當成已觀察。stale 拒絕不刷新 baseline，需重讀資源後才由呼叫者決定是否重試；斷線或 transport timeout 不自動重送。
- 接手既有或重啟後的 GUI 時，明確呼叫 `tab.snapshot(tab_id)`、`soc.info(include_cfg=true)` 和 `context.snapshot`，分別重讀 tab 存在、完整 SoC cfg、目前 active label 與所有可序列化 md/ml cfg。`context.snapshot` 可能回傳大型敏感資料，遇無法序列化的值會失敗且不刷新版本；摘要、局部 getter 與裸 `resources.versions` 都不能替代完整讀取。
- `bridge` 只管 socket/GUI subprocess。`connect(token=...)` 使用現有 GUI control-token 認證；session 留住本次憑證供斷線後重新握手，顯式切換 port 不沿用前一 GUI 的 token。未授權與 wire 不相容分別回報；MCP 工具記錄遮蔽 token。`connect(launch=...)` 對已由此 bridge 啟動且仍活著的 GUI 不會在另一個空 port 假裝再次啟動；MCP 清理只斷線，不殺 GUI。所有硬體 gate、取消與 operation 結果都仍歸 GUI owners。

## 傳輸上限

Shared SocketTransport 送出前與接收逐幀使用 shared framing 的8 MiB UTF-8 bytes上限，
不含換行，不分批。超限request在送出前拒絕，既有連線仍可使用；超限response會關閉
該連線並使pending RPC收到明確的message_too_large錯誤，不能假定mutation未執行。
兩者都不自動重送；重新連線仍走既有catalog與observation重建流程。

## Cfg 讀取

`tab.get_cfg`／`editor.get` 回完整 typed observation，包含 locked 欄位、raw/resolved/error、
validity 與 cached choices；GUI model 是來源，讀取不重新解析 md/ml。Prefix 回指定 node，
保留其完整 path；即使 prefix 是空字串也不更新整份 cfg 觀察版本。失敗讀取與裸版本表
不推進基線，其他 cfg 的更新不影響目標 cfg。Wire 格式與描述由 GUI catalog 擁有。

## 驗證

`tests/mcp/measure/` 以 public tools/session、recording transport 驗證 catalog、連線、guard、operation。GUI remote/service 測試驗證真 socket 與 GUI-origin path。離線選集只用 fake/mock，不啟動真儀器；測試路徑與 fixture 見 `tests/README.md`。
