---
status: accepted
date: 2026-10-23
---

# ADR-0081 — Point 資料以磁碟為真相：write-through、watcher 重載、revision 樂觀鎖與 agent 直接讀寫檔案

## 決策狀態與問題

本篇規範 point 檔案的寫入、重載與 revision 檢查。它補充 [[0063]] 的保存權威、[[0067]] 的 ContextService 與 alias 表、[[0068]] 的固定工具與 live catalog，以及 [[0070]] 的單一 holder。不改 [[0065]] 的 tab 固定 capture。

GUI 與 agent 都需要讀寫 point 的參數、module library 與 `aliases.yaml`。如果 owner 只更新記憶體、稍後才保存，agent 會讀到舊檔案，外部編輯也可能被下一次保存覆蓋。直接改檔則需要 owner 驗證、安裝與發布，不能只把磁碟寫入當成生效。使用者 2026-10-23：「我考慮將parameter/modules/alias的特化讀取工具從mcp tool中移除，寫入也可能？因為直接edit/read/write檔案好像對agent最友好？」，對本篇的同步方案「我認為這個方案是可行的」。

## 決策

### 磁碟是真相，記憶體是帶 revision 的快取

每個 point 的參數文件、library 文件與 `aliases.yaml` 是唯一權威。記憶體保存已安裝內容與該檔內容的 hash 作為 revision，各檔獨立。`entry.info` 與 `status` 回每個檔的路徑與 revision。

### 所有寫入 write-through

任何 writer（Inspect 對話框、writeback、MCP、recipe、啟動腳本）都經 owner：驗證 → 寫暫存檔並 rename → 算 hash → 安裝記憶體 → 發布事件。安裝成功即落盤，不存在「記憶體比磁碟新」的狀態。`resources` 既有的單檔原子替換是唯一寫檔路徑。

### 外部改檔由 watcher 重載

GUI 監看 point 目錄，去抖後讀檔算 hash：等於記憶體 revision 視為自己的寫入，忽略；不等於則解析並驗證，合法即安裝並發布「來源已重載」，不合法則保留記憶體、在 Inspect 顯示該檔的錯誤與原因，不改檔、不自動修。Tab 的固定 capture 不變，Refresh／Reset 才取新值。

### 衝突只在人有草稿時

已安裝狀態不會與磁碟衝突。Inspect 頁有未提交草稿而磁碟變更時，草稿保留，橫幅提供「重載」（丟草稿、載磁碟）與「保留」。Apply 帶載入時的 revision，與磁碟不符即拒絕並再顯示橫幅；人先重載再重做修改，不自動合併。

### Holder 門檻

重載是 mutation，沿 [[0070]] 的單一 holder：agent 非 holder 時 `context.reload` 拒絕；watcher 偵測到外部變更而人持有時不安裝，只顯示橫幅。agent 持有時 GUI 編輯停用，衝突不會發生。

### Agent 直接讀寫檔案

Agent 直接讀檔取得磁碟內容。`status` 回已安裝版本的 revision，外部候選在通過驗證前可能與它不同。

Agent 編輯檔案後呼叫 `context.reload(path, expected_revision)`，expected_revision 是編輯前的已安裝版本。Owner 讀檔、驗證、安裝、發布並回新 revision。驗證失敗回檔案路徑與原因，保留舊記憶體及磁碟候選。Revision 不符時重讀狀態與內容再決定，非 holder 則拒絕重載。既有 RPC 寫入可選帶 `expected_revision`。

### MCP 固定工具收斂

Measure 固定工具不提供參數、modules、aliases 的特化讀寫。`status` 固定工具回檔案路徑與已安裝 revision。`entry.info` 與 `context.reload` 經 `rpc_call` 存取，不另設固定工具。Live catalog 的既有 method 仍可經 `rpc_call` 使用。Recipe 內部仍走 owner API，不受影響。

## 代價與不採用的方案

- 不採「記憶體為真相、Save 才落盤」：agent 讀檔會讀到舊值，改檔會被覆蓋。
- 不採自動三方合併：表是小表，重做比合併便宜且不會錯。
- 不採 peer 檔案鎖：holder 已是互斥機制，再加鎖是第二套。
- GUI 重寫檔案時 YAML 排版與註解不保留，接受「檔案由 owner 排版」。
- 外部編輯器非原子寫入可能被讀到半檔，由去抖與「不合法保留舊值」吸收。
- 不增加跨檔原子性、fsync 或掉電保證，與 [[0063]] 一致。
