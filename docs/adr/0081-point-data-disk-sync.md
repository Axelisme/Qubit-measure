---
status: proposed
date: 2026-10-23
---

# ADR-0081 — Point 資料以磁碟為真相：write-through、watcher 重載、revision 樂觀鎖與 agent 直接讀寫檔案

## 決策狀態與問題

本篇記錄已採用的方向，實作尚未完成。它補充 [[0063]]（保存權威）、[[0067]]（ContextService 擁有 alias 表與 Apply 時機）、[[0068]]（measure 固定工具與 live catalog）與 [[0070]]（單一 holder），不改 tab 的固定 capture（[[0065]]）。

目前 point 的參數文件、module library 與 `aliases.yaml` 由 GUI owner 在記憶體安裝後才擇時落盤；MCP 為每種資料各提供特化的讀寫工具（`value.read`／`value.list`、`context.set_parameter`、`context.ml_*`、`context.aliases_set`、`tab.parameter_read`）。Agent 讀到的是工具的投影而非完整文件，記憶體可能比磁碟新，agent 若直接改檔會被 GUI 下一次保存覆蓋，且繞過驗證、發布與 holder。使用者 2026-10-23：「我考慮將parameter/modules/alias的特化讀取工具從mcp tool中移除，寫入也可能？因為直接edit/read/write檔案好像對agent最友好？」，對本篇的同步方案「我認為這個方案是可行的」。

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

讀：agent 直接讀檔，write-through 保證即現值；status 的 revision 供判斷手上內容是否過時。寫：agent 編輯檔案後呼叫 `context.reload(path, expected_revision)`，owner 讀檔、驗證、安裝、發布並回新 revision；拒絕分三種：驗證失敗（含路徑與原因，記憶體與檔案都不變）、revision 不符（重讀再來）、holder 不符。既有 RPC 寫入可選帶 `expected_revision`。

### MCP 固定工具收斂

measure 固定工具移除參數、modules、aliases 的特化讀寫工具；保留 `entry.info`／`status` 的路徑與 revision、`context.reload`，以及 `rpc_call` 對 live catalog 既有 method 的存取。recipe 內部仍走 owner API，不受影響。

## 代價與不採用的方案

- 不採「記憶體為真相、Save 才落盤」（現況）：agent 讀檔會讀到舊值，改檔會被覆蓋。
- 不採自動三方合併：表是小表，重做比合併便宜且不會錯。
- 不採 peer 檔案鎖：holder 已是互斥機制，再加鎖是第二套。
- GUI 重寫檔案時 YAML 排版與註解不保留，接受「檔案由 owner 排版」。
- 外部編輯器非原子寫入可能被讀到半檔，由去抖與「不合法保留舊值」吸收。
- 不增加跨檔原子性、fsync 或掉電保證，與 [[0063]] 一致。
