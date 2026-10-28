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

Aliases 的 revision 是固定 capture 當次觀察到的原始 bytes SHA-256，不表示內容有效。選取 point 時，壞 aliases 可連同原錯及 hash 保存在 session，參數與 library 仍可使用；實際 alias consumer 才報該 capture 的原錯。只有缺少 aliases 檔時 revision 才是 None。修復後安裝的新狀態不改舊 capture，讀取 owner 與 session 的分工見 [[0063]]。

這項語意只涵蓋 alias read／capture，不授權參數或 library 安裝壞內容，也不改變下方 watcher／reload 失敗保留舊狀態的規則。2026-10-29，使用者對「aliases hash 不保證有效」的 capture 語意與文件補充回覆「同意」。

### Point availability 與使用時驗證

Point 清單只投影 description，availability 另判參數文件可讀且能解析成 owner 文件結構，保留 header 與目前 schema version 檢查。不讀 library 或 aliases，不預驗整份 component model。局部 leaf／component 格式問題在使用或寫入時報錯，不使整個 point 不可用。

讀不到或解析不了的 point 仍不可用並附原因，其他 point 正常。UI 的停用列、不可當 clone 來源、tooltip 與 remote error 呈現保持。Availability 不作為 use_point、Load、Save 或其他讀取的額外前置條件，也不放寬下方 write-through、watcher／reload 安裝驗證與安全規則。按需讀取與寫入驗證的 owner 分界見 [[0063]]。

### 所有寫入 write-through

任何 writer（Inspect 對話框、writeback、MCP、recipe、啟動腳本）都經 owner：驗證 → 寫暫存檔並 rename → 算 hash → 安裝記憶體 → 發布事件。安裝成功即落盤，不存在「記憶體比磁碟新」的狀態。`resources` 既有的單檔原子替換是唯一寫檔路徑。

### 外部改檔由 watcher 重載

GUI 監看 point 目錄，去抖後讀檔算 hash：等於記憶體 revision 視為自己的寫入，忽略；不等於則解析並驗證，合法即安裝並發布「來源已重載」，不合法則保留記憶體、在 Inspect 顯示該檔的錯誤與原因，不改檔、不自動修。Tab 的固定 capture 不變，Refresh／Reset 才取新值。

### 衝突只在人有草稿時

已安裝狀態不會與磁碟衝突。Inspect 頁有未提交草稿而磁碟變更時，草稿保留，橫幅提供「重載」（丟草稿、載磁碟）與「保留」。Apply 帶載入時的 revision，與磁碟不符即拒絕並再顯示橫幅；人先重載再重做修改，不自動合併。

### Holder 門檻

重載是 mutation，沿 [[0070]] 的單一 holder：agent 非 holder 時 `context.reload` 拒絕；watcher 偵測到外部變更而人持有時不安裝，只顯示橫幅。agent 持有時 GUI 編輯停用，衝突不會發生。

### Agent 直接讀寫檔案

Agent 直接讀檔取得磁碟內容。`status` 回已安裝版本的 revision，外部候選在通過驗證前可能與它不同。Aliases 即使已有 revision，內容仍可能無效；不要把取得 hash 當成驗證成功。

Agent 編輯檔案後呼叫 `context.reload(path, expected_revision)`，expected_revision 是編輯前的已安裝版本。Owner 讀檔、驗證、安裝、發布並回新 revision。驗證失敗回檔案路徑與原因，保留舊記憶體及磁碟候選。Revision 不符時重讀狀態與內容再決定，非 holder 則拒絕重載。既有 RPC 寫入可選帶 `expected_revision`。

修復固定 capture 中的壞 aliases 時，agent 仍以該次 observed hash 作為編輯前的 expected_revision，沿原 holder 與 reload 契約提交。外部再次修改造成 revision 不符時照舊拒絕，不新增 wire 欄位。操作步驟見 [MCP Point 檔案讀寫](../../lib/zcu_tools/mcp/measure/README.md#point-檔案讀寫)，capture 行為見 [session owner](../../lib/zcu_tools/gui/session/README.md#contextsession-environment-與-run-snapshot) 與 [alias 檔案 owner](../../lib/zcu_tools/resources/entry/README.md#setup-與-point-alias-表)。

### MCP 固定工具收斂

Measure 固定工具不提供參數、modules、aliases 的特化讀寫。`status` 固定工具回檔案路徑與已安裝 revision。`entry.info` 與 `context.reload` 經 `rpc_call` 存取，不另設固定工具。Live catalog 的既有 method 仍可經 `rpc_call` 使用。Recipe 內部仍走 owner API，不受影響。

## 代價與不採用的方案

- 不採「記憶體為真相、Save 才落盤」：agent 讀檔會讀到舊值，改檔會被覆蓋。
- 不採自動三方合併：表是小表，重做比合併便宜且不會錯。
- 不採 peer 檔案鎖：holder 已是互斥機制，再加鎖是第二套。
- GUI 重寫檔案時 YAML 排版與註解不保留，接受「檔案由 owner 排版」。
- 外部編輯器非原子寫入可能被讀到半檔，由去抖與「不合法保留舊值」吸收。
- 不增加跨檔原子性、fsync 或掉電保證，與 [[0063]] 一致。
