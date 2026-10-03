**Last updated:** 2026-10-03, Onetone recipes and Run preview

# `zcu_tools/mcp/measure/`

這是 measure-gui 的 MCP driving adapter。它只透過 GUI 的 loopback remote socket 操作同一份 GUI 狀態；GUI core 不 import MCP。GUI remote method entries 擁有每個 wire method 的 exposure、guard、read-reveal、成功寫入刷新 baseline 與 operation policy。MCP 連線時載入 live `rpc.catalog`，不維護第二份 method/policy 表，也不根據 catalog 動態建立 tools。

## 連線與操作

- `assembly.py` 建立固定手寫工具表。01／02 提供 `connect`、`status`、`wait`、`cancel` 與三個 `rpc_*`；05 增加 `experiments`、`guide`、`tab_open`、`tab_get`、`tab_live`、`screenshot`；04 的 predictor 工具經 GUI 同一 `PredictorService` 讀、載、預測與單點 bias 校正；device 四工具沿 GUI `DeviceService` 驗證欄位與讀取現況，使用 02 的 opaque operation handle 等待、取消或逾時後恢復，不另存一份操作結果。其餘 domain tools 依各自 ticket 接入；`rpc_call` 可呼叫 catalog 中所有公開 method，包括標為 `tool` 的 method。Tool names 只提示可用的高層工具，不形成另一種 mode 或權限。`project.info` 提供已套用 project 的身份與實際目錄；`context.labels` 提供全部 labels。`tab_get` 的 artifacts 直接投影 GUI State 快照，含 status、default_path、last_saved_path 與 is_saveable。MCP 只轉換 artifact key 與 data/image kind，不自行推導 dirty 或掃描磁碟。cfg完整投影包含型別、選項、鎖定與cached值。
- `tab_save` 送出一次 GUI-owned batch operation，不在 MCP 迴圈存各 artifact。GUI 在啟動存檔前切到目標 tab 的 Data pane；讀取及非同步完成不切頁。明確 paths/comment 更新共同草稿，省略則沿用。短等完成才回 saved 實際路徑；未完成回 op，失敗保留 operation 診斷。長存檔和部分成功從 `tab_get` 的 last_saved_path 查，不另存 operation payload。Agent 須先明確讀 summary/artifacts，工具不預讀或重送 stale mutation。
- `session.py` 擁有單一 MCP session 的 catalog、bridge 與 opaque integer operation handles。明確重連或非預期 EOF 後清 catalog/舊 handle 對應；下一個 GUI incarnation 可重用 wire operation ID，但不重用此 MCP session 曾向 agent 外露的 handle。GUI-origin operation 由 `status` 收錄，與 agent-started operation 使用同一映射；wait/cancel/progress 在每次 wire 操作前確認連線，再把 opaque handle 解析成該 GUI 世代的 ID。送出前若斷線即失敗，不用舊 ID 向重啟後的 GUI 重送。這不是第二個 operation outcome store。
- `session.bind()` 回傳固定 GUI generation 的 `GuiConnection`。RPC、catalog 與 handle 對應由同一把鎖保護。每次多步 tool invocation 都保持同一 binding，包含 nested wait、後續讀取與失敗 cleanup；下一次獨立 invocation 才可重新連線。No-op connect 不失效，真正重連或斷線後的下一步明示 `connection_lost`，不自動接到新 GUI。已收到的同代回覆仍是已知結果，不因隨後 socket 關閉而變成未知。Catalog 與 handle 對外提供快照，不暴露內部可寫 alias。Stdio 仍同步處理工具呼叫。分析 worker 沿原 binding 執行固定續行流程，不自行重連，也不承諾與其他長 RPC 平行的延遲。
- GUI owner bump 資源版本，remote adapter 保存每連線的 seen。完整讀取成功才記錄宣告的資源；部分讀取、失敗、逾時與回覆編碼失敗不建立觀察。未看過的 key 即使版本 0 仍拒絕。自寫只推進先前 seen 等於寫入前版本的資源；未看過的連帶 cfg 變更不加入 seen。MCP 不保存版本、不送 expected_versions、不解析寫入收據。stale、斷線或 timeout 都不自動重送。
- `tab_get` summary/artifacts 與 `tab_live` 保留完整 `operation_state`，含 result/analysis revisions、availability 及有效 paths。cfg-only 讀取不暗中讀 snapshot；原始 result 陣列不是操作狀態的必要內容。
- `tab_open(from_file)` 只送一次 GUI `tab.open_file`，不隱藏預讀。Agent 必須先讀 context。GUI 負責建立、載入、失敗清理與聚焦；成功另回 cfg_backfill，not_applied 保留結果。新 tab 只建立存在 baseline，後续寫入仍需明確讀取對應資源。
- 接手既有或重啟後的 GUI 時，明確呼叫 `tab.snapshot(tab_id)`、`soc.info(include_cfg=true)` 和 `context.snapshot`，分別重讀 tab 操作狀態、完整 SoC cfg、目前 active label 與所有可序列化 md/ml cfg。`context.snapshot` 可能回傳大型敏感資料，遇無法序列化的值會失敗且不刷新版本；摘要、局部 getter 與裸 `resources.versions` 都不能替代完整讀取。
- 圖像由 GUI owner 渲染，MCP session 擁有暫存 PNG，server 關閉時清理。`tab_analyze` 與 `tab_interact` 在同一回覆交付絕對路徑與 MCP image content；其他圖像工具仍只回路徑。這些暫存圖不代表持久 artifact 已保存。Core 的 `ToolReply` 持有本次 structured data、PNG bytes 與 `is_error`。Stdio 在失敗時保留 partial data 和圖片，並標示 `isError=true`；call log 只記 data。圖片不經全局 buffer。`tab_live` 的 elapsed_s 來自 GUI operation handle 的單一起時，不取各進度條 elapsed 的最大值。
- `bridge` 只管 socket/GUI subprocess。`connect(token=...)` 使用現有 GUI control-token 認證；session 留住本次憑證供斷線後重新握手，顯式切換 port 不沿用前一 GUI 的 token。未授權與 wire 不相容分別回報；MCP 工具記錄遮蔽 token。`connect(launch=...)` 對已由此 bridge 啟動且仍活著的 GUI 不會在另一個空 port 假裝再次啟動；MCP 清理只斷線，不殺 GUI。所有硬體 gate、取消與 operation 結果都仍歸 GUI owners。

## 傳輸上限

Shared SocketTransport 送出前與接收逐幀使用 shared framing 的8 MiB UTF-8 bytes上限，
不含換行，不分批。超限request在送出前拒絕，既有連線仍可使用；超限response會關閉
該連線並使pending RPC收到明確的message_too_large錯誤，不能假定mutation未執行。
兩者都不自動重送；重新連線重新載入 catalog，GUI seen 從空集合開始。

## Cfg 讀取

`tab_get(include=['cfg'])` 回完整 cfg publication，含 cfg_ref、status、tree、source_basis
及 diagnostics。讀取不重新解析 md/ml，也不暗中讀 tab snapshot。`tab_edit` 將觀察到的
cfg_ref 作為 expected，原樣送一次 `tab.edit_cfg`。Path 是 string array，revision 是
canonical decimal string。GUI 原子接受整批，或保持上一份 publication。成功可以發布
Invalid；stale、busy 和 malformed batch 不重試。Cfg expected/actual 與其他資源的 seen guard 分開。

`rpc_call('tab.reset_cfg', {tab_id, expected})` 用相同 cfg_ref 契約重設 cfg。GUI resource
重新取得 adapter defaults，並回完整 publication。每次成功 command 推進一次 revision，
即使值未改變。Reset 不清除 result、不存檔，也不新增 unsaved guard；Active Run 仍拒絕 reset。

`tab_run(tab, expected)` 必須帶入觀察到的 cfg_ref。工具原樣轉送一次，不預讀、refresh 或
自動重試。GUI 只接受指定的 Valid publication，stale 會回 expected/actual；Run 使用固定的
cfg 與 source_basis，不因後續來源變更而替換輸入。

`editor.get` 保留獨立 library draft 的 typed tree 與 prefix 規則。失敗讀取及裸版本表不推進
基線。Wire 格式與描述由 GUI catalog 擁有。

## 關閉

`tab_close`與`shutdown`只送一次GUI命令，GUI在同次owner dispatch檢查active operations與全部unsaved artifacts。`discard_unsaved`不能略過busy。GUI自身data-only提示不變。

`shutdown`等待回覆中的GUI PID自然退出，最多五秒，不以shared PID file選程序。不呼叫bridge.stop或送終止信號；請求或等待逾時回stopped=false，讓操作者處理，不自動重試。

`MeasureMcpSession.close()` 永久拒絕新 execution，先喚醒等待、斷線並解除 pending RPC，再 join 本 session 的 recipe 與分析 workers，最後清理 PNG。關閉與 PNG 寫入共用資源鎖，不讓清理後的目錄重生。Server 在 `finally` 呼叫 close，不依賴正常 EOF。這不取消 GUI 已接受的副作用，也不保證硬體停止。

## Library 編輯

`ml_edit`只送一次GUI application命令。CfgEditorService使用共用CfgDraft，經ContextWritePort逐項提交；首錯即停，保留已提交前綴並清理內部草稿。回覆區分applied、failed、skipped與實際cfg。save_as不修改來源，首次成功才建立目的地。Agent須明確觀察context，沒有editor/context隱藏預讀或自動重試。

Library rename/delete只改library；LINKED參照保留舊鍵並可能失效，MODIFIED參照保留inline修改。既有draft由service反應library變更並發布，同一份狀態供widget與MCP觀察。

## Interactive

`tab_interact` 只轉送一次 active plugin command，不解讀實驗專屬命令。省略 payload 時回 committed state、commands、info、preview_active 與 figure，不改焦點。GUI 先驗證 command，再跟隨 Analysis pane 執行。此介面採 best-effort，不加 seen guard，後提交者為準；沒有來源鎖或重試。

保留命令 `done` 使用 `include_figure=false`，先取得原 operation 的 receipt，不讓 PNG 渲染失敗遮住已接受的命令。MCP 隨後 join 該 op 的 execution；GUI-origin session 尚無 execution 時，done 建立 Primary 完成責任。Done 驗證失敗不保存，也不終止仍可編輯的 session。純讀與普通 command 不接管未登記的 GUI-origin session；已有 execution 時更新其 interaction 快照。Done 以前的 PNG 失敗回 tool error，保留 op／execution，不取消互動。

## Recipes

Root `recipes/` 是普通 Python package。`recipes/__init__.py` 明確登記名稱、schema 與函式，
MCP assembly 使用同一份 registry 建立 tools。編輯 recipe 函式即可調整流程，沒有目錄掃描或熱重載。
GUI 仍擁有 cfg defaults、validation、資源 guard 與結果。

`lookback` 新建 tab，或以 `reuse_tab_id` 重設指定的 idle Lookback tab。
兩條路使用同一套有限參數。明確無效輸入不 fallback，reuse 失敗不另建替代 tab。
頻率來自明確參數、指定 library readout 的有效 leaf 或 `r_f`。
沒有可用頻率時回 `needs_parameters`，不以 GUI 模板的盲選頻率開始 Run。

`onetone_spectrum`、`onetone_spectrum_over_flux` 與 `onetone_spectrum_over_power`
使用 GUI 的頻率範圍與平均數設定。中心省略時要求有效的 `r_f`，完整 span 省略時
要求有效的 `rf_w`，範圍倍率仍由 GUI 決定。缺項一次回 `needs_parameters`，不盲掃。
明確的 library readout、設備或數值無效時失敗，不改選預設來源。
Flux 使用已註冊的設備來源與 GUI 提供的物理單位，不把 normalized flux 當實際單位。
三者沿用與 Lookback 相同的 reuse、單次 Run、raw 保存與控制流程。

Spectrum 與 flux 交給 Primary 分析。Power 是 raw-only，回 `analysis_mode=none`，
`analysis` 與 `writeback` 為 null。它在 raw 保存後以原 Run token 取得 preview，
回傳標為 `run_preview` 的 session 暫存 PNG 與同 reply 的 image content。
Preview 不是已保存的 analysis artifact，失敗保留已保存 raw 並標示 `phase=preview`。

`RecipeContext` 持有固定 GUI binding 與整段進度，只執行一次 Run。
Run 後只收錄來源對應原 Run 的 result_state。來源已被替代時回報 result_superseded，不交付另一輪快照。
Raw save 與 Primary analysis 都帶原 Run 來源，GUI 拒絕已被另一輪資料取代的來源。
需要 Primary 的 recipe 在 raw 確認成功後才交給既有 AnalysisExecution 分析、逐張存圖與交付 PNG。
首次互動交接與獨立分析工具共用 interaction delivery，回傳 committed state、commands 與可用 PNG。
交付失敗保留原 execution 與已保存 raw，後續仍可用 tab_interact 繼續。
最後讀取同次分析的 writeback 候選，不自動 accept。失敗保留已確認的 raw、圖像路徑與分析結果。

首次 recipe 呼叫最多等 300 秒，逾時返回 execution，worker 繼續。
`status(execution)` 與 `wait(execution)` 讀取這段流程的本地快照，互動階段立即交接。
`cancel` 與 `finish_early` 恰好接受 op 或 execution 之一。Recipe 的原 Run op 或目前 op
都定位同一個控制狀態。Finish early 只在 Run 階段適用，有可用 partial data 才繼續保存與分析。
Cancel 優先，禁止開始新的保存、分析、preview 或 writeback 讀取。
已開始的 raw save 不可取消，仍等待真成功、失敗或 unknown。
已開始的 preview 也保留真結果，不以取消意圖掩蓋錯誤。

Recipe 沒有重試、tab 自動清理或持久 recovery。背景工作不重連另一個 GUI，
session close 停止 admission、解除 pending RPC 並 join workers，不保證硬體已停止。

## Analysis execution

`tab_analyze` 只 start 一次，每個 op 在本 MCP session 只有一個 execution 負責完成與保存。Primary 和 Post 都沿原 op 等待、讀取 summary／實際 params／operation_state，再按原結果的 ordered figure_names 逐張保存。GUI 決定預設保存路徑，MCP 保留每張已確認的實際路徑。全部保存成功後才讀 pane preview，驗證 PNG 並交付暫存路徑與 image content。不重存 raw、不 auto accept，也不 refresh、重試或 rollback。

沒有圖時回 `save_status=not_available`。保存首錯即停，回成功 prefix 與 remaining names；已送請求但丟失回覆時標示 unknown，不宣稱檔案不存在。分析、結果讀取、保存及最終 PNG 失敗分別保留 error phase，已取得的結果與保存路徑不清空。結果被取代時如實失敗，不交付另一輪資料。

非互動 start 或 done 最多短等兩秒，返回後 worker 繼續。互動 start 立即交接 state、commands、可用圖、tab、op、params 與 execution；worker 也會觀察 GUI 端完成。Execution 是 session-local identity，不是 GUI operation ID，沒有持久 recovery。

- `status(execution=...)` 只讀本地 detached 快照，不重連 GUI。普通 `status` 另附本 session 的 executions。
- `wait` 恰好接受 op 或 execution 之一，timeout 為有限的 0 至 300 秒。Op 只等 GUI operation；execution 等分析、保存與圖像交付，遇互動立即交接。逾時只停止等待，不取消工作。
- `cancel` 同樣恰好接受 op 或 execution 之一。已登記 op 與 execution 先鎖存同一取消意圖，再沿原 binding 最多送一次 GUI cancel。Not-cancellable 不等於工作已停止；worker 繼續觀察原 op，但不開始新結果讀取、保存或 PNG。已送出的保存保留真成功、失敗或 unknown。

Execution 的 status 與 GUI operation_outcome 分開。取消後遇真 failure 仍回 failed，不以 cancelled 遮住；terminal 快照不被後來 cancel 或重連改寫。`tab_analyze`／完成的 `tab_interact` 以 partial error ToolReply 交付 failed execution；`status`／`wait` 成功讀到 failed outcome 仍是正常 query 回覆。

## Writeback

`accept(tab)` 接受目前 Primary 與存在的 Post 全部候選，不受 GUI 勾選影響，也不更改勾選。它先讀兩個結果摘要，再依 Primary、Post 順序 preview 並寫入各 pane 的完整 IDs；跨 pane 的同名 ID 不合併。無結果、無草稿或空候選的 pane 記為 skipped。摘要與 preview 不更新觀測游標，寫入仍需通過原 GUI guard。

`accept` 首錯即停，不重試或 rollback。completed 只列 GUI 已確認成功的 pane 與其 written 結果；failed_stage 指出失敗 pane，其餘尚未處理的 pane 列在 not_started。摘要或 preview 失敗不開始該 pane 的 mutation；寫入失敗可能已有部分副作用，回覆會明示這點。若任一摘要查詢失敗，尚未開始任何 pane 寫入。

`writeback` 的 preview 直接投影 GUI 共享草稿與目前 context，不在 MCP materialize cfg。寫入只送一次 `tab.writeback_write`；GUI 依序修改指定草稿，首錯保留已改前綴且不開始 context apply。全部成功後一次 apply 指定 IDs，不改 GUI 勾選；結果以含 id、kind、target、before、after 的列表保留跨 kind 同名目的地。MCP 不隱藏預讀、不重試。GUI 在改草稿前透過明確 view 命令切到目標 analysis/post pane；preview 與非同步完成不切頁。

## 驗證

`tests/mcp/measure/` 以 public tools/session、recording transport 驗證 catalog、連線、guard、operation。GUI remote/service 測試驗證真 socket 與 GUI-origin path。離線選集只用 fake/mock，不啟動真儀器；測試路徑與 fixture 見 `tests/README.md`。
