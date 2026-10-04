**Last updated:** 2026-10-04, captured execution summaries and session previews

# `zcu_tools/mcp/measure/`

Measure MCP 透過 GUI 的 loopback remote socket 操作同一份 GUI 狀態。GUI core 不 import MCP。GUI `RemoteMethodEntry` 擁有 wire schema、exposure、guard、read-reveal、寫入後觀察刷新與 operation policy。MCP 每次連線讀取 live `rpc.catalog`，不複製 policy 表，也不依 catalog 動態建立 tools。

## 固定工具與 RPC

`assembly.py` 組合固定的 23 個 tools。Recipe registry 擁有 11 個 recipe 的名稱、schema 與執行入口。其餘為 `connect`、`status`、`wait`、`cancel`、`finish_early`、`rpc_list`、`rpc_describe`、`rpc_call`、`tab_analyze`、`tab_interact`、`tab_close` 與 `accept`。

日常量測優先 recipe，既有資料分析使用共用分析工具。細部 setup、cfg、保存、writeback 及排查由 RPC 承接。日常／排查是使用指引，不是權限模式。`rpc_call` 接受 catalog 中全部 `rpc` 與 `tool` methods；tool 名稱只提示高層入口。GUI 仍驗證每次操作。公開 method 的參數、回覆及前置條件由 `rpc_describe` 提供。

Raw RPC 不提供高層工具的結果聚合、PNG 解碼或 canonical analysis-image 保存流程。例如 `device.connect` 回傳 handle，caller 用 `wait(op)` 等待後再讀 `device.snapshot`。`tab.save_artifacts` 的 reserved destinations 不代表保存成功，完成或失敗後應讀 `tab.snapshot` 的 artifact 狀態與實際路徑。

## Recipe 與 execution

Recipe 將既有 GUI 操作串成有界實驗流程，不在 MCP 複製實驗核心或 cfg defaults。參數來源、缺參數與分析分支由個別 recipe 宣告。首次呼叫等待最多 300 秒；缺參數、失敗或互動需求會提早交付。仍執行時回傳 execution，背景接續不因等待逾時而停止。

Flux recipe 的 `flux_unit` 可斷言 GUI 裝置的實體單位，不換算數字。只有已確認的 FakeDevice、unit=`none` 且 caller 明確指定 `native` 才使用 native 座標。回覆保留此單位標記。其他不符情境在 Run 前拒絕，不由 recipe 連裝置或修改共用安全規則。

Client deadline 必須超過 300 秒並留傳輸與回覆開銷。Stdio server 同步處理請求，首次等待期間不保證同連線的另一控制請求立即處理。Client timeout 不等於取消，不可因此自動重跑。

全域 `status` 列出 GUI operations 與目前 MCP session 的非終態 executions，終態只給數量及按 ID 查詢提示。`status(execution)` 讀指定 execution 的本地摘要，不重新連線。`wait(op)` 只觀察 GUI operation；`wait(execution)` 包含後續結果讀取、保存及預覽交付。已接受的分析失敗以 outcome data 回報，和查詢失敗分開。Execution ID 不跨 MCP server session，也不是持久恢復機制。

`finish_early` 對 recipe 停止採集，有可用結果就先保存 raw，再繼續分析及保存。`cancel` 優先，停止後續分析與保存；已啟動且不可取消的保存仍等真實結果。Registered analysis 的 cancel 也不再啟動新的結果讀取。GUI cancellation 回覆獨立放在 `gui_cancel`，不能拿它覆寫 execution 的既有 terminal outcome。未註冊的 `cancel(op)` 沿用直接 GUI hook。

Recipe 不自動挑選重用 tab，也不自動清理。明確 `reuse_tab_id` 的流程先確認可用，再 reset、套本次 cfg 與 Run。關閉由 `tab_close` 明確指定。

## 分析、互動與接受

`tab_analyze` 在既有資料上啟動 Primary 或 Post 分析。Execution 負責結果、實際參數、失效內容、canonical 圖像保存與預覽交付。互動分析立即交接 tab、op、狀態、可用命令與圖像。

GUI 的分析投影把非有限 summary 數字換成 null，以 `invalid` 記錄欄位路徑與原因。
Execution 保存同一份投影，recipe、`status(execution)` 與 `wait(execution)` 不重新推導原因。
不可估誤差不刪除有限 fit value、warning 或已確認的保存路徑，也不觸發自動 accept。
這項表示轉換不改 operation 的 failed outcome 或 generic context 的拒絕規則。

`tab_interact` 省略 payload 時讀 committed state、commands、info、preview_active 與 figure，不改焦點。帶 payload 時 GUI 驗證命令並跟隨 Analysis pane。`done` 接住原 analysis operation，然後加入其 execution 的完成讀取與保存。此 method 不加 seen guard，較晚的 owner-loop commit 生效。沒有來源鎖或自動重試。

Recipe、`tab_analyze`、`wait(execution)` 與 `tab_interact(done)` 使用同一 execution 摘要。摘要列出 Run 前捕捉的 resolved 條件與來源、Primary/Post estimates 和 details、warnings、全部候選及 destination 身分。Run 或分析已送出而 receipt 未確認時保留 unknown，不從缺少 handle 推斷未啟動。`run_id` 目前為 null，execution ID 仍是 session-local。

`status(execution, detail="full")` 保留該 execution 已捕捉的 native 資料，包括完整 cfg publication、raw expressions、source_basis、analysis results 與 writeback proposals。查詢不新增 RPC 或 guard 觀察。Artifact 依 section、名稱與 members 分層，每個 member 是完整路徑及 status 的清單；reserved 不代表已保存，後續失敗不清掉已保存的前綴。

Preview PNG 是 MCP session 專屬暫存檔，同時可附 MCP image content。Server 結束後移除。摘要的 `previews` 固定有 run、primary、post 三個完整 path 字串清單，未取得為空清單。同階段去重並保留首見順序，不猜 named image 身分。摘要的 interaction 不重複 figure；full 保留 native figure、preview 與 interaction。`status` 不附圖片。持久保存路徑在摘要的 `artifacts`，full 的 `saved_images` 只列已確認持久圖像，不把 preview 當成已保存產物。

`accept(tab)` 寫入 Primary 與既有 Post 的全部當前候選，包括 GUI 未勾選項。摘要列出每個候選的 target、proposal 與 current，不把 GUI 勾選狀態當作接受篩選；勾選旗標保留在 full。Destination 摘要只列 context/project 身分，native context readiness 保留在 full，不能替代新的 guard 觀察。它不改勾選，不用 preview 刷新 guard，不回滾已完成寫入。Primary 先於 Post，首錯停止並列出 confirmed completed、skipped、failed stage 與 not_started。Caller 必須先觀察 tab／context，核對提案與當前目的地。個別候選的調整、選擇與寫入使用 `tab.writeback_*` RPC。

## 連線與觀察

`connect` 可連既有 GUI 或依 launch 參數啟動，這一步不連硬體。Control token 留在 session 供重新握手；顯式切換 port 不沿用前一 GUI token，tool log 遮蔽 token。認證失敗與 wire 不相容分別回報。Bridge 不會把仍活著的自有 GUI 當成可在另一空 port 再啟動的程序。

Session 將 GUI operation ID 映射為 opaque integer handle。明確重連或 EOF 後清除 catalog 與舊 handle 對應。GUI 可重用 wire ID，MCP session 不重用已外露 handle。GUI-origin operation 經 `status` 使用同一映射。每次 wire 操作前先確認連線，再解析 handle；不向新 GUI 重送舊 ID。

GUI owner 維護每連線的 seen。成功完整讀取才建立觀察，部分 getter、失敗、逾時或回覆編碼失敗都不建立。版本零也不能替代未曾讀取。自寫只推進先前 seen 等於寫入前版本的資源，不把未看過的連帶變更加入 seen。MCP 不保存第二份版本表，不送 expected_versions，也不以隱藏讀取解鎖一般 mutation。

接手既有或重啟後的 GUI 時，按操作需要明確讀 `tab.snapshot(tab_id)`、`soc.info(include_cfg=true)`、`context.snapshot` 與 `device.snapshot`。Context 完整讀取可能包含大型敏感資料，不可序列化時會失敗而不刷新觀察。`status`、摘要與裸 `resources.versions` 都不替代完整讀取。

Cfg 使用 `tab.get_cfg` 的 publication 與顯式 cfg_ref，包含 cfg identity 及 canonical decimal revision。`tab.edit_cfg`、`tab.reset_cfg` 與 `tab.run_start` 必須帶觀察到的 ref。Run 只接受指定 Valid publication，不換成最新 cfg。這個 ref 與 tab、SoC、device 的 per-connection seen guards 分開。`tab.open_file` 需要先觀察 context；成功的新 tab receipt 只認證存在性，後續資源仍需明確讀取。

Stale、斷線或 timeout 都不自動重送。它們不證明 mutation 沒有副作用；先讀現況，再由 caller 決定下一步。

## 傳輸與關閉

Shared SocketTransport 以 shared framing 限制每個 frame 為 8 MiB UTF-8 bytes，不含換行，不分批。超大 request 在送出前拒絕，連線仍可用。超大 response 關閉連線並回 `message_too_large`，不能假定 mutation 未執行。重新連線會重載 catalog，GUI seen 從空集合開始。

`tab_close` 與 RPC `app.shutdown` 都由 GUI 在同次 owner dispatch 檢查 active operations 及 unsaved artifacts。`discard_unsaved` 不能略過 busy。Shutdown 回覆的 `shutting_down` 與 PID 表示已接受正常關閉要求，不保證程序已退出；RPC 不等 process exit、不強制終止。

MCP server 結束時斷線、加入其 workers，再清理暫存 PNG。它不關閉 GUI，也不保證停止硬體。MCP 不訂閱或佇列 GUI events；完成情況透過 snapshots、operation wait 與 execution 讀取。

## 驗證

`tests/mcp/measure/` 經 public tools/session 與 recording transport 驗證 recipe、分析、catalog、連線、guard 與 operation。GUI remote/service 測試驗證 socket 與 GUI-origin 路徑。離線選集只用 fake/mock，不啟動真儀器；fixture 與測試歸屬見 `tests/README.md`。
