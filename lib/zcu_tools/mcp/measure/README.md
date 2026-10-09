**Last updated:** 2026-10-09, single-Run target and temporary cfg preview

# `zcu_tools/mcp/measure/`

Measure MCP 透過 GUI 的 loopback remote socket 操作同一份 GUI 狀態。GUI core 不 import MCP。GUI `RemoteMethodEntry` 擁有 wire schema、exposure、guard、read-reveal、寫入後觀察刷新與 operation policy。MCP 每次連線讀取 live `rpc.catalog`，不複製 policy 表，也不依 catalog 動態建立 tools。

## 固定工具與 RPC

`scripts/run_measure_mcp.py` 把使用者的 recipe definitions 注入 session 與 `assembly.py`。Registry 驗證名稱、手寫 schema 與 callback，工具表拒絕不一致的注入及重名。Framework 不反向 import 使用者 recipes，也不從 GUI catalog 產生 tools。共用工具為 `connect`、`status`、`wait`、`cancel`、`finish_early`、`rpc_list`、`rpc_describe`、`rpc_call`、`tab_analyze`、`tab_interact`、`tab_close`、`answer` 與 `apply_writeback`，以及 `simulation_initialize`、`device_set_value`、`recipe_guide`。

## Setup 工具

`recipe_guide(recipe)` 使用 registry 的 adapter mapping 讀原生 guide，不從 recipe 名稱猜 adapter。`simulation_initialize` 接現行 GUI coordinator，重用有效 simulation，切換時可能斷開真實裝置。它不套用 project，也不連真實硬體。Caller 仍須取得切換授權。

`device_set_value(name, value, unit)` 先讀裝置 snapshot，只送 value 更新，再等待和核對 actual native value。它不更改 output、mode 或 rampstep。實體 unit 必須與 snapshot 相同。FakeDevice unit=none 只接受明確的 native，不換算工作座標。

兩個 mutation 工具保留 pre_read、start、wait、post_read、requested、before/after、native operation 及 verification。step completed 只表示該步成功。等待逾時不取消 operation，snapshot 有目標值也不代表 operation 完成。Start receipt 不確定時保留 unknown，明確 stale 拒絕時停止。工具不重連、重送或回滾。Caller 用已知 op 的 wait 和完整 snapshot 接手。

## 日常與細部操作

日常量測優先 recipe，既有資料分析使用共用分析工具。細部 setup、cfg、保存、writeback 及排查由 RPC 承接。日常／排查是使用指引，不是權限模式。`rpc_call` 接受 catalog 中全部 `rpc` 與 `tool` methods；tool 名稱只提示高層入口。GUI 仍驗證每次操作。公開 method 的參數、回覆及前置條件由 `rpc_describe` 提供。

Raw RPC 不提供高層工具的結果聚合或 PNG 解碼。例如 `device.connect` 回傳 handle，caller 用 `wait(op)` 等待後再讀 `device.snapshot`。`tab.save_data` 與 `tab.save_artifacts` 同步返回當次完整 Save report，不回 Save handle。Report 的 run_id 沿用 Run 時產生的同一個 ID，即使全部輸出失敗也不為 null。Caller 用各路徑的 status 與 error 判斷保存成功，不以 run_id 是否存在判斷。Report 保留全部 members，不用 await 或再次讀 snapshot。Readonly snapshot 的 confirmed run_id 仍只表示已有持久輸出成功。

## Recipe 與 execution

Recipe 將既有 GUI 操作串成有界實驗流程，不在 MCP 複製實驗核心或 cfg defaults。參數來源、缺參數與分析分支由個別 recipe 宣告。首次呼叫等待最多 300 秒；缺參數、失敗或互動需求會提早交付。仍執行時回傳 execution，背景接續不因等待逾時而停止。

`onetone_spectrum_over_flux` 必填 `flux_device` 與 `flux_unit`，不從 GUI 選擇預設裝置或單位。`flux_unit` 必須符合 GUI 裝置的實體單位，不換算數字。FakeDevice 的 snapshot unit=`none`，caller 必須明確傳 `native`。Recipe 確認裝置後使用 native 座標，回覆保留此單位標記。其他不符情境在 Run 前拒絕，不由 recipe 連裝置或修改共用安全規則。

Client deadline 必須超過 300 秒並留傳輸與回覆開銷。Stdio server 同步處理請求，首次等待期間不保證同連線的另一控制請求立即處理。Client timeout 不等於取消，不可因此自動重跑。

全域 `status` 的 `entry` 是 null 或 GUI `entry.info` 的 UUID、name、result_path、database_path，不投影 chip／qubit／resonator。它只讀 metadata，不建立 context observation。未 opened entry 時 `ready.missing` 包含 `entry`。

全域 `status` 列出 GUI operations 與目前 MCP session 的非終態 executions，終態只給數量及按 ID 查詢提示。`status(execution)` 讀指定 execution 的本地摘要，不重新連線。`wait(op)` 只觀察 GUI operation；`wait(execution)` 包含後續結果讀取、同步保存及當次 image content 交付。已接受的分析失敗以 outcome data 回報，和查詢失敗分開。Execution ID 不跨 MCP server session，也不是持久恢復機制。

`finish_early` 對 recipe 停止採集，有可用結果就先保存 raw，再繼續分析及保存。`cancel` 優先，停止後續分析與保存；已啟動且不可取消的保存仍等真實結果。Registered analysis 的 cancel 也不再啟動新的結果讀取。GUI cancellation 回覆獨立放在 `gui_cancel`，不能拿它覆寫 execution 的既有 terminal outcome。未註冊的 `cancel(op)` 沿用直接 GUI hook。

Recipe 不自動挑選重用 tab，也不自動清理。明確 `reuse_tab_id` 的流程先確認可用，再 reset、套本次 cfg 與 Run。`target` 可搭配新 tab 或重用 tab，只覆寫當次 Run 的第一個 used alias，不傳到 tab.new、不改全域 alias 表。參數／library 查詢使用目前來源與同一 target。帶 target 的 cfg 預覽必須使用原 observed ref；source 變更造成 stale 時拒絕，不換成新 ref 或重試。Execution 的 resolved conditions 捕捉這份 temporary preview，不冒用全域 publication。關閉由 `tab_close` 明確指定。

Author Run 的 cfg、來源與 completion 由 `recipe.py` 的 opaque handles 擁有。Framework 讀 detached snapshots，不讀 handle 的 private backing。`raw_save.py` 捕捉原 Run 的當次同步 Save report，保留全部輸出與該 Run 的 run_id，全部輸出失敗亦保留這個 ID。Author handle 使用這份保存契約。沒有 SaveReport 時 steps.raw_save 為 null，不推論 not_started、unknown 或 Save 是否已送出。save_raw 的傳輸與 admission 失敗照常拋出，recipe 回覆的原因只在 error；不加 requested 旗標。

Author analysis 綁定原 Run 與同一 Run 的 Primary。它重用 `AnalysisExecution`，在 recipe 的 worker 完成分析、圖像與候選 capture。Recipe-local close 和 session close 都能停止接續。寫回提問只篩選已捕捉的 stable names，不重讀 GUI 或寫入。`recipe_execution.py` 擁有 generator、單一 recipe admission、一次性取消、captured question 與 typed snapshot。它以同一 worker 完成 author handles；analysis 共用 driver 的本地 condition 交付進度。Session 與 tool table 使用必填的同一份注入 definitions。Guide 與 analysis-only scalar estimates 也讀 owning registry。單一 recipe 包含 interactive 與 awaiting_answer；terminal 之後才放行下一個呼叫。

## 分析、互動與寫回

Recipe 以 yield 提出 completed capture 的寫回候選。`answer(recipe, decision)` 的 recipe 是 execution ID，decision 為 accepted 或 skipped；它只回答，不寫入也不授權。普通 `tab.accept(items)` 寫 GUI 當前 draft。摘要的 `question_items`／`question_preview` 是 active question，`writeback.receipts` 則列出實際寫入與失敗前綴；兩者不能互相冒充。Query 只讀 local capture。Session close 喚醒並收束 generator finally。

`tab_analyze` 在既有資料上啟動 Primary 或 Post 分析。Execution 捕捉結果、實際參數、失效內容、完整 analysis artifact 的 Save report、當次 image content 及 writeback 候選。互動分析立即交接 tab、op、狀態、可用命令與圖像。

Analysis completion 等原 GUI operation 完成，同步保存完整 analysis artifact，再以原 opaque operation 讀一次 writeback preview。沒有圖像也讀候選。Interactive handoff 不 capture，`done` 加入原 completion，完成後才 capture 定案 draft。屬於 recipe 的 done 等待該 recipe 的下一個結果／提問，不回另一個 analysis execution；standalone 保持原路徑。Recipe 分析由自己的 worker 完成同一份 capture，不另建等待 worker，也不另讀候選。Full 的 writeback 為 null 表示尚未取得；has_draft=false／items=[] 才是已確認空 draft。Read 失敗以 writeback_read error 回報，保留已知 result、當次 Save report 及可用 image content。來源 superseded 不改讀新 pane，不 retry，也不刷新 guard。Recipe wait 在 done 後等待本地 completion 的交接，不再提早交付舊 interactive handoff。

GUI 的分析投影把非有限 summary 數字換成 null，以 `invalid` 記錄欄位路徑與原因。
Execution 保存同一份投影，recipe、`status(execution)` 與 `wait(execution)` 不重新推導原因。
不可估誤差不刪除有限 fit value、warning 或已確認的保存路徑，也不觸發自動寫回。
這項表示轉換不改 operation 的 failed outcome 或 generic context 的拒絕規則。

Framework 原樣傳遞 native analysis summary 與 invalid 路徑，不特別解讀 fit_quality、不重算品質或設定寫回門檻。有擬合的 summary 附 fit_quality 是實驗定義者的約定，見 zcu_lab README。

Analysis-only 沒有 recipe 身分。摘要使用注入 definitions 已宣告的 scalar value_key，完全相同的 name/error_key/unit 宣告可合併。Present key 的宣告衝突直接 ValueError，不猜實驗身分，不新增 tab.snapshot 讀取。每個 pane 保留完整原生 summary 與 invalid；scalar estimates 不抽掉 summary 的欄位，framework 不投影 quality。

`tab_interact` 省略 payload 時讀 committed state、commands、info、preview_active 與 figure，不改焦點。帶 payload 時 GUI 驗證命令並跟隨 Analysis pane。`done` 接住原 analysis operation，然後加入其 execution 的完成讀取與保存。此 method 不加 seen guard，較晚的 owner-loop commit 生效。沒有來源鎖或自動重試。

Recipe、`tab_analyze`、`wait(execution)` 與 `tab_interact(done)` 使用同一 execution 摘要。摘要列出 Run 前捕捉的 resolved 條件與來源、Primary/Post scalar estimates、原生 summary 與 invalid、全部候選及 destination 身分。Run 或分析已送出而 receipt 未確認時保留 unknown，不從缺少 handle 推斷未啟動。`run_id` 取已捕捉 Save report 的原 Run 身分，全部輸出失敗也保留同一 ID。沒有捕捉到 report 時才為 null。各 output 的 status/error 表示保存結果。這與 readonly tab snapshot 的 confirmed run_id 不同，後者在尚無持久輸出成功時仍為 null。Execution ID 仍是 session-local。

`status(execution, detail="full")` 保留該 execution 已捕捉的 native 資料，包括完整 cfg publication、raw expressions、source_basis、analysis results 與 writeback proposals。查詢不新增 RPC 或 guard 觀察。Artifact 沿用 producer 的 section、名稱與完整 members，每個 member 保留全部路徑、format、status 與 error。Reserved 不代表已保存。Partial 保留當次成功與失敗輸出；resave 以當次 report 取代選中 artifact 的輸出，不累積跨次成功路徑。

當次工具回覆可附記憶體中的 MCP image content，不落地 session 暫存 PNG。之後要看圖，用 `tab.get_figure` 取得 GUI 當下圖像，不保證是先前回覆的圖。`status` 不附圖片，也不重新擷取圖像。

`artifacts` 只記錄持久輸出的當次保存結果，不帶 `lifetime` 欄位。Execution ID 只在 MCP session 有效。持久路徑是本機已確認的輸出。同一 run 再 Save 可覆寫 run 成員，Labber 每次另產新檔。保存紀錄不保證檔案仍存在，查詢不自動 stat 或補存。

`apply_writeback(tab, items=None)` 按 preview 的穩定 `target_name` 選擇當前候選。省略 items 或傳 null 寫全部，空陣列寫零項。它不看 GUI 勾選，也不回答 recipe。工具先讀 Primary 與既有 Post 的當前 preview，再檢查未知、重複及跨 stage 同名；全部檢查通過後才用當前 session ID 寫入。寫入順序是 Primary 再 Post，首錯停止，不 retry 或 rollback。收據只列 GUI-confirmed completed，並保留 skipped、failed stage、not_started 及失敗 stage 的不確定性。

摘要列出每個候選的 target、proposal 與 current，full 保留勾選旗標。Destination 摘要只列 context/project 身分，native context readiness 保留在 full。Preview 與摘要都不刷新 guard。Caller 寫入前仍須明確觀察 tab／context，核對提案與當前目的地。候選內容的調整使用 `tab.writeback_*` RPC。

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

MCP server 結束時斷線並加入其 workers。它不關閉 GUI，也不保證停止硬體。MCP 不訂閱或佇列 GUI events；完成情況透過 snapshots、operation wait 與 execution 讀取。

## 驗證

`tests/mcp/measure/` 經 public tools/session 與 recording transport 驗證 recipe、分析、catalog、連線、guard 與 operation。GUI remote/service 測試驗證 socket 與 GUI-origin 路徑。離線選集只用 fake/mock，不啟動真儀器；fixture 與測試歸屬見 `tests/README.md`。
