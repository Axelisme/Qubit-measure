# ADR-0068 — 遠端前端、傳輸與 agent 介面

**狀態：** accepted。

## 問題與選擇

Qt 視窗、socket client 與 MCP agent 都能接觸同一個 GUI application。若每個入口另存可提交的狀態、複製 guard，或由共用 socket 層決定實驗方法，使用者與 agent 會依入口不同而看到不同結果。Remote 是 application 的 driving adapter；它呼叫同一批 owner 的查詢與命令，不建立第二份業務真相。Measure 的 `RemoteControlAdapter` 接在共用 endpoint 上，承接 Controller 的診斷並將請求送到 application owner。GUI 的草稿與互動分析 committed session 仍由其原 owner 持有；GUI-local pointer preview 不是遠端可提交的狀態。Qt 的 screenshot、dialog 和焦點是 frontend 能力。Agent 寫入後切換 tab／pane 的行為由 measure app 的 view 命令決定，不由 socket 或通用 MCP bridge 暗中決定。共用 application 狀態與前端邊界見 [[0067]]。

## 分層與連線

`gui.remote` 提供 NDJSON framing、request／reply、握手、per-client queue、backpressure 和 lazy push endpoint。`mcp.core.bridge` 持有 agent 程序的 GUI socket、連線與可選的 GUI subprocess；`mcp.core.stdio_server` 提供 stdio MCP 機制。四個 app 使用共用 GUI remote 機制，但各自註冊 wire method、驗證與 event serializer。Measure 還擁有版本 guard、operation tracking、診斷和 editor 清理；fluxdep、dispersive 與 autofluxdep 的 remote 接口只讀。共用機制接收 app 注入的 metadata 和 owner scheduler，並不持有實驗或資源政策。精確封套、上限、版本與 method 說明見 [measure remote README](../../lib/zcu_tools/gui/app/measure/remote/README.md)。

同機跨程序的 MCP 使用 socket；同機並不等於同程序呼叫。GUI 可停用 remote socket，GUI core 不依賴 MCP。MCP `connect` 可接上現有 GUI，或經明確選項啟動 GUI；MCP server 結束只斷開連線，不關閉 GUI。外部 CLI／MCP workflow 擁有 agent 的啟動。GUI 不啟動 agent terminal、不管理可恢復的 agent session，也不注入 bootstrap prompt。Connection authentication 與 method exposure 是兩道不同的邊界：有 token 時 GUI 在 method 前驗證；沒有 token 的 loopback 允許本機可連線程序控制 GUI。隱藏 method 不等於授權機制，現有連線方式不承諾任意跨機部署。

## Method 與一致性

Measure GUI 的 `RemoteMethodEntry` 同時宣告 method schema、agent exposure、guard dependencies、成功讀取所揭露的資源及 operation key。GUI 的 `rpc.catalog` 只投影非 internal method 呼叫所需的名稱、描述、參數 schema、timeout、exposure、tool 路由與 operation key，不把 guard／reveal policy 複製到 MCP。MCP 每次連線重新讀 catalog。MCP 組合根 `scripts/run_measure_mcp.py` 將 `zcu_lab` 的 generator recipe definitions 注入 framework。共用工具固定包含分析／控制／關閉、`answer`、`apply_writeback`、`rpc_list`、`rpc_describe`、`rpc_call`，以及 `simulation_initialize`、`device_set_value`、`recipe_guide`。日常量測優先 recipe，既有資料分析使用 `tab_analyze` 與 `tab_interact`，細部操作及排查使用 RPC。這是指引，不是權限模式。固定工具不從 catalog 動態生成。`rpc_call` 接受 catalog 中的 `rpc` 與 `tool` method，tool 路由只提示可用的高層工具。`internal` 不列入 catalog。`project.info` 與 `context.labels` 是公開查詢，分別提供已套用 project 的身份及目錄、全部 context labels。每個入口最終仍經 GUI 驗證，不提供任意程式碼執行。載入、cfg、保存及個別 writeback 由公開 RPC 承接。`tab.save_artifacts` 回傳的 MCP handle 由 `wait(op)` 觀察，但 raw RPC 不接管 recipe 或共用分析的後續保存。Recipe 與共用分析的 execution 由 MCP session 持有，`wait(execution)` 觀察完整接續；`cancel` 記錄停止接續的意圖，GUI cancellation 的結果另外回報。Recipe 的 `finish_early` 停止採集後繼續處理可用結果，`cancel` 則不啟動新的分析或保存。固定的 `tab_interact` 對應 GUI `tab.interact`，讀取不切焦點、command 跟隨 Analysis pane；此 method 不用 seen guard。工具的個別輸入、畫面跟隨與 operation 結果見 [measure MCP README](../../lib/zcu_tools/mcp/measure/README.md)。

Setup 工具只編排既有 GUI owners。`simulation_initialize` 走現行 coordinator，切換時可能斷開真實裝置。`device_set_value` 宣告完整 pre-read，只送 value，不改 output 或 rampstep。Unit 不換算。兩者保留 native operation、階段與前後 snapshot。等待逾時不取消，工具不自動重連、retry 或 rollback。`recipe_guide` 從 authoritative recipe registry 的 adapter mapping 讀 native guide。這些流程不改 GUI method policy，也不授權 raw RPC 隱藏讀取。

Recipe、`tab_analyze`、`wait(execution)` 與互動完成回覆使用同一 execution 摘要。MCP 在 Run 前捕捉 resolved publication，以它投影條件與來源，不以後來的 GUI 狀態代替當次事實。`status(execution, detail="full")` 讀同一已保存的 native snapshot，不新增 RPC 或 guard 觀察。摘要列出 Primary/Post 結果、全部 writeback candidates 及 destination 身分。

Artifact 依 section、名稱與 members 分層，每個 member 保留完整路徑及保存狀態。預留目的地不代表保存成功，後續失敗也不清掉已保存的前綴。Session preview 路徑獨立放在 `previews.run/primary/post`，不列為持久 artifact。

Execution ID 只屬於 MCP session，`run_id` 目前為 null。全域 status 列非終態 executions 與終態數量，完整歷史按已知 ID 查詢。Run 或分析已送出而 receipt 未確認時保留 unknown，不從缺少 handle 推斷未啟動。這個表示層不改 GUI facts、保存、guard 或 writeback owner。

Recipe-owned done 回該 recipe 的後續 handoff，standalone analysis 保留原 execution。`answer(recipe, decision)` 只回答已 capture 的寫回提問，普通 `tab.accept(items)` 才寫當前 draft。Recipe 摘要的 writeback.receipts 保留實際寫入與失敗前綴，不從回答 accepted 推斷成功。

Standalone `apply_writeback(tab, items=None)` 按穩定 `target_name` 選擇當前 Primary／Post draft，不看 GUI 勾選，也不回答 recipe。省略 items 寫全部，空序列寫零項。MCP 在任何寫入前讀完兩個 pane 的當前 preview，拒絕未知、重複及跨 stage 同名，再解析當前 session ID。這些讀取不刷新 guard。寫入維持 Primary 先、Post 後，首錯停止。收據只列 GUI-confirmed writes，保留失敗 stage 的不確定性，不 retry 或 rollback。

Measure tab cfg 使用明示的 `CfgRef`，不另加 per-connection cfg seen。`tab.get_cfg` 回完整 publication，`tab.edit_cfg`、`tab.reset_cfg` 與 `tab.run_start` 必須帶 caller 觀察到的 identity／revision。Reset 由既有 cfg resource 取得目前 adapter defaults，不複製預設值或新增 unsaved guard。另一條連線讀到的 ref 也可用，但不能代替 authentication 或請求連線的 tab、SoC、device guards。Stale 回 expected／actual，不換成最新版本。直接 RPC 保留完整 publication，edit／reset／Run 只轉送 supplied ref 一次，不隱藏重讀、refresh 或 retry。Recipe 在其已宣告流程內觀察 cfg，重用 tab 時 reset，再套本次參數與 Run；這不授權 RPC 自動修補 stale。Cfg source publication 與固定 Run acceptance 見 [[0065]]。

共用 ParamSpec 同時擁有 string enum 的宣告驗證、schema 投影與 request membership。Flux plugin 從 domain FluxLineRole 提供選項，typed Action 仍做 domain 驗證。MCP flux recipe 的 native 座標只允許已確認的 FakeDevice、unit=none 與 caller 明確 opt-in，不延伸到共用 device 或 cfg 安全規則。

Measure remote 擁有分析結果的 JSON 投影。它將非有限 summary 數字換成 null，以 `invalid` 記錄原 summary 路徑與 `non_finite` 原因。MCP execution 保存這份已讀投影，不猜測物理原因，不改 operation outcome、保存事實或 writeback policy。這項轉換不延伸到 generic context 或 framing。

其他 guarded resources 仍由 GUI owner 在自己的序列中比較每條連線的 seen map。成功的完整讀取才建立宣告的觀察；部分讀取、失敗與回覆編碼失敗不建立新觀察，版本零也不能代替未曾觀察。成功寫入只推進先前看過且版本相符的資源。建立新 tab 的例外只認證存在性，`tab.new` 與 `tab.open_file` 成功回傳 tab ID，且存在版本由 0 變 1 時，GUI 將該資源記入該連線的 seen。Result 等其他 guarded resources 仍須明確完整讀取。Load、library commit 與 writeback 仍保留各自的 context guard。

MCP 不保存第二份 seen，也不用 wire expected-version table 或隱藏預讀解鎖。其他資源的 Stale 須重讀對應完整 snapshot，再由 caller 決定是否重試；event origin 不能授權寫入。Guard 不替代 operation snapshot、互斥及後續提交檢查，生命週期歸 [[0066]]。Timeout、斷線或回覆編碼失敗不證明命令沒有副作用。Reply failure 可撤回該回覆建立的 seen observation，但不回滾 business publication。沒有明確冪等契約時不自動重送。

## 事件與錯誤

Application state 和明確讀取是資料來源。EventBus 產生同步的 domain facts 與 `EventMeta(seq, origin)`；dispatch 宣告 origin，operation 跨執行緒顯式攜帶原發起者與 operation id。Seq 表示同一 process 的順序，不是可靠交付證明；過濾或重啟後的跳號不能直接判定遺失。Wire 的 EventBus push 加上 seq 與不含 client id 的 origin，保留原 event payload；subscriber 可自行收合呈現，不能改變必要的 domain commit 順序。Endpoint 只有找到 matching subscriber 才建立、編碼 payload，並在 enqueue 前重驗原 recipients 的訂閱和連線狀態。完成 unsubscribe／disconnect 後不會收到遲到的 push。診斷由 Controller 直接 fan-out，不經 EventBus 訂閱；這不表示 socket 故障時仍能交付。Measure MCP 不訂閱業務 push，而以讀取與 operation wait 接手進度；重連重新讀 snapshot，沒有 replay 承諾。高頻 plot／progress 不預設推送給每個 client。

可由 caller 修正的失敗由 producer 用 remote-independent `ExpectedError` 類別顯式分類。共用 dispatch 只翻譯已標記的 invalid input 與 failed precondition；handler 直接提出的 `RemoteError` 保留原本的結構化資料。它不從 `RuntimeError`、訊息字串或 reason 前綴猜測分類。Unexpected failure 留下診斷。`wait` 呼叫本身失敗與 operation outcome 為 failed 是兩件事；timeout 亦不是安全重試的許可。精確 error code 與 reason 見 [measure remote README](../../lib/zcu_tools/gui/app/measure/remote/README.md)。

## 代價與相關文件

共用 transport 讓四個 app 不必維護四套 socket 與 framing，但 app 仍要各自維護 method 和 domain policy。固定工具與 live catalog 分開維護呈現與可達 method；這比只生成一份動態工具表多一層入口，換來常用判斷點的明確工具與低頻能力的即時發現。Wire 支援 push 不強迫 LLM client 訂閱，也不承諾 replay。Cfg 使用邊界見 [[0065]]，operation、cancel 與 shutdown 見 [[0066]]，GUI 程序組裝見 [[0064]]。
