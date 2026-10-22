---
status: accepted
---

# ADR-0068 — 遠端前端、傳輸與 agent 介面

**狀態：** accepted。

**Last updated:** 2026-10-22，workflow async host 與 request_id

## 問題與選擇

Qt 視窗、socket client 與 MCP agent 都能接觸同一個 GUI application。若每個入口另存可提交的狀態、複製 guard，或由共用 socket 層決定實驗方法，使用者與 agent 會依入口不同而看到不同結果。Remote 是 application 的 driving adapter；它呼叫同一批 owner 的查詢與命令，不建立第二份業務真相。Measure 的 `RemoteControlAdapter` 接在共用 endpoint 上，承接 Controller 的診斷並將請求送到 application owner。GUI 的草稿與互動分析 committed session 仍由其原 owner 持有；GUI-local pointer preview 不是遠端可提交的狀態。Qt 的 screenshot、dialog 和焦點是 frontend 能力。Agent 寫入後切換 tab／pane 的行為由 measure app 的 view 命令決定，不由 socket 或通用 MCP bridge 暗中決定。共用 application 狀態與前端邊界見 [[0067]]。

Recipe 自動推導的 owner、async 協作及取消結果見 [[0075]]；MCP call log 與 ledger 的分界見 [[0063]]。本篇維護傳輸與遠端投影，不複製這些規則。

## 分層與連線

使用者在 2026-10-13 裁定：「mcp跑的腳本不要放scripts，這個資料夾是給用戶使用的。」因此 `scripts/` 只放使用者自己執行的入口。使用者選址「C」，即 repo root 的新資料夾 `mcp_servers/`，agent 的 measure MCP 組合根放在這裡。Workflow 工具由同一 measure server 提供，不保留獨立 workflow server。組合根同時知道 lab 定義與 framework，放在 lib 會反向依賴使用者套件，放在 `zcu_lab` 則混入框架啟動責任。`tools/` 只放 repo 品質檢查。Fluxdep、dispersive 無需 lab 注入，維持 lib 的 `server.py`。Repo 的三份 MCP client 設定使用同一組入口，各自保留既有啟動與曝光政策。Workflow 離線驗證使用 measure headless owner、RPC 與 MCP 的合成定義，不維護第二個 app 或獨立模擬腳本。

`gui.remote` 提供 NDJSON framing、request／reply、握手、per-client queue、backpressure 和 lazy push endpoint。`mcp.core.bridge` 持有 agent 程序的 GUI socket、連線與可選的 GUI subprocess；`mcp.core.stdio_server` 提供 stdio MCP 機制。三個 app 使用共用 GUI remote 機制，但各自註冊 wire method、驗證與 event serializer。共用 `RemoteControlServiceBase` 在 owner dispatch 統一檢查控制身份。Measure 與 fluxdep 注入同一 application authority，method 預設要求 holder，明確 readonly methods 允許 observer。控制轉移、固定 socket 身份及 GUI 停用見 [[0070]]。Measure 另外擁有 operation tracking、診斷、writeback pane mapping 和 editor 清理。Fluxdep 透過同一 Controller 提供 pipeline commands、app-owned search 與四種 interactive Session。Interactive state／PNG 是同次 committed capture 的投影；身份與 terminal receipts 由 app owner 和 wire projection 決定。Fluxdep MCP 組裝 GUI 的 RPC method，以 agent 的決策點提供手寫工具；GUI declarations 提供可重用的參數 schema，不生成逐 method 工具。RPC 回完整資料，MCP 回計數、範圍、差量及原生失效原因，決策點另呼叫 render method 附 MCP image content。Lifecycle 與 PNG integrity producer 與現存 MCP callers 共用，server exit 只 disconnect；不重試、不回滾或另建 operation registry。MCP→GUI remote consumption 由 C13 約束。Dispersive 的 remote 接口仍只讀。Measure 的 workflow namespace 呼叫同一 headless WorkflowControlFacet，提供 native run 控制、表單、records 與 PNG，不經 UI pane 或另建 Engine。共用機制接收 app 注入的 metadata 和 owner scheduler，不決定實驗或資源政策。精確封套、上限、版本與 method 說明見 [measure remote README](../../lib/zcu_tools/gui/app/measure/remote/README.md)。

2026-10-12 使用者裁決 RPC 與 MCP 的分工：「前者(MCP)面向agent，設計上要考慮agent使用負擔，後者(RPC)是控制接口，設計上要簡單基礎通用。前者透過組裝後者來兼顧雙方需求」。Fluxdep 工具直接表達階段意圖，open、context id 與 search token 只在單次組裝內使用。

Auto Align 與 search 中途沒有 agent 可影響結果的判斷，因此 MCP 輪詢到結束才回，不設整體等待上限。等待時同一 stdio server 不處理其他呼叫。GUI 仍持有 context lifetime、operation、完整 inputs、recorded quality 與 fit freshness。GUI Show 和 MCP 使用同一 fit renderer；MCP 不回過期結果的疊圖，GUI 仍顯示過期與原因。這是 fluxdep 的局部契約，不把 fit 依賴模型升為所有 app 的規則。跨 app 的 MCP 設計原則見 [[0075]] 的 MCP 設計原則。

同機跨程序的 MCP 使用 socket；同機並不等於同程序呼叫。GUI 可停用 remote socket，GUI core 不依賴 MCP。MCP `connect` 可接上現有 GUI，或經明確選項啟動 GUI；MCP server 結束只斷開連線，不關閉 GUI。

Measure `connect(script=...)` 只在 launch 新 GUI 時選擇可信 `startup(ctx)`，取代預設腳本；`clean=True` 不跑任何腳本且與 script 互斥。Attach 不重跑 startup，顯式 script 選項在 attach 時拒絕，不改既有 GUI。GUI／session／MCP 不載入 simulation environment，FTF 由一般啟動腳本使用公開 facets 安裝，沒有 `--simulate` 或 `simulation.initialize` 相容入口，見 [[0064]]。

外部 CLI／MCP workflow 擁有 agent 的啟動。GUI 不啟動 agent terminal、不管理可恢復的 agent session，也不注入 bootstrap prompt。Connection authentication 與 method exposure 是兩道不同的邊界：有 token 時 GUI 在 method 前驗證；沒有 token 的 loopback 允許本機可連線程序控制 GUI。隱藏 method 不等於授權機制，現有連線方式不承諾任意跨機部署。

## Method 與一致性

Measure GUI 的 `RemoteMethodEntry` 同時宣告 method schema、agent exposure、readonly 與 operation key。GUI 的 `rpc.catalog` 只投影非 internal method 呼叫所需的名稱、描述、參數 schema、timeout、exposure、tool 路由與 operation key，不把 holder 規則複製到 MCP。MCP 每次連線重新讀 catalog。MCP 組合根 `mcp_servers/measure.py` 將 `zcu_lab` 的 generator recipe definitions 注入 framework。共用工具固定包含分析／控制／關閉、`answer`、`apply_writeback`、`rpc_list`、`rpc_describe`、`rpc_call`，以及 `simulation_initialize`、`device_setup`、`recipe_guide`。日常量測優先 recipe，既有資料分析使用 `tab_analyze` 與 `tab_interact`，細部操作及排查使用 RPC。這是指引，不是權限模式。固定工具不從 catalog 動態生成。`rpc_call` 接受 catalog 中的 `rpc` 與 `tool` method，tool 路由只提示可用的高層工具。`internal` 不列入 catalog。`entry.info` 與 `point.list` 是公開 metadata 查詢，分別提供 opened entry 的 UUID／name／owner paths、目前 entry 的工作點清單。`entry.new/open` 明確建立／開啟 entry，成功發布 entry metadata 與 point list，不自選 point。此時 parameters=None、readiness=EMPTY，entry.info、point.list 與 setup source binding 仍可查。Run、WB 與目前 point 的 Inspect Set 要求先選取或建立 point，application 以既有 failed precondition 拒絕。SoC、device 與獨立 predictor 沿自己的 admission。不從 chip／qubit 名稱推導 entry。每個入口最終仍經 GUI 驗證，不提供任意程式碼執行。載入、cfg、保存及個別 writeback 由公開 RPC 承接。`tab.save_data` 與 `tab.save_artifacts` 同步回傳當次完整 Save report，沒有 Save handle、await 或保存後 snapshot 二讀。Raw RPC 不接管 recipe 或共用分析的後續保存。Recipe 與共用分析的 execution 由 MCP session 持有，`wait(execution)` 觀察完整接續；`cancel` 記錄停止接續的意圖，GUI cancellation 的結果另外回報。Recipe 的 `finish_early` 停止採集後繼續處理可用結果，`cancel` 則不啟動新的分析或保存。固定的 `tab_interact` 對應 GUI `tab.interact`，讀取不切焦點、command 跟隨 Analysis pane；此 method 的純讀與 command 都要求 holder。工具的個別輸入、畫面跟隨與 operation 結果見 [measure MCP README](../../lib/zcu_tools/mcp/measure/README.md)。

Setup 工具只編排既有 GUI owners。`simulation_initialize` 走現行 coordinator，切換時可能斷開真實裝置。`recipe_guide` 從 authoritative recipe registry 的 adapter mapping 讀 native guide。這些流程不改 GUI method policy，也不授權 raw RPC 隱藏讀取。

### Device setup

RPC `device.setup(name, cfg)` 只收完整 Info constructor cfg，不合併 cached 舊值。RPC 以既有 Info 的 model class 驗 required、extra、validator 和 limits；connected admission 由 DeviceService 擁有。YOKO mode 必填，value_unit 可省略，由 Info 依 mode 產生並保存，顯式矛盾拒絕。RPC 保非同步 operation handle。Raw caller 明確讀 snapshot、改完整 cfg、setup、wait，再讀 snapshot，不新增 `device.set_value` RPC。

固定 MCP `device_setup(name, cfg)` 接 partial patch。它讀 before Info，合併完整候選後送一次 RPC。Before Info 有 value_unit 欄位且 patch 含 value 時，caller 必須同時明填 value_unit，不借 before unit 或 constructor 推導代替意圖。這項規則依 Info 欄位判斷，不比對儀器名稱，FakeDevice 同樣適用。只改 output／rampstep 不需 unit。其他 model 規則仍由 RPC 驗證，不在 MCP 建第二份 validator。

工具沒有 timeout 參數，最多等待 300 秒，逾時回 running 與已知 handle，不取消，caller 用 `wait(op)` 續等。回覆保留 pre_read、start、wait、post_read、native operation 和 before／after。Requested.cfg 在 before 讀取並完成合併前為 null，之後是完整候選，不保證 model 接受。Before.info 為 null 時 pre_read failed；有 Info 但 disconnected 時沿 service 回 start failed，保完整候選與 before，after 為 null。工具不做 verification、單位換算、重連、retry 或 rollback。

握手版本以 [wire_version.py](../../lib/zcu_tools/gui/app/measure/remote/wire_version.py) 的 WIRE_VERSION／GUI_VERSION 與 [server.py](../../lib/zcu_tools/mcp/measure/server.py) 的 MCP_VERSION 宣告為準。只有 wire 版本用於相容比對，其餘是程式版本投影。

Recipe、`tab_analyze`、`wait(execution)` 與互動完成回覆使用同一 execution 摘要。MCP 在 Run 前捕捉宣告的 resolved summary facts，以它們投影條件與來源，不保留 publication 或 raw input 副本，也不以後來的 GUI 狀態代替當次事實。`status(execution, detail="full")` 讀同一已保存的 native snapshot，不新增 RPC。摘要列出 Primary/Post 結果、全部 writeback candidates 及 destination 身分。

Artifact 依 section、名稱與 members 分層，每個 member 保留當次報告的全部 format、path、status 與 error。預留目的地不代表保存成功；partial 與失敗也保留同次全部輸出。Resave 取代選中 artifact 的報告，不累積跨次成功路徑。當次 MCP image content 只在記憶體中交付，不建立 session preview 檔案或 previews 欄位。之後 tab.get_figure 取得 GUI 當下的圖，不保證是先前回覆的圖。

Execution ID 只屬於 MCP session。`run_id` 取已捕捉 Save report 的原 Run 身分，全部輸出失敗也保留同一 ID。沒有捕捉到 report 時才為 null。各 output 的 status/error 表示保存結果。Readonly tab snapshot 的 confirmed run_id 仍在尚無持久輸出成功時為 null。路徑描述本機輸出，不保證檔案仍在；同一 run 再 Save 可以覆寫 run 成員，Labber 每次另存新檔。全域 status 列非終態 executions 與終態數量，完整歷史按已知 ID 查詢。Run 或分析已送出而 receipt 未確認時保留 unknown，不從缺少 handle 推斷未啟動。這個表示層不改 GUI facts、保存、domain gate 或 writeback owner。

Recipe-owned done 回該 recipe 的後續 handoff，standalone analysis 保留原 execution。`answer(recipe, decision)` 只回答已捕捉的寫回提問，不寫入。Recipe 的 `accept_writeback` 一次交付 Primary／Post 的 captured revisions 與 IDs 給 `tab.writeback_accept`。GUI owner 在同次提交檢查全部選定 drafts，任一過期就在第一筆寫入前拒絕。`tab.writeback_preview` 回 revision，單 pane L1 `tab.writeback_write` 的 edit/write 契約不變。接受收據以 pane 與穩定 ID 記錄實際寫入和失敗前綴，full details 的 writeback_receipts 保留各次診斷，不從回答 accepted 推斷成功。2026-10-22 代裁的 captured acceptance 機制與原話見 [[0075]]。

Standalone `apply_writeback(tab, items=None)` 按穩定 `target_name` 選擇當前 Primary／Post draft，不看 GUI 勾選，也不回答 recipe。省略 items 寫全部，空序列寫零項。MCP 在任何寫入前讀完兩個 pane 的當前 preview，拒絕未知、重複及跨 stage 同名，再解析當前 session ID。這些讀取不授權寫入。寫入維持 Primary 先、Post 後，首錯停止。收據只列 GUI-confirmed writes，保留失敗 stage 的不確定性，不 retry 或 rollback。

Measure `tab.get_cfg` 回 Form authored input、ordinary leaf values 與 directed errors。`tab.edit_cfg`、`tab.reset_cfg` 與 `tab.run_start` 不收 cfg ref 或 expected revision。`tab.describe_cfg` 另回單層 metadata 與同次局部 input／value／Form preview errors，不附 lowered cfg errors。Workflows 的 `workflow.form` 不附 schema，`workflow.describe_form` 可描述同次 batch candidate，不提交 draft 或 Engine。

`tab.run_cfg_command` 是 live catalog 的通用 Form 指令入口，由既有 rpc_call 使用，不另增 MCP tool。Path 依 [[0079]] 的目標契約使用 str 欄位名／int list index，name 只選目前 active description 的公開 command，不提供任意方法執行。Remote 呼叫 resource.run_command，保留控制身份、busy、lifetime 與 mutation guard。成功回 Form input／values／errors；未知指令、inactive path 與換算失敗不留下部分寫入。Command 只修改掛載欄位與 descendants，sweep 的指令掛在 sweep row。Remote 不求值、不重建座標或採樣。Step 的 input 保留 authored 原稿，values 使用共用 Form 的有效端點投影，read 不寫回原稿。to_inline 專用入口不變。

[[0079]] 的 indexed editing 與逐元素 recipe 設計已採用，實作尚待完成。Describe 列出 list index 與元素 name，不增加名稱 path grammar。Owner 在單次 readonly 呼叫提供完整元素觀察，MCP 不以 N 次讀取拼接。Recipe 使用 participants 與 name-keyed overrides，由 owner 在 Run 前解析位置；逐葉保留 explicit／default／derived，actual、estimates 與 writeback candidates 以同次元素 name 標示。既有 Run facts capture 與 writeback owner 邊界不變。

Edit 以 model-field paths 原子寫入目前 input。`tab.refresh_sources(aliases=...)` 重新 capture，保留全部 authored input 與 explicit selections，包括 expression 原稿和既有 direct 值。`tab.reset_cfg(aliases=...)` 重新 capture 並重算 defaults。兩者的單次 aliases 都不改全域列或保存為來源覆寫。Run 只接受畫面上目前有效的 Form result，在 focus 或 operation 效果前拒絕非空 aliases。直接 RPC 只轉送一次，不隱藏 refresh 或 retry。Recipe 在建立或重用 tab 後都先 Reset，再設定參數與 Run，只有 Reset 收單次 aliases。Fixed sources、source provenance 與 acceptance 見 [[0065]]。Holder、authentication、domain busy 與硬體 guards 分別檢查，不以 cfg 版本授權。

共用 ParamSpec 同時擁有 string enum 的宣告驗證、schema 投影與 request membership。Flux plugin 從 domain FluxLineRole 提供選項，typed Action 仍做 domain 驗證。`device.snapshot.unit` 讀 cached Info 的 `value_unit`，YOKO 為 A／V，Fake 為 native。缺 Info 標 none，不按 device type 猜單位，不讀硬體。Zcu_lab flux recipe helper 比對 caller 明填 unit 與 snapshot unit，包括 native，沒有 Fake 特例；none 不是可用 flux 座標。Recipe policy 不延伸到共用 device 或 cfg 安全規則。

Measure remote 擁有分析結果的 JSON 投影。它將非有限 summary 數字換成 null，以 `invalid` 記錄原 summary 路徑與 `non_finite` 原因。MCP execution 原樣保留這份 summary 與 invalid，不解讀 fit_quality、不改寫路徑，不猜測物理原因，不改 operation outcome、保存事實或 writeback policy。有擬合時附 fit_quality 是實驗定義者的約定，框架不檢查。這項轉換不延伸到 generic context 或 framing。

認證與 wire 檢查後，MCP 以 `control.connect(label, mode)` 初始化身份，再讀 live catalog。Owner 預設自動取得未鎖定 Local，readonly 不取得。相同初始化不 reacquire，換身份須重連。`control.status/acquire/release` 的回覆及 `not_holder`、`control_locked`、`control_held`、`readonly_connection` 錯誤保留原生結構；MCP 不重分類或重試。完整 wire 初始化及管理例外見 [[0070]]。

讀取是取得當前資料，不建立 per-connection mutation baseline。Resource counters 可供來源、結果 replacement 與 provenance 使用，但不再以 seen map、guard dependencies 或 observation rollback 限制寫入。Result-bound operation ID、editor／interactive lifetime、domain busy、互斥及後續提交檢查仍由 owner 維持，生命週期見 [[0066]]。Timeout、斷線或回覆編碼失敗不證明命令沒有副作用，也不回滾 business publication。沒有明確冪等契約時不自動重送。

`context.snapshot` 從同一次公開 capture 回 native target、point、library 與有序 aliases。Alias rows 保留 authored value 與 resolved 名稱，單列解析失敗附 error；它不是歷史 Run 的參數版本。Parameter／library／alias 變更事件提示重讀這份 snapshot。既有 tab 使用 Open／Refresh／Reset 的固定來源。需明確 Refresh 才在保留 input 下取新來源，Reset 則重算 defaults；來源事件不自動刷新。Caller 使用 native snapshot，不使用舊 md／ml getters。Optional summaries 保留 dict tree 與 scalar，摘要 array／list leaves。Path 優先選取完整的單一值；這些投影不授權寫入。完整與部分讀取都使用同一固定 capture，精確投影見 remote owner README。

## Workflow 使用同一 measure owner

Workflow 工具併入 measure MCP，固定表由 assembly/context/session 組裝，不從
workflow catalog 生成。RPC 的 workflow namespace 使用 measure RemoteMethodEntry，
沿同一 live catalog、holder dispatch、timeout 與 operation mapping。MCP 不保存
第二份 Engine、draft、permission 或 defaults。C13 限制 MCP 只消費 GUI remote
契約；C16 禁止 framework 反向載入 lab 定義。Caller 在 measure 組合根注入附件。

使用者於 2026-10-21 說：「對mcp來說，……沒有模式之分，就是兩種操作，tab或workflow」。
GUI toggle 和目前 selection 不影響 RPC/MCP 的 target。`workflow_open(name)`
回 opened tab_id，同名可多開。Form 與 Start 使用 tab_id，native controls、
records 和 figures 使用明填的 run_id。Status 省略 run_id 回 catalog names 與
全部 opened tabs，不只目前 tab。確切參數與回覆見 [measure MCP](../../lib/zcu_tools/mcp/measure/README.md#workflow-tools)。

使用者於 2026-10-21 裁定：「舊的workflow gui/mcp可以移植或刪除。」因此移除
獨立 workflows app、launcher 與 MCP server。保留第二個殼會使同一 workflow
具有兩份 session 與不同的控制入口。Headless commands 由 measure facet 直接
提供，不需要舊 Window，也不經 pane 的 port 反向執行 application 命令。
Hosting 與 pane dispose 的責任見 [[0067]]。

Start 省略/null batch 使用 shared parsed draft。明填 batch，包括 []，使用
owner 當次 fresh captured defaults 加 edits。Form batch preview 則在 owner 的
call-local candidate 上讀當前 draft，不提交。Description 與 values 來自同一
candidate 和固定 capture，不下載 schema tree。Tunables Apply 的 [] 提交當前
draft，只有 native publication 成功才接受 input/base。Expression 留在 FormInput，
journal 只記普通值。Form 與 lowered cfg 的責任見 [[0065]]。

Start／Resume 回 pending operation receipt，不代表 native acceptance。
Pause／Stop 只代表控制請求，不代表 producer 已停止。Pause 保留原 operation
與 RUN lease，Resume 重用同一 handle。終局由原 owner delivery 交付。
Busy close 回 false，不暗中 Stop、drain 或 disconnect。這只是本程序互斥，
不是跨程序硬體仲裁。原生 run、device capture 與保存仍依 [[0062]]、[[0063]]。

Readonly methods 允許 observer 查 status、Form preview、records 與 figures。
Mutation 要通過 holder、busy 與 owner 的 RUN/Form 規則。Native run mismatch、
lifecycle、Form path 與拒絕保留 code/message/reason/data；unexpected producer
失敗仍是 controller_error。不因 timeout、斷線或傳輸失敗重試或回滾。

Records 直接投影 Engine facts，since 是嚴格大於的 commit_seq cursor。
Figures 使用同一 retained Plots，status 與 names 不 render。Figure 在 owner
以 shared renderer 產生 PNG，MCP 驗證 PNG 並附 image content，不在文字中保留
base64，也不建立 preview 檔。Terminal records/figures 保留到 replacement 或 close。

2026-10-22 依單一 async 腳本契約同步 workflow answer。Status 提供
pending_question 的 request_id/question_type/question/answer_schema 固定 envelope，
authored fields 只在 question 內。Engine 先比對 request_id，再驗答案 schema 與
JSON 轉換。過期 ID 或無 pending 問題拒絕並回目前 pending_request_id，狀態不變，
也不記 accepted answer。Workflow answer 回原 operation_id，MCP 映射成 Start 的
同一 pending handle。

共用 async host、answer Future 與 control 序列依 [[0075]] 的「Async 協作與控制命令」。
整個 coroutine 只佔一個 worker，Resume／Answer／Stop 不另排 execution 或 drain segment。
RPC/UI owner thread 不執行作者 coroutine。控制 I/O 失敗保留 cause，原 worker 完成
finally 與 executor drain 後才交終局。原 owner terminal delivery 才釋放 operation
與 RUN lease，維持 [[0066]] 的 settlement。狀態仍以公開 reads／polling 觀察，
此變更不承諾 native lifecycle push 或 subscribe。

## 事件與錯誤

Application state 和明確讀取是資料來源。EventBus 產生同步的 domain facts 與 `EventMeta(seq, origin)`；dispatch 宣告 origin，operation 跨執行緒顯式攜帶原發起者與 operation id。Seq 表示同一 process 的順序，不是可靠交付證明；過濾或重啟後的跳號不能直接判定遺失。Wire 的 EventBus push 加上 seq 與不含 client id 的 origin，保留原 event payload；subscriber 可自行收合呈現，不能改變必要的 domain commit 順序。Endpoint 只有找到 matching subscriber 才建立、編碼 payload，並在 enqueue 前重驗原 recipients 的訂閱和連線狀態。完成 unsubscribe／disconnect 後不會收到遲到的 push。診斷由 Controller 直接 fan-out，不經 EventBus 訂閱；這不表示 socket 故障時仍能交付。Measure MCP 不訂閱業務 push，而以讀取與 operation wait 接手進度；重連重新讀 snapshot，沒有 replay 承諾。高頻 plot／progress 不預設推送給每個 client。

啟動腳本的 owner commands 沿既有 EventOrigin 歸因，operation 保留原發起者直到 terminal delivery。Measure 僅略過 script-origin device operation 的 terminal dialog，terminal fact、failed handle 與 log 都照常交付。預設腳本逐項隔離失敗，最後顯示狀態列摘要。使用者手動 GUI device 錯誤仍顯示對話框，不以「啟動中」全域抑制通知。未捕捉的 startup 錯誤仍顯示含腳本路徑／行號的錯誤對話框，保留已完成部分與 callbacks，見 [[0064]]。

可由 caller 修正的失敗由 producer 用 remote-independent `ExpectedError` 類別顯式分類。共用 dispatch 只翻譯已標記的 invalid input 與 failed precondition；handler 直接提出的 `RemoteError` 保留原本的結構化資料。它不從 `RuntimeError`、訊息字串或 reason 前綴猜測分類。Unexpected failure 留下診斷。`wait` 呼叫本身失敗與 operation outcome 為 failed 是兩件事；timeout 亦不是安全重試的許可。精確 error code 與 reason 見 [measure remote README](../../lib/zcu_tools/gui/app/measure/remote/README.md)。

## 代價與相關文件

共用 transport 讓三個 app 不必維護四套 socket 與 framing，但 app 仍要各自維護 method 和 domain policy。固定工具與 live catalog 分開維護呈現與可達 method；這比只生成一份動態工具表多一層入口，換來常用判斷點的明確工具與低頻能力的即時發現。Wire 支援 push 不強迫 LLM client 訂閱，也不承諾 replay。Cfg 使用邊界見 [[0065]]，operation、cancel 與 shutdown 見 [[0066]]，GUI 程序組裝見 [[0064]]。
