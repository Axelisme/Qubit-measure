---
status: accepted
---

# ADR-0075 — Agent 介面分層

**狀態：** accepted。使用者於 2026-10-16 以「都採用建議」確認歸檔。本篇記錄入口分層、recipe 與 workflow 的共用 async host、Ask 及取消契約。不新增工具、不改 method policy，也不授權硬體操作。
**關聯：** [[0068]] 的 remote 與 measure 固定工具、[[0062]] 的 workflow run、[[0063]] 的保存權威、[[0069]] 的核心與前端；控制權見 [[0070]]，workflow engine 見 [[0062]]。

## 問題

Measure、workflows、fluxdep 與 dispersive 的 MCP 各自提供工具，[[0068]] 也寫了「日常量測優先 recipe、排查使用 RPC」的指引。這些入口位於哪一層、agent 應該在哪裡介入、層與層如何交接參數，目前分散在各篇 ADR 和 module README，沒有一份文件寫在同一處。缺少這份定位時，新增工具容易放錯層，agent 也不知道從哪一層開始。

## 決策

2026-10-12 使用者裁決 MCP 的定位：「提供agent從底層到抽象層的方法，最底層是實驗環境與硬體操作，這對用戶與agent都不可直接控制，上一層是gui/notebook層級控制，包含儀器控制，實驗建立，完整cfg控制，實驗參數控制等。再上一層是recipe，將常用實驗包裝層一個參數量小的自動gui操作，其餘參數用自動推導的方式，只在需要agent時yield給agent，這樣正常使用agent只需要傳入真的需要其判斷的資訊即可。再上一層是workflow gui/mcp，包裝常用實驗流程進一次實驗執行，比如overnight實驗，或是t1 overflux實驗，只需要要一次設定，就能完成跨多組實驗的流程。同層的還有跨實驗整合分析層，比如fluxdep_fit gui/mcp，支援跨多次量測數據甚至跨多qubit的數據整合分析。」

同日使用者裁定 L1 與 L2 的差別：「L1中，每個實驗更像是一組多部流程，agent可以在中間介入。L2中實驗就是一個整體，agent只能控制輸入與接受輸出。」隨後補充：「L2不是完全無法介入中間，而是中間明確需要agent時使用yield來與agent協作。」

同日使用者裁定 L3 與自動推導的歸屬：「L3的workflow也是yield來協作。」「自動推導規則會由workflow/recipe內部，或是gui adapter的初始參數生成負責。」

### 四層

| 層 | 內容 | Agent 入口 | Agent 的控制粒度 |
| --- | --- | --- | --- |
| L0 實驗環境與硬體 | SoC、儀器 driver、program 執行 | 沒有直接入口，只能經由 L1 的 owner | 不適用 |
| L1 GUI／Notebook | 儀器控制、實驗建立、完整 cfg、實驗參數、單次分析與保存 | Measure 的 `rpc_list`／`rpc_describe`／`rpc_call`、`device_setup`、`simulation_initialize`、`tab_analyze`、`tab_interact`、`apply_writeback` | 一次實驗是多步流程，agent 可以在任兩步之間介入 |
| L2 Recipe | 操作 GUI tab 的腳本，可串多個 tab 與實驗 | Measure 的各 recipe 工具，搭配 `recipe_guide`、`status`、`wait`、`answer`、`cancel(scope)` | Agent 控制輸入、接受輸出，中途只在 recipe 明確提問或互動交接時協作 |
| L3 Workflow | 一次設定，執行跨多組實驗的流程，例如 overnight、T1 over flux | Measure MCP 的 workflow 工具與 answer | Workflow 以 Ask 決定提問位置；agent 另可用 Pause／Resume／Stop 與 tunables 介入 |
| L3 跨實驗整合分析 | 整合多次量測的資料做聯合分析，例如 flux-dependence fit | Fluxdep MCP 的決策點工具；dispersive MCP 只讀 | 分析步驟由 agent 驅動，順序不固定；agent 的判斷落在宣告的決策點，程式能判斷的事不交給 agent |

Notebook 是使用者在 L1 的入口，agent 目前不經 Notebook 操作。

### Agent 介入點

正常使用時，agent 只提供真正需要它判斷的資訊。自動推導得出的參數由 recipe 或 workflow 補齊，agent 不必逐一傳入。Recipe 與 workflow 在作者明填的 Ask 處停下，把問題與 answer schema 交給 agent。

L1 與 L2 的差別在於誰決定介入點。L1 由 agent 決定，它可以在任兩步之間停下、讀取或修改。L2 由 recipe 決定，只在明確需要 agent 的位置交還控制權，其餘步驟自動執行。交接可以落在輸入端（缺少必要參數）、中途（互動分析交接 tab，agent 以 `tab_interact` 完成選取）或輸出端（寫回確認）。Agent 需要在 recipe 沒有交接的位置調整步驟時，改用 L1。

Recipe 作者以 `await tab.run()` 與 `await tab.analyze(stage, result=T)` 等待工作結果，不表示每一步都要 agent 回答。互動分析以 `interactive=True` 明確交接 GUI tab。`await tab.ask_writeback()` 或 `await session.ask(question)` 才是等待 agent 或使用者回答的提問。回答本身不寫入資料，作者接受寫回時另呼叫同步 `tab.accept_writeback(items)`。取消不回傳 cancelled 分支給作者，作者以 `finally` 保存已有的 Run，終態與 runs 由 runtime 回報。

L3 workflow 由作者決定在哪裡交還控制權。一次執行跨越多次實驗，交接可出現在實驗之間或之內。`await env.run(...)` 等待 Engine 執行工作，`await env.ask(question)` 才是交給 agent 的提問。

2026-10-21 使用者修正 L2 與 L3 的定位與協作模型。以下保留當時的協定術語，現行 async 契約見下節。推導見 [Recipe 接口重設計](../discussions/2026-10-21-recipe-interface-redesign.md)、[三角色協定](../discussions/2026-10-21-three-role-protocol.md)、[Workflow 定位與承載](../discussions/2026-10-21-workflow-positioning-and-hosting.md)。

- Recipe 是操作 GUI tab 的腳本，可開多個 tab、串多個實驗；「參數少、單次實驗」不再是定位。原話：「recipe是一組操作gui的定位，其實不只能跑一次實驗，是可以做到類似workflow的行為的」。Recipe 與 workflow 的區別在執行基底（GUI tab 對 Engine）與誰持有迴圈，兩者都是腳本，輸入以 pydantic `FormModel` 宣告。Recipe inputs 只驗證 pydantic 與 MCP JSON schema，不渲染 GUI Form；workflow 的表單由其宿主提供。原話：「該統一的是使用者與作者的認知與慣例」。
- 三角色協定：agent、腳本、runtime。腳本等 runtime 是 `Wait(op)`，腳本等 agent 是 `Ask(question)`，agent 主動通知腳本是 `Signal(kind, payload)`。Recipe 驅動器與 workflow Engine 底層共用這份協定，`tab.run()`、`env.run()` 等都是糖。原話：「我心智模型是三個角色：agent/recipe/runtime，recipe等agent就是Ask，recipe等runtime就是Wait。agent主動送訊息給recipe就是signal之類的」；「Engine底層應該也要用Ask/Wait模式，其他都是糖」。
- Recipe 執行分兩層：驅動引擎與實驗語義解耦，只處理 yield 流程；便利層提供 `run`／`analyze`／`ask_writeback` 等語法。原話：「recipe執行要分兩層，一層是驅動引擎，與實驗語義解耦，只負責處理yield的流程。另外一層利用前者，提供run/analysis/ask_writeback等便利語法」。Recipe inputs 的欄位由作者定義意義，框架不設帶 overrides 的基底。原話：「overrides不該出現在base class中，超過負責的範疇了」。
- Workflow 由 measure-gui 承載（同程序、同 session、同 RUN lease、同 MCP server），步驟不經 measure tab 執行。原話：「我也傾向B+D」（B 嵌入 measure-gui）。

以上保留 2026-10-21 的討論原話。當時的 `Wait`／`Ask`／`Signal` 與 yield 描述已由下節的 async host 契約取代。Recipe 的公開契約見 [measure MCP README](../../lib/zcu_tools/mcp/measure/README.md#recipe-與-execution)，workflow 的作者面見 [workflow README](../../lib/zcu_tools/experiment/workflows/README.md)。

跨實驗整合分析與 workflow 同屬 L3，但不由 workflow 決定 yield。Fluxdep 是獨立、可選的分析 app，不併入 measure 或 workflows。Agent 在資料可用性、對齊、採樣、躍遷假設、驗收與剔除六個決策點介入，各步可以重做。程式處理前置條件、計算與發布；agent 判讀證據並選下一步。人與 agent 共駕同一份 application state，寫入由 [[0070]] 的 holder 控制。

上層不重做下層的工作。Recipe 經由 L1 的 owner 操作 GUI。Workflow 經實驗 runtime 執行並自行保存，分工見 [[0062]]。跨實驗分析讀取下層已保存的資料。

### Async 協作與控制命令

2026-10-22 使用者裁定作者面改為 `async def`／`await`：「我之所以多次嘗試將generator換成async/await，是因為我認為作者應該比較熟悉原生python語法，而不是框架自己定義的語法，前者上手應該會比較容易」。這取代舊的 yield from 作者介面。Asyncio 在 caller 指定的 worker 排程，不改 GUI scheduler、operation owner、RUN lease 或 drain 邊界。Host 從不對作者 task 呼叫 cancel。

`experiment.script_host.ScriptHost` 共用於 recipe 與 workflow。Caller 以 catalog 實名建立根 Scope 並交給 host，`host.root` 就是該物件。

- `run(coroutine)` 在 caller worker 執行一次作者 coroutine，等 finally 與 executor drain 後返回或 raise。
- `await call(fn, *, scope=None, name=None, on_completed=None)` 在 executor 執行 fn。未填 scope 時，operation child 掛根；明填時掛該顯式 parent。回傳 CallResult 的 epoch、partial_result、stopped 是 fn 完成時的事實。
- on_completed 是 framework recording hook。fn 正常返回後，host 在作者 worker、operation 尚存活時呼叫一次 hook，再檢查根取消。Hook 同步、不 await、不重入；fn 或 hook 的真實例外原樣傳播。這讓 journal 與 recipe Run facts 在根取消時仍保留完成資料。
- `await ask(question)` 回具名 Answer，公開唯一 request id。錯誤、遲到或重複答案不消耗問題。根取消使 await raise ScriptCancelled，不向作者交付 Optional。
- `control(fn)` 與 cancel／pause／resume／answer 共用 loop 控制序列。Fn 只做 memory-only、數微秒級的操作，不 await、不阻塞、不 I/O。
- `close()` 請求根停止，off-worker 等待 coroutine、finally 與 executor drain，不強制終止 task。

作者 await 的 AST 規則統一套用到 `zcu_lab/` 所有 async def，不識別 decorator、不分 entry 與 helper，也不做可達性分析。只允許 `env.*`／`tab.*`／`session.*` 呼叫，或靜態解析到 lab async 定義的 helper 呼叫。Helper 可以呼叫另一個 helper，自己的本文也受檢查。工具拒絕 `asyncio.*`、`create_task`、`gather`、`wait_for`、`TaskGroup`，明列原語的直接 Call 不論有無 await 都拒絕，包括 `async with TaskGroup()`。其他 async with／async for 不另禁。`tools/check_script_awaits.py` 接入 quality gate，不是完整 type checker，別名、動態分派與包裝仍可能繞過規則。

#### 顯式 Scope 與取消出口

使用者於 2026-10-22 裁定：「signal scope應該搭配有作用域保證的with以及for來規範scope範圍」「你認為signal是不是應該是顯式的？」「可以考慮讓scope成為zcu_lib/core的契約」「it.commit_dir可以對應到scope的層級與深度」。Scope 由 experiment 定義，包裝 StopSignal。底層 driver、clock、program builder 只收 `scope.stop`，不認識 host。

根 kind 限 workflow／recipe／run。Workflow 與 recipe 根的 name 由 host 使用 catalog 實名顯式注入，不做 fallback。非 None 的 name 必須非空，不含 #、/、:。框架用 `branch(kind, name=...)` 建 iteration／operation／schedule／acquire；作者用 `child(label)` 建自訂 scope，label 不得含分隔符或使用保留 kind。With 或顯式 release 管理壽命，release 後不得再建 child。

作者的 env 就是 workflow 根。每個 child 提供相同的 run／call／setup_device／wait_until／dir／commit 能力。作者用 `with env.scope(label) as scope` 與 `for item, iteration in zip(items, env.iterations())` 明填 parent，沒有隱式 current scope。

Workflow 根取消且 expected=None 時，Stop owner lock 先寫 StopRequested／manifest 並 latch，再由 host.cancel 原子設 root。持久化不進 control callback。此分支的 cancelled=True 表示 owner 已接受並記錄 Stop；host 若仍活則同時設 token，不保證 drain。Host 已退出但 owner 尚未 finalization 時，接受 Stop 仍可使終態為 cancelled，不以 host.cancel 的 refusal 推翻 owner 的接受。其他 selector 直接由 host.cancel 核對，不寫 Stop 意圖。Terminal workflow 對合法請求回 typed scope-not-active refusal，錯誤語法仍 fail fast。持久化失敗保留 failed 與原始 cause，不偽裝成 accepted。

Host 首次根取消在自己的 owner lock 內 latch bit 並記 typed ack，不等待 loop 通知。重複取消回第一次 ack，不再排 loop 工作。這避免作者 finally 阻塞 loop 時，控制端也被迫等待。Ack 表示 token 已設，不表示工作已 drain。

Recipe runtime 用公開 control(fn) 在同一 loop 回合檢查與停止當前 Run token，避免 check／ack race。Analysis、Ask 或閒置時的 run cancel 回 cancelled=False、reason="no run in progress"，不停止下一個 operation。

Iteration generator 的 next 釋放前一趟，close 釋放最後一趟。Host 不建隱式 iteration，也不以 commit 更換 iteration。Recipe 無作者 iteration，路徑直接是 `recipe:<name>/operation#N`。

Commit 是同步 snapshot 寫檔，不經 pause gate。它不啟動硬體工作，取消收尾也必須能存已有成果。Async operation／ask 才經 pause gate。Iteration.cancelled 只退出目前 point，主 for 用 continue，point helper 用 return。Point 的 try/finally 同步 commit 一次，不在 finally await，根 ScriptCancelled 繼續往上傳播。

- 根 workflow／recipe 被取消表示腳本本身停止。下一個 call／ask await raise 普通 Exception 子類 ScriptCancelled，不是 asyncio.CancelledError。進行中的 call 先合作 drain、保留 completion facts，再 raise。真正的 fn、recording hook 或 cleanup 例外優先傳播。
- 根以下的取消表示等待中的工作被打斷。Acquire 只停一次取得；schedule 停該段後續取得；iteration 跳過目前這一點。後續 sibling 不繼承已釋放 child 的停止。Workflow 交付 Completed(partial_result=True) 或 Failed，由作者決定繼續、返回或重試。
- run 是 operation 的選擇器別名。Recipe 只接受進行中的 tab.run；analysis／Ask／閒置時回 no run in progress。Operation 可選本地 analysis；其無結果的合作取消回 AnalysisResult(status="failed")，不冒充根取消。
- 根取消後 finally 仍在原 worker 執行，同步 Save 保持可用。Recipe 終態為 cancelled，report=None，runs 保留 run_id、partial 與每次 Save 收據。ScriptCancelled 不記入 Host.error 或 recipe error；真實錯誤仍保留診斷。

StopSignal 的 set 只停止自己與子孫。Set_error 只記本 scope 第一個錯誤並向下停止，不改祖先 event／error。Release 向上交付 partial／error 事實，operation boundary 截止，不污染後續 sibling。GUI 單次 run 根視為 operation boundary。GUI result_state.partial_result 與 metadata stopped 讀 Run root 的 partial_result，cancel_requested 只記使用者整次 Stop 意圖。

Pause 只停止目前 operation、增加 epoch、關 gate，不停止 iteration。Resume 開 gate。Workflow 丟棄 Pause 造成的 stale stopped attempt，重跑同 request；普通 scope cancel 直接交值。自然完成也等 gate 後交付，不重跑。主動命令仍限控制命令與步驟邊界的 tunables，不支援 Run 中改科學參數或同時等待多來源。

#### 路徑、選擇器與跨 process 投影

路徑由父子鏈計算。根為 `workflow:<name>`／`recipe:<name>`，GUI 根為 run。Child 段為 kind:name#N，無 name 時省略冒號與 name；作者 label 為 label#N。序號按同 parent 的 kind 從 1 遞增。Operation name 只能由 caller 提供，不從 callable 反射。

Status 的 active_path 是根到第一個有多個存活 child 的節點的單一鏈。Live_paths 是所有存活葉完整路徑的 list[str]，不是逗號字串。沒有 fork 時 active_path 等於唯一葉路徑。GUI workflow pane 顯示這兩項原生投影。

Cancel 搜整棵存活子樹，接受 kind、kind#N、operation:<name> 與 live_paths 中的完整路徑精確比對。多筆命中回 cancelled=False、reason="ambiguous" 與候選完整路徑 candidates。同層同名 operation 必須填 #N；跨 parent 的 kind#N 仍多筆就繼續 ambiguous。不選 latest child。Expected 只接受正整數，拒絕 bool，與 #N 同填須一致。合法但不存活的 selector 回 scope not active，序號不符回 sequence mismatch，均不設 token。Root 無序號。

Recipe 本地樹在 MCP process，GUI Run 是另一棵樹。Schedule／acquire 與 composite 完整路徑經 operation.cancel_scope 代理至目前 tab.run 的 handle。Operation.status_scope 回 GUI active_path 與 live_paths。Recipe 省略 GUI run 根，把尾段接到明確的本地 Run operation，並以同一規則投影 cancel ack 的 active_path／candidates。GUI 的 cancelled／reason 原值保留。

兩邊只在自己的控制序列或樹鎖下核對，不宣稱跨 process 原子。細層 op target 必須仍為目前 tab.run handle，舊 op 回 scope not active，不經歷史歸屬改打新 Run。Execution target 的 expected 只檢查目前 GUI root，下一 Run 可重用序號；跨 Run 精確取消使用觀察到的 op handle。Acknowledgement 只證明已處理控制命令，不證明 operation 或 cleanup 已 drain。

#### 擬合失敗以值交付

AnalysisFailed 定義於 Qt-free 的 analysis 葉模組，experiment 公開同一型別。這個依賴方向遵守 analysis 位於 experiment 下層的 import contract，不讓 fitting 反向 import experiment。

只有 analyze work 的 AnalysisFailed 產生 OperationOutcome.failure_kind="analysis"。Record、writeback、postprocessing 與任意程式錯誤不降級。Operation.await wire 保留分類；recipe 對 analysis 分類回 AnalysisResult(result=None, status="failed", reason=...)。作者決定保存、跳過、重試或停止。成功 summary 仍按作者型別驗證，缺欄位不補值。

Fitting owner 的 fit_func 把 optimizer 不收斂與已知自由參數數目下的資料不足轉為 AnalysisFailed，不再回初猜加 warning。Model callback 自己的例外原樣傳播。其餘 backend 的有效性或 fallback 契約不因此全改，已分類路徑見 fitting README。這些 frontend→experiment／analysis 依賴受既有 import contracts 約束。

### 層與層之間以共享狀態交接

各層之間不靠 agent 搬運數值。下層寫入、上層讀取的是同一份共享狀態：

- 工作點的參數與 module library，由 writeback 更新。
- Ledger 與 Experiment Data File，保存權威見 [[0063]]。
- `params.json` 的 typed handoff，例如 fluxdep 寫入、dispersive 讀取的 fit 結果。

Agent 可以讀這些狀態來判斷，但後續實驗取用參數時，以共享狀態為準，不以 agent 對話中記住的數值為準。

### Captured writeback acceptance

Recipe 的問答捕捉候選，接受時以穩定 item IDs 寫入 owner 當下 draft 的值與目的地。
GUI／MCP 採所有權模型。問答期間的 draft 編輯或替換是 holder 的意圖，不觸發 stale 拒絕。
Preview 不回 revision，`tab.writeback_accept` 不帶 `expected_revision`，也不還原捕捉時的 proposal。
Recipe 將全部 Primary／Post 選定 IDs 一次交給這個 RPC。GUI owner 在同一次提交檢查全部 IDs，
unknown／duplicate 在第一筆寫入前拒絕。通過後按 Primary、Post、各 pane 的候選順序寫入，
首錯停止，收據保留成功前綴與診斷，不 retry／rollback。
單 pane L1 `writeback_set`／`writeback_write` 繼續編輯並寫入當前 draft。

收據用 owner item ID 關聯確認項目，不將 authored alias 名稱與 concrete target 比較。
`WritebackReceipt.items` 是成功 ID 對應的原捕捉候選，不是實際寫入值快照。
接受回覆以成功 IDs 關聯原捕捉候選。實際值與目的地取提交當下的 owner draft。
2026-10-23 補充：同步寫入前由 owner 配置一次性 submission_id，完成項另存 ID、值、固定目的地。
Caller 先 begin，再帶同一 ID 提交；begin 不表示已寫入，同 ID 重複提交拒絕。
逾時回覆帶 ID 並要求先查狀態，再決定是否重送，不把 10 秒 handler 逾時當成寫入失敗。
唯讀 receipt 查詢保留成功前綴，不重讀當前 draft 或目的地。
實際值由每筆寫入 owner 回傳並由 executor 保留，GUI 只關聯 item ID、pane 與固定目的地。
不能在整批後 readback，因為後續父／子路徑寫入可能改變先前的值；讀取失敗也不能抹去成功前綴。
具名結果與單一 accepted event 的 owner 契約見 [ADR-0063](0063-persistence-ownership.md#失敗路徑與遷移邊界)。
Receipt 只存在 GUI 生命週期的記憶體，不保證重啟恢復。這取代原先只回 IDs 的限制，
原因是接受回覆可能逾時，但 owner 已完成寫入；查詢必須能回當次實際內容。
作者處理 failed receipt，或以 `raise_for_error()` 保留診斷並終止流程，finally 仍 Save。
Report 不以 answer.accepted 宣稱已寫入。Full execution details 保留每次接受收據。

2026-10-23 使用者裁定：「0464沒意義，目前gui/mcp是所有權模型，如果中途改了writeback那就是agent有意的」「那本來就該接受。」
本裁定取代 2026-10-22 的 revision 綁定與 stale 拒絕。跨 pane 一次提交保留，recipe 的 `accept_writeback` 只走 `writeback_accept`。
Tunables Apply 的 revision 是另一機制，見 [[0062]]，不受本裁定影響。

### 自動推導的 owner

自動推導參數的規則有兩類 owner：

- Recipe 與 workflow 在內部推導自己流程需要的參數。
- GUI adapter 在建立實驗時生成初始參數，例如從工作點或 module library 取預設值。

這兩類規則各自屬於自己的 owner，不抽到與前端無關的共用位置。同一個量在不同 owner 中可以有不同的推導方式，各 owner 對自己的規則負責。

Recipe framework 只提供通用 cfg 操作，不知道 calibration、實驗欄位的意義或取值退回順序。實驗 helper 留在 zcu_lab，以通用操作組成流程。原裁決日期未記錄，使用者於 2026-10-16 map 結案時確認沿用，原話「都採用建議」。原裁決原話為「框架應該連calibration這個概念都不該知道」。

Recipe 只曝露入口需要 agent 決定的參數。曝露多少、如何對到 aliases 或 cfg，由 recipe 作者決定，framework 不生成一份通用實驗參數面。原裁決日期未記錄，使用者於 2026-10-16 map 結案時確認沿用，原話「都採用建議」。原裁決原話為「recipe的參數應該是這個實驗入口需要agent需要agent決定的參數，其他都隱藏起來，所以要曝露多少是這個recipe的設計決定。」

### 選層

Agent 先找符合目標的 L3 workflow 或跨實驗分析，再找 L2 recipe，最後才用 L1。L1 也用於 recipe 無法涵蓋的細部設定、排查與修正。這是指引，不是權限模式，每一層的公開入口都可以直接呼叫，與 [[0068]] 一致。

降到下層時，agent 要說得出上層缺少什麼。降層的原因若反覆出現，應補進上層，不讓 agent 長期在 L1 手動重組同一流程。

## MCP 設計原則

這些原則適用於 GUI app 的 agent MCP，不要求 MCP 完整覆蓋 GUI 或 RPC。2026-10-12 使用者裁決分工：「前者(MCP)面向agent，設計上要考慮agent使用負擔，後者(RPC)是控制接口，設計上要簡單基礎通用。前者透過組裝後者來兼顧雙方需求」。同日裁決操作粒度：「對agent來說，tool本身就是意圖，送什麼tool就切換到什麼階段。」

### RPC 是控制接口，MCP 面向 agent

RPC 提供簡單、基礎、通用的控制，回完整資料，出圖使用獨立 method。MCP 組裝 RPC、分派階段、摘要回覆並選擇出圖時機。GUI 是唯一的業務真相，MCP 不保存另一份業務狀態或驗證規則。傳輸及控制身份分工見 [[0068]]、[[0070]]；MCP→GUI remote 的依賴由 C13 約束。

### 從決策點列能力

先列 agent 必須判斷的事和需要的證據，再列完成這些判斷所需的工具。其餘 GUI 能力留在 RPC。Fluxdep 的決策點是資料可用性、對齊、採樣、躍遷假設、驗收和剔除。2026-10-12 使用者裁定範圍：「mcp的功能不必完全覆蓋gui的能力，只要能滿足：agent用mcp開啟或連上fluxdep_fit gui，讀取/加入/移除新數據，錨定對稱點，生成採樣點，Export數據集與採樣結果，Restore之前的數據集與採樣結果，Filter過濾此採樣點，附帶參數擬合目標參數，查看結果。每一步不一定照順序，也不一定只有一次。」

### 在決策點回證據圖

程式不能代為判斷的事回圖，機械命令回數字摘要。設計者給每個工具決定預設出圖時機，agent 可以要求 image=true。證據圖與 GUI 共用 renderer，兩個前端看同一份資料。Fluxdep 的 finish 回階段圖，fit 回 Show 疊圖。過期或沒有 recorded quality 的結果不回疊圖，這項限制由 fluxdep 的結果契約決定。

### Tool 即意圖

Fluxdep 對某個對象送階段工具，就進入該階段。Agent 不管理 open、cancel 或 context id。階段以編輯命令與 finish 組成；finish 接受並回證據圖，未編輯直接 finish 表示照原樣接受。切換顯示不取消別的對象草稿，GUI 跟隨 agent 操作的對象。Context lifetime 與同對象換階段的草稿處理仍由 GUI owner 決定。這不取消 measure 的 recipe cancel 入口。

### 命令直接表達動作

模式是命令的參數，不另設隱藏模式。能由其他命令表達的操作不另設命令。Fluxdep 的每個編輯階段以 finish 結束。它的 move_line 明填角色、位置與 conjugate；MCP 不暴露 swap_lines、set_tool、perform_on_all、clear 或 cancel。

### 不遷就 GUI 的操作方式

2026-10-14 使用者裁決：「mcp不要遷就gui的操作方式，能夠直接設定setup/device/inspect/...就不用打開dialog，也不要設定dialog focus自動跟隨」。

- Setup、Device、Inspect（參數、modules、aliases）等 dialog 編輯的內容，MCP 直接以 RPC 讀寫負責保存的 owner，不經過 dialog。
- Agent 的操作不開啟 dialog，也不切換 dialog 的分頁或焦點。`dialog.screenshot` 只擷取使用者已開的 dialog 目前顯示的頁，不改畫面。
- 本節只管 dialog。Agent 操作分頁時 GUI 跟隨的規則不變，使用者同日確認：「tab跟隨我保留」。

### 只回判斷所需的摘要

逐點陣列留在 RPC 與匯出檔。MCP 回計數、範圍與差量；changes 只列變動欄位的 [before, after]，invalidated 投影原生 publication 的下游失效。點或 mask 的新增／移除計數用 [added, removed]，不與欄位差量混用。參數 schema、範圍與領域詞彙放工具描述，不在每次回覆重複。精確值讀匯出檔。

### 長操作看 agent 中途能否影響結果

操作進行中有 agent 能判斷且能影響結果的事，就回 token 並提供進度與提早停止。沒有時，MCP 輪詢 RPC 到結束，不設整體上限，也不把 token 交給 agent。代價是同一 stdio server 等待期間不處理其他呼叫。Fluxdep 的 auto_align 與 search 採後者；measure Run 可看進度與 early_stop，採前者。

### 人與 agent 共駕

Measure／fluxdep 的 GUI 與 agent 共用同一份 application state，同時只有 holder 寫入，見 [[0070]]。狀態、圖與失效標示由同一 owner 提供，MCP 回覆能在 GUI 核對。控制轉移不取消已接受的工作，拒絕後讀現況再決定，不重送或回滾。本篇不把 holder enforcement 擴張到其他 app。

### 先做簡單設計，再按需加法

第一版只實作能力清單所需的命令與回覆。覆寫、選項與細部欄位有具體需求才加入。Fluxdep 的摘要與組裝需求來自實際分析中逐點陣列、schema 與重讀的成本，不把成本估算當驗收門檻。

### 判讀依據放知識庫

Skill 放流程與完成條件，每個決策點保留最小判準，並指向知識庫入口。完整判讀依據、反例與圖例放 measure_knowledge，不由單一前端擁有。GUI、MCP 與 Notebook 的 skill 可以共用它；skill 在沒有知識庫時仍能依最小判準運作。

### Fluxdep 的依賴模型不是跨 app 原則

2026-10-12 使用者對將失效與品質提升成通用範式裁定：「這點還不確定，先不進範式。」Fluxdep 的 alignment → points → Filter → fit 依賴、recorded quality、過期圖與匯出限制是 app 的局部契約，見 [[0068]] 及 app README。其他 app 的失效模型由各自的需求決定。

## 現況與後續

- Recipe 與 workflow 共用 Ask。Agent 用 request_id 回答同一 worker 上的問題，不另起 execution segment。GUI 與 MCP 投影同一 pending question，錯誤或過期回答不消耗問題。
- 無人回應時，Ask 停在原處等待。2026-10-12 使用者裁定：「就卡住，有需求再加設計。」Ask 不套預設決定、不逾時或跳過。根取消以 ScriptCancelled 解開等待。因此含 Ask 的 workflow 在無人值守時可能整段停住，overnight 流程要靠宣告的接受度減少提問。
- L0 邊界只靠政策維持。Agent 有 shell 時，技術上仍可繞過 MCP 碰觸硬體。目前由 MEASUREMENT 角色規則約束「不以分析腳本旁路控制硬體」。2026-10-12 使用者裁定：「有需要再加設計。」出現具體失敗類別時，再升級成權限設定或執行期強制。

## 取捨

- **自動化往下推，而不是讓 agent 編排每一步。** 另一種做法是只提供 L1 粒度的工具，由 agent 決定每一步參數、每一步之後的判斷與下一步。那樣新增實驗的成本低，agent 看得到每一步的完整資料。本篇選擇自動化往下推。Recipe 與 workflow 的步驟由程式碼決定，agent 不必逐步選擇操作，減少模型推理與工具往返。代價是每個 recipe 與 workflow 都要寫程式碼，自動推導的規則必須正確，agent 也較晚察覺推導規則沒有涵蓋的異常。
- **保留 L1 完整入口。** 只提供 recipe 與 workflow 會讓少見情況無路可走。保留 L1 的代價是工具面較大，agent 可能過早降層，所以選層指引要求說明降層原因。
- **推導規則留在各 owner，而不是抽成共用規則。** 共用規則能讓同一個量只有一種推導方式，但會把 recipe、workflow 與 GUI adapter 綁在同一份定義上，任何一方調整推導都要顧及其他使用者。留在各 owner 讓每個流程能依自己的需要推導，代價是同一個量的推導可能重複或不一致，需要時由各 owner 自行對齊。
- **以共享狀態交接，而不是由 agent 傳值。** 這讓 GUI、Notebook 與 agent 看到同一份校準結果，也讓 workflow 能無人值守地延續參數。代價是寫回需要明確的接受流程與 provenance。

本篇不新增 package 依賴方向。MCP 經 GUI remote 的方向沿 [[0068]] 與 C13；framework 不反向 import 使用者定義沿 [[0072]] 與 C16；workflow Engine 不依賴前端沿 [[0062]] 與 C18。這些既有 `.importlinter` contracts 不因介面層級而放寬。

## 相關文件

- [measure MCP README](../../lib/zcu_tools/mcp/measure/README.md)：固定工具、recipe 與 execution 的使用契約。
- [fluxdep MCP README](../../lib/zcu_tools/mcp/fluxdep/README.md)：跨量測整合分析的決策點工具。
- [zcu_lab workflows README](../../zcu_lab/workflows/README.md)：具體 workflow 的校準、接受度與保存。
- [run-measure-gui skill](../../.agents/skills/run-measure-gui/SKILL.md)：agent 量測時的操作規則。
- [討論紀錄](../discussions/2026-10-12-agent-interface-layers.md)：本篇的推導過程與被否決的選項。
