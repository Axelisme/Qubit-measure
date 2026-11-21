---
status: accepted
---

# 0075 — Agent 介面分層

使用者於 2026-10-16 以「都採用建議」確認歸檔。本篇記錄入口分層、recipe 與 workflow 的共用 async host、Ask 及取消契約。不新增工具、不改 method policy，也不授權硬體操作。

**Last updated:** 2026-11-21，guide 與知識庫的分工

**關聯：** [[0068]] 的 remote 與 measure 固定工具、[[0062]] 的 workflow run、[[0063]] 的保存權威、[[0069]] 的核心與前端；控制權見 [[0070]]。

## 問題

Measure、workflows、fluxdep 與 dispersive 的 MCP 各有工具。Agent 需要知道從哪一層開始、在哪裡介入，以及後續步驟該去哪裡取資料。新增工具時也要知道誰負責執行與保存。

## 決策

### 四層

**規則。** 入口按執行責任與 agent 的控制粒度分四層。

| 層 | 執行責任與入口 | Agent 的控制粒度 |
| --- | --- | --- |
| L0 實驗環境與硬體 | SoC、儀器 driver、program 執行，只經 L1 的 owner 操作 | 無直接入口，使用者也不直接控制 |
| L1 GUI／Notebook | 儀器、實驗、完整 cfg、實驗參數、單次分析與保存。Agent 用 [Measure 固定工具與 RPC](../../lib/zcu_tools/mcp/measure/README.md#固定工具與-rpc) | 實驗是多步流程，agent 可在任兩步間介入 |
| L2 Recipe | [Measure recipe 工具](../../lib/zcu_tools/mcp/measure/README.md#recipe-與-execution)操作 GUI tab，可串多個 tab 與實驗 | Agent 控制輸入、接受輸出，中途只在 recipe 明確交接時協作 |
| L3 Workflow | [Measure workflow 工具](../../lib/zcu_tools/mcp/measure/README.md#workflow-tools)一次設定跨多組實驗；measure-gui 承載，步驟不經 measure tab | 作者決定 Ask 的位置，agent 也可用 Stop 與 tunables |
| L3 跨實驗整合分析 | Fluxdep MCP 提供決策點工具；dispersive MCP 只讀 | Agent 驅動分析，順序不固定；程式能判斷的事不交給 agent |

**例子。** Overnight 或 T1 over flux 用 L3 workflow；要對多次量測、甚至多個 qubit 的資料做聯合擬合，用同層的 fluxdep。

**邊界。** Notebook 是使用者的 L1 入口，agent 目前不經 Notebook。Recipe 透過 L1 owner 操作 GUI；workflow 走實驗 runtime、自行保存。Measure-gui 與 workflow 共用程序、session、RUN lease 和 MCP server，承載細節見 [Measure GUI README](../../lib/zcu_tools/gui/app/measure/README.md#workflow-hosting)。Fluxdep 是獨立、可選的分析 app，不併入 measure 或 workflows，只讀取下層已保存的量測資料做分析。

**依據。** 2026-10-12，使用者定下分層：「提供agent從底層到抽象層的方法，最底層是實驗環境與硬體操作，這對用戶與agent都不可直接控制，上一層是gui/notebook層級控制，包含儀器控制，實驗建立，完整cfg控制，實驗參數控制等。再上一層是recipe，將常用實驗包裝層一個參數量小的自動gui操作，其餘參數用自動推導的方式，只在需要agent時yield給agent，這樣正常使用agent只需要傳入真的需要其判斷的資訊即可。再上一層是workflow gui/mcp，包裝常用實驗流程進一次實驗執行，比如overnight實驗，或是t1 overflux實驗，只需要要一次設定，就能完成跨多組實驗的流程。同層的還有跨實驗整合分析層，比如fluxdep_fit gui/mcp，支援跨多次量測數據甚至跨多qubit的數據整合分析。」見[分層討論](../discussions/2026-10-12-agent-interface-layers.md)。Recipe 不再以參數少或單次實驗來定義，協作也不再用 `yield`，見後兩節。同日對 L0 的強制機制裁定：「有需要再加設計。」2026-10-21，使用者對承載答「我也傾向B+D」，其中 B 是嵌入 measure-gui。見 [Workflow 定位與承載](../discussions/2026-10-21-workflow-positioning-and-hosting.md)。

### Agent 介入點

**規則。** L1 由 agent 選擇何時停下、讀取或修改。L2 與 L3 workflow 由腳本作者決定在哪裡交接。Recipe 與 workflow 自行推導其餘參數，只在需要判斷時以 Ask 交付問題與 answer schema。Recipe 操作 GUI tab；workflow 由 Engine 承載並持有自己的流程迴圈。兩者以 pydantic `FormModel` 宣告輸入。Recipe inputs 驗證 pydantic 與 MCP JSON schema，不渲染 GUI Form；workflow 表單由宿主提供。

**例子。** `await tab.run()` 及 `await tab.analyze(stage, result=T)` 是等工作，不是問答。互動分析以 `interactive=True` 交接 GUI tab，由 agent 用 `tab_interact` 完成選取。`await tab.ask_writeback()` 才是提問；回答不會寫入，作者接受時另呼叫同步 `tab.accept_writeback(items)`。Recipe 也可以在輸入端詢問缺少的參數。Workflow 中 `await scope.call(fn)` 等一般工作，`await env.ask(question)` 才交給 agent；交接可在實驗之間或之內。

**邊界。** 需要在 recipe 未交接的位置調整步驟時，改用 L1。跨實驗分析不由 workflow 決定介入點：程式管前置條件、計算與發布；agent 判讀證據、選下一步，可重做。人與 agent 共駕同一份 application state，由 [[0070]] 的 holder 寫入。Recipe 的 `target`、`reuse_tab_id`、`overrides` 是各腳本按需宣告的慣例，不是基底欄位；framework 不從 inputs 讀取開 tab 或 Reset 的兩個慣例欄位。輸入型別、`open_tab` 的角色對照、`KEEP` 與依序套用 override 的失敗契約見 [Measure recipe 輸入](../../lib/zcu_tools/mcp/measure/README.md#共用輸入與自訂步驟)。

**依據。**
- 2026-10-12，使用者區分 L1 與 L2：「L1中，每個實驗更像是一組多部流程，agent可以在中間介入。L2中實驗就是一個整體，agent只能控制輸入與接受輸出。」隨後補充：「L2不是完全無法介入中間，而是中間明確需要agent時使用yield來與agent協作。」同日指出「L3的workflow也是yield來協作。」見[分層討論](../discussions/2026-10-12-agent-interface-layers.md)。
- 2026-10-21，使用者說「recipe是一組操作gui的定位，其實不只能跑一次實驗，是可以做到類似workflow的行為的」、「該統一的是使用者與作者的認知與慣例」。執行機制與實驗語義分開；使用者當時說：「recipe執行要分兩層，一層是驅動引擎，與實驗語義解耦，只負責處理yield的流程。另外一層利用前者，提供run/analysis/ask_writeback等便利語法」。輸入由作者定義：「overrides不該出現在base class中，超過負責的範疇了」。見 [Recipe 接口重設計](../discussions/2026-10-21-recipe-interface-redesign.md)。當時的 `Wait`／`Ask`／`Signal` 與原話見[三角色協定](../discussions/2026-10-21-three-role-protocol.md)，現行等待與提問改用下一節的 async 契約。
- 2026-11-19，使用者問：「為什麼overrides不能作為慣例存在？不作為base class屬性，但幾乎每個recipe都會宣告他，這樣不就沒問題了嗎？」2026-11-20 以「好，我沒問題了，全部採用建議」採用各 recipe 自行宣告、lib 提供型別。見 [logical cfg 路徑與 recipe overrides](../discussions/2026-11-20-logical-cfg-paths-and-overrides.md)。
- 2026-11-20，使用者同意 recipe 自行宣告 `target`，並追問基底：「1B 2A 3A,目前共通RecipeInputs到底有什麼？ 4A」。確認原有兩個欄位後裁決：「那reused也從基底移除好了」。見 [Two-qubit MCP recipes 的首版範圍與輸入](../discussions/2026-11-20-two-qubit-recipes.md)。

### Async 協作與控制命令

**規則。** Recipe 與 workflow 的作者使用 `async def`／`await`，共用 `experiment.script_host.ScriptHost`。Caller 以 catalog 實名建立根 Scope；`host.root` 就是該物件。Host 在 caller 指定的 worker 執行作者 coroutine 一次，完成 `finally` 與 executor drain 才返回或拋錯，不 cancel 作者 task。它不改 GUI scheduler、operation owner、RUN lease 或 drain 邊界。Workflow 框架提供 `Scope.call`、`CallContext`，lib 的 `run_core` 組合資源並轉接 core，作者使用兩者且明選保存的 completion。Device snapshot 存在 `RunRecord.metadata`，不進 operation facts，見 [[0062]]、[[0063]]。

**例子。** Workflow 自行實作同步 `acquire(cfg, *, context)`，再 `await run_core(scope, acquire, cfg)`。一般工作用 `call`，每次只執行 `fn` 一次，不自動重試。`on_completed` 在作者 worker、operation 尚存活時同步記錄完成資料，再查根取消；hook 不 await、不重入，`fn` 或 hook 的原錯誤照樣傳出。[共用 ScriptHost](../../lib/zcu_tools/experiment/README.md#shared-async-script-host)與 [workflow 的 operation 契約](../../lib/zcu_tools/experiment/workflows/README.md#operation-與控制)說明結果、資源與關閉。

**邊界。** `ask` 回具名 Answer；GUI 和 MCP 投影同一個 pending question，`request_id` 對應同一 worker 上的等待。錯誤、遲到或重複答案不消耗問題。無人回答時維持 `waiting`，不套預設、不逾時或跳過；Answer 只讓同一 coroutine 繼續一次，根 Stop 可中斷等待。Journal 記 `QuestionAnswered`，不新增 `QuestionAsked`，也不記 Pause／Resume 事件。主動命令限控制命令與步驟邊界的 tunables，不支援 Run 中改科學參數或同時等待多來源。`control(fn)` 與 cancel／answer 共用 loop 控制序列，callback 只做數微秒、記憶體內的操作，不 await、不阻塞、不做 I/O。Lab 所有 `async def` 都受 [await AST 檢查](../../tools/README.md#lab-script-await-guardrail)，它不是完整型別檢查；唯一允許從 lib await 的 helper 是 `zcu_tools.experiment.run_core.run_core`。

**依據。**
- 2026-10-12，使用者對無人回答裁決：「就卡住，有需求再加設計。」見[分層討論](../discussions/2026-10-12-agent-interface-layers.md#討論中使用者的判斷與修正)。
- 2026-10-22，使用者裁定作者改用原生 async：「我之所以多次嘗試將generator換成async/await，是因為我認為作者應該比較熟悉原生python語法，而不是框架自己定義的語法，前者上手應該會比較容易」。見 [Workflow 協作設計的後記](../discussions/2026-10-22-workflow-single-generator.md#後記2026-10-22)。
- 2026-11-01，使用者定下框架、helper、workflow 的分工：「框架提供“能力“接口，搭配helper來組合能力與轉接core介面，再來是workflow內使用以上兩者」，以「1a 2a 3我記得record不是有meta屬性欄位？ 4可以」確認具體分工。這取代 `env.run` 與 ScriptHost 單次停止追蹤。使用者也以「5a」裁定 Ask 的 journal 行為。原裁決未附入庫討論，分別見 Git `8cde241c2`、`2a710ae44`。

#### 顯式 Scope 與取消出口

**規則。** Experiment 擁有包裝 `StopSignal` 的 Scope，底層 driver、clock、program builder 只收 `scope.stop`。作者以顯式 parent 決定工作歸屬，沒有隱式 current scope。取消表示盡快合法結束目標 Scope，不再為它排新工作。Root Stop 結束腳本；下一個 call 或 ask 拋普通 `ScriptCancelled`，不是 `asyncio.CancelledError`。進行中的 call 先合作 drain、記完成事實再拋出；`fn`、recording hook 或 cleanup 的真錯誤優先。根以下停的是這次等待的工作，正常回傳照常交付，拋錯原樣傳出，由作者選擇攔錯、保存、跳過或重試。

**例子。** Workflow 用 `with env.scope(label) as scope` 或 `for item, iteration in zip(items, env.iterations())` 管 scope 壽命。Acquire 只停一次取得，schedule 停該段後續取得，iteration 跳過目前 point。作者在主迴圈 `continue`、在 point helper `return`。Point 用 `try/finally` 同步 commit 一次，不在 `finally` await，也不吞根 `ScriptCancelled`。Commit 寫 snapshot，不啟動硬體、不換 iteration。Recipe 的 `finally` 也留在原 worker，可 Save 已有的 Run。根取消時 `report=None`，runtime 回 cancelled 與帶 `run_id`、`stop_requested`、Save 收據的 runs；`ScriptCancelled` 不記入 Host.error 或 recipe error。

**邊界。**
- 根限 workflow／recipe／run；框架與作者分別用 `branch` 和 `child` 建子層。With 或 release 收尾，release 只 detach，之後不能再建 child。Iteration generator 的 next／close 釋放前一趟／最後一趟。Recipe 無作者 iteration。Workflow 的根與 child 能力見 [workflow Scope 契約](../../lib/zcu_tools/experiment/workflows/README.md#顯式-scope-與生命週期)。Scope 不提供 run。
- `StopSignal.set` 只影響自己和子孫，reason 只保存與轉交。出生時繼承已停止的祖先，不繼承已釋放 sibling 的停止。`CallResult.stopped` 與 `Completed.facts.stopped` 只讀 operation 自己的 token，不由子孫反推停止或資料完整度。`Scope.cancel` 在自己的鎖內核對並設 token，只回 `CancelResult`，沒有 callback；ScriptHost 不追蹤單次 `stop_requested`。GUI／recipe 前端在自己的取消入口記錄該事實；workflow 由作者明填 completion。
- Workflow 根取消且 `expected=None` 時，Stop owner 先在自己的鎖內寫 `StopRequested`／manifest 並 latch，才通知 host 設 token；持久化不進 loop callback。Owner 還沒 finalization 時，即使 host 已退出，接受 Stop 仍可使終態 cancelled。其他 selector 交 host 核對，不寫 Stop 意圖。終態 workflow 對合法請求回 typed refusal，壞語法仍 fail fast；持久化失敗保留 failed 與原 cause。Host 首次根取消在自己的鎖內設 bit 並記 ack，不等 loop，重複請求回同一 ack；ack 不是已 drain。見 [workflow 控制契約](../../lib/zcu_tools/experiment/workflows/README.md#operation-與控制)。
- `run` 是 operation 的別名。Recipe 以同一 loop 回合核對並停止目前的 `tab.run`；分析、Ask 或閒置時回 `cancelled=False`、`reason="no run in progress"`，不停止下一個 operation。本地分析可選 operation；無結果的合作取消不是根取消，見[分析黑箱](#analyze-黑箱與作者的例外處理)。
- GUI Run 的 core 正常回傳就發布結果並標為 finished，不論停止目標；拋錯就是 failed。GUI Stop、MCP 不帶 scope 的 cancel 與 `cancel_scope("run")` 記 `stop_requested`，不以它、`cancel_requested` 或 `on_cancel` 決定 terminal。存檔的 completion stopped 另記提早結束。Failed pane 的 `source_operation_id` 與 `stop_requested` 只屬此次請求，不代表有 record；recipe 只查同一 handle，查失敗仍傳原 acquisition error。Notebook 計時器與作者對 partial 的責任見 [[0069]]。沒有 Pause gate、epoch 或丟棄重跑。

**依據。**
- 2026-10-22，使用者：「signal scope應該搭配有作用域保證的with以及for來規範scope範圍」「你認為signal是不是應該是顯式的？」「可以考慮讓scope成為zcu_lib/core的契約」「it.commit_dir可以對應到scope的層級與深度」。見[顯式 Scope 討論](../discussions/2026-10-22-explicit-scope-contract.md)；現行正常回傳與例外規則以後續裁決為準。
- 2026-10-31，使用者對停止通道、欄位改名與 `env.call` 例外直接傳出答「Signal可以有reason欄位可以填，但不解析，1同意 2你怎麼做？這能做到？  3可以 4a」。原裁決未附入庫討論，見 Git `76422a1cb`。當時 advisor 代裁的 ScriptHost 單次追蹤已被 2026-11-01 的三層分工取代；舊授權與設計見同一 Git 記錄，不影響前端的停止事實。
- 2026-11-01，使用者將停止定為「1是，具體來說是，盡快合法結束這個scope，2B，這算是緊急操作？所以要快 3我考慮移除這功能，太難實作了」，再以「6b 其餘採用建議」移除 Pause／Resume 並定 GUI 正常回傳一律 finished。原裁決未附入庫討論，見 Git `2a710ae44`。

#### 路徑、選擇器與跨 process 投影

**規則。** Scope 由父子鏈產生路徑，按同 parent 的 kind 從 1 編號；根無序號。Workflow／recipe 的根是 `workflow:<name>`／`recipe:<name>`，GUI 根是 `run`。Child 用 `kind:name#N`，無 name 省略冒號與 name，作者 label 用 `label#N`。非 `None` 的 name 不得為空或含 `#`、`/`、`:`；作者 label 不用分隔符或保留 kind。Workflow／recipe 以 catalog 實名注入根的 name，不做 fallback；operation 名由 caller 提供，不反射 callable。`active_path` 到第一個 fork 為止，沒有 fork 就是唯一葉路徑。`live_paths` 是葉路徑的 `list[str]`，不是逗號字串；GUI workflow pane 顯示兩個原生投影。

**例子。** Recipe 在 MCP process 有自己的樹，GUI Run 是另一棵樹。Recipe 在本地 `recipe:<name>/operation#N` 後接 `operation.status_scope` 讀到的 GUI 尾段，不重列 GUI 的 `run` 根。針對 schedule、acquire 或 composite 完整路徑取消時，以當前 `tab.run` handle 代理 `operation.cancel_scope`。取消回覆的 `active_path` 與 `candidates` 使用同一投影，GUI 的 `cancelled` 與 `reason` 原樣保留。

**邊界。** Cancel 搜整棵存活子樹，只接受唯一匹配，不選最新 child。Kind 可選非葉 root，未具名 GUI 的 `run` 按 kind 解析；具名 root 的完整路徑仍受葉路徑規則限制。同層同名 operation 要加 `#N`，跨 parent 仍匹配多筆就回 `ambiguous` 與候選。`expected` 須為正整數，不收 bool，和 `#N` 同填須一致；不存活或序號不符均不設 token。選擇器與拒絕理由見 [Measure MCP 取消契約](../../lib/zcu_tools/mcp/measure/README.md#取消與缺參)。兩邊各在自己的控制序列或樹鎖核對，不宣稱跨 process 原子。Execution 的 `expected` 只核對目前 GUI 根，下一 Run 可重用序號；跨 Run 精確取消要用觀察到的 op handle，舊 op 回 `scope not active`，不打到新 Run。Ack 只證明已處理命令，不證明工作或清理已 drain。

**依據。** Scope 路徑與顯式邊界源於 2026-10-22 的[顯式 Scope 討論](../discussions/2026-10-22-explicit-scope-contract.md)，使用者原話見[上一節](#顯式-scope-與取消出口)。現行跨 process 投影與 handle 核對見 [Measure runtime](../../lib/zcu_tools/mcp/measure/README.md#runtime-與回覆)。

#### Analyze 黑箱與作者的例外處理

**規則。** `core.analyze` 正常回傳就發布，拋錯就把 operation 標為 failed 並將原錯誤交回呼叫者。前端不按型別或原因分類，outcome 與 wire 不帶 `failure_kind`。成功 summary 照作者型別驗證，缺欄位不補值。`OperationOutcome.from_exception` 是唯一把例外轉成失敗顯示訊息的地方，使用 `repr(error)`；失敗結果拒絕缺少或空的訊息，GUI wire、MCP recipe 與 setup 不補預設。

**例子。** Fitting owner 的 `fit_func` 以 `AnalysisFailed` 回報 optimizer 不收斂或已知自由參數數目時的資料不足，不回初猜加 warning。Model callback 自己的錯原樣傳播。作者可直接從 Qt-free 的 `zcu_tools.analysis.failure` import 這個型別，決定捕捉後保存、跳過、重試或停止；未捕捉則 recipe failed。像 `ValueError()` 或 `ValueError('fit did not converge')`，顯示原 `repr`，不靠文字或型別挑選流程。

**邊界。** 子層分析合作取消且沒有結果時可回 `AnalysisResult(result=None, status="failed", reason=...)`；這不是 fitting 例外，根取消仍拋 `ScriptCancelled`。以字串原因建立的失敗保留原字串；event、UI 狀態與 log 的文字不受此規則影響。Experiment 不 re-export `AnalysisFailed`，fitting 不反向 import experiment。其他 backend 的有效性和 fallback 契約不因此全改，見 [fitting README](../../lib/zcu_tools/analysis/fitting/README.md#fitting-helpers)。

**依據。** 2026-10-31，使用者：「我可以接受analyze遇到問題就raise，但我不能接受框架識別raise的原因」。2026-11-01，使用者：「9我其實只要呈現error本身就好，比如就顯示ValueError()」，在 `repr(error)` 與「型別: 訊息」兩案選「9a」。原裁決未附入庫討論，見 Git `43813fba3`、`c6a685be5`。

#### 通用契約只有 ask 與正常結束

**規則。** Recipe 與 workflow 腳本對 agent 的通用契約只有 `ask` 與正常結束。`tab.ask_writeback`、`WritebackQuestion` 都是 helper，不限制提問的次數、順序或種類。Helper 不好用就修 helper，不擴充原語。

**例子。** Mux GE 的 Primary 與 Post 都有整筆替換同一個 mux readout module 的候選。舊 helper 把它們併入同一題，只以 name 判重，擋下合法選擇。修正是依 stage 出題，每次分析後各問一次，不改 `ask`。

**邊界。** 根停止與取消是 host 的控制命令，不是協作原語。需求不能用 `ask` 加 helper 表達時，例如需要 agent 主動通知腳本，才由使用者裁定原語設計。

**依據。** 2026-11-19，使用者：「我记得 recipe 不管是 ask 什么，底层都是基于同一个 ask。我自己认为 recipe 的通用契约就只有 ask，还有正常结束。这是一种协作式的原语，在之上架的工具都属于 helper 的范围。那现在很明显就是你这个 helper 不好用。本来 recipe 可以问任意的东西，回答任意的结果，没有什么次数或者是种类限制。」見 [Recipe 的寫回 helper 與 ask 原語](../discussions/2026-11-19-recipe-ask-helpers.md)。

### 層與層之間以共享狀態交接

**規則。** 下層寫入共享狀態，上層讀同一份狀態。Writeback 更新工作點參數與 module library；Ledger 與 Experiment Data File 的保存權威見 [[0063]]。

**例子。** [Fluxdep](../../lib/zcu_tools/gui/app/fluxdep/README.md#entry-and-export) 匯出時，符合條件才寫 predictor JSON。Fluxdep 透過 `entry.parameters()` 更新全域擬合參數。[Dispersive](../../lib/zcu_tools/gui/app/dispersive/README.md#entry-and-predictor) 從同一 entry 載入操作者選定的 predictor，使用檔內的能量與 flux frame。兩個 app 共用 [entry 選取與路徑規則](../../lib/zcu_tools/gui/app/analysis_project/README.md)，不讀 `params.json`。

**邊界。** Agent 可以讀資料做判斷，但後續實驗應取用共享狀態，不從對話帶入記住的數值。寫回需要明確接受與 provenance。

**依據。** 分層與交接的推導見[分層討論](../discussions/2026-10-12-agent-interface-layers.md)，使用者 2026-10-16 以「都採用建議」確認歸檔。

### Captured writeback acceptance

**規則。** Recipe 問答捕捉候選，接受時只把所選 item IDs 一次交給 GUI owner 的 `tab.writeback_accept`。Owner 依當下 draft 的值與目的地寫入。問答期間 holder 編輯或替換 draft 是有效意圖，不做 stale 拒絕，不還原舊 proposal。Owner 在第一筆寫入前核對 Primary／Post 全部 IDs，拒絕 unknown 或 duplicate；通過後按 Primary、Post、各 pane 的候選順序寫。首錯停止，保留成功前綴與診斷，不 retry 或 rollback。

**例子。** 同步寫入前先向 owner 取得一次性 `submission_id`，再帶同一 ID 提交；begin 不表示已寫入，重複提交會拒絕。若 10 秒 handler 逾時，拿回 ID 先查唯讀收據，再決定是否重送。收據只活在 GUI 記憶體，記當次成功 ID、實際值與固定目的地，不重讀當前 draft 或目的地。後續父／子路徑寫入可能改變先寫的值，不能在整批之後 readback；讀取失敗也不能抹去成功前綴。Wire 細節見 [Measure remote 收據契約](../../lib/zcu_tools/gui/app/measure/remote/README.md#captured-writeback)，具名結果與單一 accepted event 見 [ADR-0063](0063-persistence-ownership.md#失敗路徑與遷移邊界)。

**邊界。** `WritebackReceipt.items` 按成功 ID 關聯原捕捉候選，不比較 authored alias 與 concrete target，也不是實際值的快照；每筆實際值由寫入 owner 回傳，由 executor 保存，GUI 關聯 ID、pane 與固定目的地。作者處理 failed receipt，或用 `raise_for_error()` 保留診斷並終止，`finally` 仍可 Save。Report 不以 `answer.accepted` 宣稱已寫入；full execution details 保留每次接受收據。單 pane L1 `writeback_write` 仍編輯並寫入當前 draft。Preview 不回 revision，accept 不帶 `expected_revision`；workflow tunables 的 revision 是另一機制，見 [[0062]]。

**依據。** 2026-10-23，使用者：「0464沒意義，目前gui/mcp是所有權模型，如果中途改了writeback那就是agent有意的」「那本來就該接受。」這取代 2026-10-22 的 revision 綁定，保留跨 pane 一次提交。原裁決未附入庫討論，見 Git `acab11b8e`。2026-10-23 補充的一次性 ID 與逾時收據，以及 owner 回實際值的由來，見 Git `6b571aa5a`、`927b14e7a`。

### 自動推導的 owner

**規則。** Recipe 與 workflow 在內部推導流程需要的參數。GUI adapter 在建立實驗時產生初始參數。推導留在各 owner，不抽成與前端無關的共用規則；同一個量可按各流程需要推導。

**例子。** GUI adapter 從工作點或 module library 取預設值。Recipe 只曝露入口需要 agent 決定的參數；曝露多少、如何對到 aliases 或 cfg 由作者決定。

**邊界。** Recipe framework 只提供通用 cfg 操作，不知道 calibration、實驗欄位或取值退回順序，也不產生一份通用實驗參數面。實驗 helper 留在 `zcu_lab`，用通用操作組成流程。推導若需對齊，由各 owner 負責。

**依據。** 2026-10-12，使用者：「自動推導規則會由workflow/recipe內部，或是gui adapter的初始參數生成負責。」見[分層討論](../discussions/2026-10-12-agent-interface-layers.md)。使用者另裁定「框架應該連calibration這個概念都不該知道」、「recipe的參數應該是這個實驗入口需要agent需要agent決定的參數，其他都隱藏起來，所以要曝露多少是這個recipe的設計決定。」後兩句的原裁決日期未記錄，未附入庫討論；使用者於 2026-10-16 以「都採用建議」確認沿用，見 Git `854f4198b`。

### 選層

**規則。** 先找符合目標的 L3 workflow 或跨實驗分析，再找 L2 recipe，最後才用 L1。降層時說明上層缺少什麼；同一原因反覆出現就補進上層。

**例子。** Recipe 不提供某個細部設定，或需要排查、修正中間步驟時，用 L1 RPC，不長期在 L1 手動重組同一流程。

**邊界。** 這是指引，不是權限模式。各層公開入口可以直接呼叫，與 [[0068]] 一致；L0 沒有直接入口。

**依據。** 見[分層討論](../discussions/2026-10-12-agent-interface-layers.md)。使用者 2026-10-16 以「都採用建議」確認歸檔。L0 的強制邊界另見[現況與後續](#現況與後續)。

## MCP 設計原則

這些原則適用於 GUI app 的 agent MCP，不要求 MCP 完整覆蓋 GUI 或 RPC。

### RPC 是控制接口，MCP 面向 agent

**規則。** RPC 提供簡單、基礎、通用的控制並回完整資料；MCP 組裝 RPC、分派階段、摘要回覆並選擇出圖時機。GUI 是唯一的業務真相，MCP 不另存業務狀態或驗證規則。

**例子。** RPC 將取圖放在獨立 method；MCP 在需要證據的決策點呼叫它。

**邊界。** 傳輸與控制身份分別見 [[0068]]、[[0070]]；MCP 到 GUI remote 的依賴受 C13 約束。

**依據。** 2026-10-12，使用者：「前者(MCP)面向agent，設計上要考慮agent使用負擔，後者(RPC)是控制接口，設計上要簡單基礎通用。前者透過組裝後者來兼顧雙方需求」。見 [Fluxdep MCP 設計](../discussions/2026-10-12-fluxdep-mcp-design.md#mcp-設計範式)。

### 從決策點列能力

**規則。** 先列 agent 必須判斷的事與需要的證據，再列完成判斷所需的工具。

**例子。** Fluxdep 的決策點是資料可用性、對齊、採樣、躍遷假設、驗收與剔除。工具由這些決策點決定，不按 GUI 選單抄一份。

**邊界。** 其餘 GUI 能力留在 RPC。Fluxdep 的決策點不強制固定順序，也可重做。

**依據。** 2026-10-12，使用者裁定：「mcp的功能不必完全覆蓋gui的能力，只要能滿足：agent用mcp開啟或連上fluxdep_fit gui，讀取/加入/移除新數據，錨定對稱點，生成採樣點，Export數據集與採樣結果，Restore之前的數據集與採樣結果，Filter過濾此採樣點，附帶參數擬合目標參數，查看結果。每一步不一定照順序，也不一定只有一次。」見 [Fluxdep MCP 設計](../discussions/2026-10-12-fluxdep-mcp-design.md)。

### 在決策點回證據圖

**規則。** 程式不能代判斷的事回證據圖，機械命令回數字摘要。設計者為各工具定預設，agent 可要求 `image=true`。GUI 與 MCP 共用 renderer 和資料。

**例子。** Fluxdep 的 finish 回階段圖，fit 回 Show 疊圖。

**邊界。** Fluxdep 的過期或缺 recorded quality 的結果不回疊圖，這是 app 的結果契約，不是通則。

**依據。** 見 2026-10-12 的 [Fluxdep MCP 設計](../discussions/2026-10-12-fluxdep-mcp-design.md#mcp-設計範式)。

### Tool 即意圖

**規則。** Fluxdep 的階段工具指定對象，呼叫即進入該階段。Agent 不管理 open、cancel 或 context id。階段以編輯命令和 finish 組成；未編輯直接 finish 表示照原樣接受，finish 回證據圖。

**例子。** 成功的非唯讀命令讓 fluxdep GUI 顯示目標譜、主頁（Spectra 或 Fit）與階段，不轉移控制權。

**邊界。** `project.setup` 的對象是一直可見的標頭，不切頁。切換顯示不取消其他對象的草稿；context 壽命與同對象換階段時的草稿處理由 GUI owner 決定。Measure 的 recipe cancel 入口不受影響。

**依據。** 2026-10-12，使用者：「對agent來說，tool本身就是意圖，送什麼tool就切換到什麼階段。」見 [Fluxdep MCP 設計](../discussions/2026-10-12-fluxdep-mcp-design.md)。2026-11-20，使用者以「關於fluxdep gui，1A 2A」裁定跟隨主頁與階段，見 [Fluxdep GUI 跟隨 agent](../discussions/2026-11-20-fluxdep-gui-follows-agent.md)。

### 命令直接表達動作

**規則。** 模式是命令參數，不設隱藏模式。能由其他命令表達的操作不另設命令。

**例子。** Fluxdep 的 `move_line` 明填角色、位置與 `conjugate`，不另暴露 `swap_lines` 或 `set_tool`。

**邊界。** Fluxdep 的編輯階段由 finish 結束，MCP 不暴露 `perform_on_all`、`clear` 或 `cancel`。

**依據。** 見 2026-10-12 的 [Fluxdep MCP 設計](../discussions/2026-10-12-fluxdep-mcp-design.md#討論中使用者的判斷與修正)。

### 不遷就 GUI 的操作方式

**規則。** MCP 直接用 RPC 讀寫保存 owner，不經 dialog。Agent 操作不開 dialog，也不切換 dialog 分頁或焦點。

**例子。** Entry、Device、Inspect 的參數、modules 或 aliases 直接交 owner，不為此開設定視窗。`dialog.screenshot` 只擷取使用者已開 dialog 的目前頁，不改畫面。

**邊界。** 這條限制只管 dialog，agent 操作主視窗分頁時的 GUI 跟隨規則不變。

**依據。** 2026-10-14，使用者：「mcp不要遷就gui的操作方式，能夠直接設定setup/device/inspect/...就不用打開dialog，也不要設定dialog focus自動跟隨」。同日確認：「tab跟隨我保留」。見[主視窗跟隨的先例整理](../discussions/2026-11-20-fluxdep-gui-follows-agent.md)，原裁決未附入庫討論。

### 只回判斷所需的摘要

**規則。** 逐點陣列留在 RPC 與匯出檔，MCP 回計數、範圍與差量。`changes` 只列變動欄位的 `[before, after]`；`invalidated` 投影原生 publication 的下游失效。

**例子。** 點或 mask 新增、移除用 `[added, removed]`，不混作欄位的前後值。

**邊界。** 精確值讀匯出檔。參數 schema、範圍與領域詞彙放工具描述，回覆不重複。

**依據。** 見 2026-10-12 的 [Fluxdep MCP 設計](../discussions/2026-10-12-fluxdep-mcp-design.md#mcp-設計範式)。

### 長操作看 agent 中途能否影響結果

**規則。** Agent 在操作中能判斷且能影響結果時，回 token 並提供進度與提早停止。不能時，MCP 輪詢 RPC 到結束，不設整體上限，也不交 token 給 agent。

**例子。** Measure Run 可看進度與 `early_stop`，需要 token。Fluxdep 的 `auto_align` 與 search 等完成才回。

**邊界。** 等待期間同一 stdio server 不處理其他呼叫；只因操作久，不另加 token。

**依據。** 2026-10-12，使用者修正長操作一律回 token 的做法：「範式7不一定，重點是看agent在過程中能否發揮判斷的影響力，比如measure-gui的run就會回token，過程中可以檢測進度並early_stop。」見 [Fluxdep MCP 討論](../discussions/2026-10-12-fluxdep-mcp-design.md#討論中使用者的判斷與修正)。

### 人與 agent 共駕

**規則。** Measure 與 fluxdep 的 GUI 和 agent 共用 application state，同一時刻只有 holder 能寫入，見 [[0070]]。

**例子。** MCP 回覆的狀態、圖與失效標示都由同一 owner 提供，可以在 GUI 核對。

**邊界。** 轉移控制不取消已接受的工作；遇拒絕先讀現況再決定，不重送或回滾。本篇不把 holder enforcement 延伸到其他 app。

**依據。** 2026-10-12 的 [Fluxdep MCP 設計](../discussions/2026-10-12-fluxdep-mcp-design.md#mcp-設計範式)記錄共駕的分工；控制權契約由 [[0070]] 擁有。

### 先做簡單設計，再按需加法

**規則。** 第一版只做能力清單所需的命令與回覆。覆寫、選項與細部欄位有具體需求才加；便利的小功能等同一問題多次出現，第一次先靠文件或既有操作處理。

**例子。** 2026-11-19 的 FTF 任務只遇到一次 reload 後 tab 沿用舊 capture，不加「來源已變更」標記。要取新值，用既有的 Refresh 或 Reset。

**邊界。** 會給錯誤結果、遺失資料或擋住流程的問題，第一次遇到就修。同一任務中的 heralded density 失敗與 RPC 參數錯誤訊息屬這類。Fluxdep 的摘要與組裝需求來自逐點陣列、schema 和重讀的成本，不把成本估算當驗收門檻。

**依據。** 2026-11-19，使用者：「我遇到很多這種小功能，單個看沒什麼問題，但累計起來維護很困難，我沒有多次遇到問題不太想加，讓程式儘可能簡單好維護。」見 [Point 與 library 資料流的摩擦](../discussions/2026-11-19-point-library-frictions.md)及 2026-10-12 的 [Fluxdep MCP 設計](../discussions/2026-10-12-fluxdep-mcp-design.md#mcp-設計範式)。

### 判讀依據放知識庫

**規則。** Skill 寫流程、完成條件和各決策點的最小判準，並指向知識庫入口。完整判讀依據、反例與圖例放 `measure_knowledge`，供 GUI、MCP 與 Notebook 的 skill 共用。實驗 guide 寫給懂物理、沒看過實作的讀者，說明量什麼、怎麼操作、初值怎麼來、結果怎麼讀。板子限制與 QICK、program build、upload 的底層報錯不放 guide，沒有數字的中性說法也不放。知識庫的症狀表收錯誤訊息，judge 條目收判讀、單位換算與改法。

**例子。** [Single-shot Length Rabi guide](../../zcu_lab/v2/singleshot/len_rabi/guide.md)寫掃描上限初值是 4 × x180 長度，不寫 drive generator 的 envelope 容量。遇到 `generator 2: buffer length must be 8192 samples or less.`，讀者從知識庫症狀表找 judge 條目，查 samples 如何換成 µs、如何調整 `Length → Stop`、改用 `flat_top` 前要確認什麼。

**邊界。** 表單或結果中直接看到的行為仍寫進 guide，即使來自固定上限。例如 [reset_check guide](../../zcu_lab/v2/singleshot/reset_check/guide.md) 說 Gain Stop 初值取 4 × x180 gain 與 1.0 中較小者；[CPMG guide](../../zcu_lab/v2/twotone/time_domain/cpmg/guide.md) 說明 Minimum time 的公式及何時留空。Skill 不必有知識庫也能依最小判準運作；guide 不連到獨立、未被本 repo 追蹤的知識庫檔案。

**依據。** Skill 與知識庫的分工見 2026-10-12 的 [Fluxdep MCP 設計](../discussions/2026-10-12-fluxdep-mcp-design.md#mcp-設計範式)。2026-11-21，使用者先要求：「然後guide請不要包含取決於板子資訊的內容，其應該是中性的說明，要提供錯誤解析，應該放知識沉澱的疑難排解」。同日再定 guide 的讀者與範圍：「Guide的定位是給只有物理知識，完全沒看過實作的使用者角度看的，寫太多實作細節沒有意義，所以板子限制不要放guide，guide是介紹這個實驗，與其行為描述的說明，底層qick抱錯屬於疑難雜症，進知識沉澱的judge」。規則取自使用者裁決，guide／知識庫分工的條文由 designer 代寫，使用者 2026-11-21 覆核措辭，定為慣例（原話見 [ADR-0073〈自動推導點到為止〉](0073-measure-form-derivation.md#自動推導點到為止)）。見[表單預設超過板子限制，錯誤解析放知識庫](../discussions/2026-11-21-form-board-limits.md)。

### Fluxdep 的依賴模型不是跨 app 原則

**規則。** Fluxdep 的流程依賴與品質判定是 app 的契約，不升成共用 MCP 原則。

**例子。** Alignment → points → Filter → fit 決定下游失效；recorded quality、過期圖與匯出限制由 Fluxdep owner 管理，見 [[0068]] 與 [Fluxdep MCP README](../../lib/zcu_tools/mcp/fluxdep/README.md#fit-and-recorded-evidence)。

**邊界。** 其他 app 按各自的需求決定失效模型，不套用 Fluxdep 的依賴鏈。

**依據。** 2026-10-12，使用者對提升為通用範式裁定：「這點還不確定，先不進範式。」見 [Fluxdep MCP 設計](../discussions/2026-10-12-fluxdep-mcp-design.md#討論中使用者的判斷與修正)。

## 現況與後續

- 無人回答時，[Ask](#async-協作與控制命令)留在等待狀態。因此無人值守的 overnight 流程靠宣告的接受度減少提問。中途接續用新 run 與 saved_run reader，見 [[0062]]。
- L0 靠 MEASUREMENT 角色政策維持「不以分析腳本旁路控制硬體」，不是執行期強制。有 shell 的 agent 在技術上仍能繞過 MCP。出現具體失敗類別才升級成權限設定或執行期強制，見[分層討論](../discussions/2026-10-12-agent-interface-layers.md#留下的問題)。

## 取捨

Recipe 與 workflow 把步驟寫在程式裡，省下 agent 的推理和工具往返。每個流程要有作者，自動推導要正確；沒涵蓋的異常可能較晚被 agent 看見。保留 L1 讓少見情況仍有入口，代價是工具較多，agent 也可能過早降層。

推導留在各 owner，修改一個流程不必改別的流程，但不同流程的參數推導可能重複或不一致。共享狀態讓 GUI、Notebook、agent 用同一份校準結果，workflow 也能無人值守地取用參數。被否決的選項比較見[分層討論](../discussions/2026-10-12-agent-interface-layers.md#考慮過的選項與比較)。

本篇不新增 package 依賴方向。MCP 經 GUI remote 沿 [[0068]] 與 C13；framework 不反向 import 使用者定義沿 [[0072]] 與 C16；workflow Engine 不依賴前端沿 [[0062]] 與 C18。不放寬既有 `.importlinter` contracts。

## 相關文件

- [measure MCP README](../../lib/zcu_tools/mcp/measure/README.md)：固定工具、recipe 與 execution 的使用契約。
- [fluxdep MCP README](../../lib/zcu_tools/mcp/fluxdep/README.md)：跨量測整合分析的決策點工具。
- [zcu_lab workflows README](../../zcu_lab/workflows/README.md)：具體 workflow 的校準、接受度與保存。
- [run-measure-gui skill](../../.agents/skills/run-measure-gui/SKILL.md)：agent 量測時的操作規則。
- [討論紀錄](../discussions/2026-10-12-agent-interface-layers.md)：本篇的推導過程與被否決的選項。
