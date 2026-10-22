# `zcu_tools/mcp/measure/`

**Last updated:** 2026-10-22

**讀者：** agent。量測 MCP 的啟動、表單、operation 與寫回契約。

Measure MCP 透過 GUI 的 loopback remote socket 操作同一份 GUI 狀態。GUI core 不 import MCP。GUI `RemoteMethodEntry` 擁有 wire schema、exposure、readonly 與 operation policy。MCP 每次連線讀取 live `rpc.catalog`，不複製 policy 表，也不依 catalog 動態建立 tools。

## Startup scripts

`connect(..., script="/absolute/path/startup.py")` forwards trusted
`startup(ctx)` code only when this call launches a new GUI. An explicit script
replaces the composition-supplied default. Omitting it uses the default;
`clean=True` skips all scripts and conflicts with an explicit script. Scripts
may connect hardware; launch only reviewed code with the required authorization.

Explicit paths on attach, including an already-connected early return or
`if_missing` finding an existing GUI, fail with `script_requires_new_gui`.
The existing GUI is unchanged and startup never reruns on attach. Close it and
use `launch="new"` to select another script.

`connect(launch="new", project_root="/absolute/existing/directory", script="...")`
selects a separate data root. Omit `project_root` to use the repo root. MCP
forwards path strings unchanged; the launcher validates the data directory
before assembly. Entry owners create `result/` and `Database/` when needed.
Code, catalogs, logs and script-path resolution keep their existing roots.
An explicit data root on attach fails with `project_root_requires_new_gui`.
The [FTF startup example](../../../../zcu_lab/simulation_envs/FTF/README.md)
creates its entry, first point and tab without a separate entry CLI or setup RPC.

## 固定工具與 RPC

`mcp_servers/measure.py` 把使用者的 `RecipeDeclaration` 清單注入 session 與 `assembly.py`。`@recipe(name, description)` 從 async 函式的 inputs model 產生 MCP JSON schema；工具表拒絕不一致的注入及重名。Framework 不反向 import 使用者 recipes，也不從 GUI catalog 產生 tools。共用工具為 `connect`、`status`、`wait`、`cancel`、`rpc_list`、`rpc_describe`、`rpc_call`、`tab_analyze`、`tab_interact`、`tab_close`、`answer` 與 `apply_writeback`，以及 `device_setup`、`recipe_guide`。

## Setup 工具

`recipe_guide(adapter)` 以明填的 adapter 名查詢 GUI `adapter.guide`，回傳 adapter 與 `guide: {markdown, resource_dir}`。Recipe 可以操作多個 adapter，工具不推測 recipe 到 adapter 的 mapping。Markdown 保留全文；resource_dir 是 GUI host 的絕對目錄字串，未宣告 guide 時是 null。Guide 中的圖片使用相對路徑，agent 需要時自行從該目錄讀取。工具不讀圖、不附圖；跨機 agent 可能無法存取這個路徑。全域 status 不讀或投影 installed predictor。

`device_setup(name, cfg)` 接受 partial Info patch。工具先讀 before snapshot，合併 before.info 與 caller patch，再送一次完整 `device.setup(name, cfg)` RPC。Before Info 有 value_unit 欄位且 patch 含 value 時，caller 必須明填 value_unit，不能借 before 的 unit。這項規則同樣適用於 FakeDevice。只改 output／rampstep 不需要 unit。欄位必填、extra、mode/unit 與 limits 由 RPC 的同一 Info model validator 拒絕，connected admission 由 GUI DeviceService 擁有。儀器名稱、flux 規則與單位由 caller 和裝置 owner 決定。

Raw RPC 以完整 cfg 非同步執行 setup。Caller 先讀 `device.snapshot`，修改完整 Info cfg，呼叫 `device.setup`，用 `wait(op)` 等待，再讀 snapshot。全部 model-required 欄位必須提供，default 欄位沿 Info constructor。YOKO mode 必填；raw RPC 可省略 value_unit，由 Info 依 mode 產生。這不取代固定工具對 value patch 的明填要求。

`device_setup` 固定最多等待 300 秒。逾時保留正在執行的操作，回傳 running 與已知 op；caller 用 `wait(op)` 續等。需要短等時使用 raw RPC 與 caller 的 wait。Client deadline 必須大於 300 秒並預留 transport/reply overhead。

Device setup 保留 pre_read、start、wait、post_read、requested、before/after 與 native operation。Device requested 是 `{name, cfg}`，cfg 在 before-read 與合併完成前為 null，之後是完整候選，不保證 model 接受。Before.info 為 null 時不能合併，pre_read failed。Before 有 Info 則送一次 RPC，由 service 拒絕未連線裝置，回 start failed、完整 candidate、before facts 與 null after。Device 不做 verification；snapshot 有目標值不代表 operation finished。Native finished／failed／cancelled、driver error 與 unknown 原樣保留。Step completed 只表示該步成功。工具不重連、重送或回滾。Caller 用已知 op 的 wait 和完整 snapshot 接手。

## 日常與細部操作

日常量測優先 recipe，既有資料分析使用共用分析工具。細部 setup、cfg、保存、writeback 及排查由 RPC 承接。日常／排查是使用指引，不是權限模式。`rpc_call` 接受 catalog 中全部 `rpc` 與 `tool` methods；tool 名稱只提示高層入口。GUI 仍驗證每次操作。公開 method 的參數、回覆及前置條件由 `rpc_describe` 提供。

`rpc_call(method="predictor.list_files")` 回傳 `{files: [filename, ...]}`，列出目前 entry 的 `predictors/` 內按名稱排序的 `.json` 檔名。不需要選 point。未開 entry 或目錄不存在時回 `{files: []}`，其他 I/O 錯誤保留診斷。列表不讀取或驗證內容，也不建立目錄或寫入檔案。把回傳的 filename 交給 `predictor.info`、`predictor.predict` 或 `predictor.calibrate`。

Predictor 透過 `rpc_call(method="predictor.predict"|"predictor.calibrate"|"predictor.info", params=...)` 查詢。每次恰填一個來源：`filename="fitted.json"` 讀 opened entry 的 `predictors/fitted.json`，或 `params={"model":"transmon","EJmax":20,"EC":0.25,"d":0.1,"flux_half":0,"flux_period":1,"flux_bias":0}`（SQUID transmon，d 為 junction asymmetry），或 `params={"model":"fluxonium","EJ":5,"EC":1,"EL":0.5,"flux_half":0,"flux_period":1,"flux_bias":0}` 做獨立試算。filename 必須包含 `.json`，不接受 path、component 或 alias，也不需要選定工作點。直接參數等同 dialog 手動輸入，不需要 entry。兩種來源都不讀 dialog/cache。能量用 GHz；flux_half、flux_period、flux_bias 與 device_value 用來源的 native device units，不是 flux quanta；frequency_mhz 與預測頻率用 MHz。

`predictor.info` 回 `{model, params, missing}`，省略或 null 欄位保持未知。Predict 另填 device_value，Calibrate 另填 device_value 與 frequency_mhz。兩者需要完整有效參數，可填 from_level/to_level，預設 0/1。Calibrate 回 `{flux_bias_before, flux_bias_after, frequency_mhz, clamped}`，不寫元件、dialog 或檔案。兩種模型都選最小絕對 bias 的根，不選最接近目前 bias 的根。frequency_mhz 是實際達到的 MHz。不可達目標回全域最近頻率並標 clamped=true，同距時選最小絕對 bias。可達目標標 clamped=false。無效輸入仍拒絕，數值求解失敗不以 fallback 掩蓋。

filename 與 inline params 必須恰一個非 null。省略與 null 都視為未給。皆未給或皆給非 null 回 `invalid_params/invalid_predictor_source`。檔名或內容無效回 `invalid_params/invalid_predictor_file`。檔案無法讀取回 `precondition_failed/predictor_file_unavailable`；未開 entry 回 `precondition_failed/no_entry`。直接參數格式或物理值無效回 `invalid_predictor_params`。計算缺欄位回 `missing_predictor_params`，message 與 data.missing 列出缺項。工具沿 GUI 的 live catalog 描述與錯誤傳遞，不另建參數或寫回邏輯。沒有 target、load/set/clear 或 installed model。

Raw RPC 不提供高層工具的結果聚合或 PNG 解碼。例如 `device.connect` 回傳 handle，caller 用 `wait(op)` 等待後再讀 `device.snapshot`。`tab.save_data` 與 `tab.save_artifacts` 同步返回當次完整 Save report，不回 Save handle。Report 的 run_id 沿用 Run 時產生的同一個 ID，即使全部輸出失敗也不為 null。Caller 用各路徑的 status 與 error 判斷保存成功，不以 run_id 是否存在判斷。Report 保留全部 members，不用 await 或再次讀 snapshot。Readonly snapshot 的 confirmed run_id 仍只表示已有持久輸出成功。

## Recipe 與 execution

`recipe_api.py` 是作者的公開入口。`@recipe(name, description)` 宣告
`async def (session: RecipeSession, inputs: Inputs) -> Report`，
Inputs 繼承 `FormModel`，Report 是作者選擇的 `BaseModel`。
框架推導 inputs schema，不另維護手寫 schema、SummaryParameter 或 SummaryEstimate 宣告。
Recipe inputs 只驗 JSON schema 與 pydantic，不渲染 GUI Form。
Adapter tab 的 Form 仍由 GUI owner 產生與驗證。

`session.open_tab(adapter, aliases=..., reuse=...)` 開 tab 或驗證 reuse，再 Reset 捕捉來源。
省略 aliases、None 或 {} 使用 Reset 當時的全域表。
非空 mapping 只覆寫這次固定 capture，不寫全域 aliases 或 memento。
Recipe 不自動選重用 tab，也不自動關閉 tab；關閉用 `tab_close`。
`session.context.read(source_path)` 直接讀 session 層的 `value.read`，不需先開 tab。
`tab.read(path)` 讀該 tab 固定 Form capture，回 detached `FormField`。
Structured value 保留 owner 的 `YamlValue` mapping，作者需要 model 時自行 TypeAdapter 解碼。
Value_status 區分 available、missing、unavailable；available null 是有效觀察，不證明整份 cfg 有效。

Form path 是非空 tuple，str 是原始欄位名，int 是完整 list 的原始位置。
不拆 dotted string，不按 name 或 participants 順序重編位置。
`tab.set`、`select`、`apply`、`call` 同步交給 GUI owner。
寫入接受 FormValue 與 Expr，讀取側不回 Expr。
`set(path, None)` 保留 GUI input，停用 module 用明確的 `"off"`。
Readonly、ref、selector、來源與 cfg admission 都由 Form owner 檢查，讀取不授權寫入。

作者用 `await tab.run()`、`await tab.analyze("primary"|"post", result=T)`、
`await tab.ask_writeback()` 與 `await session.ask(question)` 串接流程。
Run 回 `RunResult`，不自動保存。
Analyze 用 TypeAdapter 將 owner summary 驗證為 T，不補欄位或修資料，也不自動保存或 preview。
`interactive=True` 明確開啟互動交接，公開 `PendingAnalysis(tab, stage, op)`。
Status／wait 提前回 `status="interactive"`，GUI 完成同一 operation 後原 await 繼續，不建第二個 runner。

`tab.writeback_proposal()` 捕捉 completed stages 的候選。
`ask_writeback()` 等待經 evidence 驗證的 `WritebackAnswer`，回答不寫入。
只有 `tab.accept_writeback(decision.items)` 寫入選定的 captured 候選。
Unknown、重複或已變更的項目在第一筆寫入前拒絕。
Recipe 將 preview 的 owner revisions 與選定 IDs 一次交給 `tab.writeback_accept`。
Owner 在同一提交核對 Primary／Post 全部選定 drafts，任何 draft 改動或替換都在第一筆寫入前拒絕。
單 pane L1 `tab.writeback_write` 仍編輯並寫入當前 draft，不供 recipe captured acceptance 使用。
Primary 先於 Post，首錯停止，成功前綴保留，不 retry 或 rollback。
`WritebackReceipt` 以 pane 與 owner item ID 關聯 confirmed items，不拿 alias 名稱比 concrete target。
Proposal、receipt 與 full details 的 structured mappings 各自分離，不共用 retained acceptance evidence。
作者原地修改 proposal 的 nested value 後接受，會在任何 owner write 前拒絕。
作者可檢查收據或呼叫 `receipt.raise_for_error()`，不可將 decision.accepted 當作寫入成功。
Full details 的 writeback_receipts 保留每次接受的確認項目與錯誤，失敗後的 finally Save 仍執行。
`WritebackDestination.path` 是 owner 原樣的 opaque actual_path，
label 是 preview 當時 active context 的顯示標籤，不是 captured identity。

`tab.save()` 同步呼叫 `tab.save_data` 與 `tab.save_artifacts(artifacts="all")`，
保存 raw data 與全部已完成階段的 JSON／圖。
兩個 RPC 都在 GUI owner 同步完成，沒有 Save operation 或 handle 要 await。
每次 Save 的完整 owner report 記在 runs，不累積跨次成功宣稱。
同 run 的 native 成員可覆寫，Labber 每次另產新檔。
每個 output 的 status／error 才是保存結果，不能只看 run_id。
通常在 return 前以 finally Save 一次；想先保住 raw 也可在 Run 後另存。
Run 未成功取得來源前不要進 Save finalizer，避免 Save 蓋掉原錯誤。

### 取消與缺參

`cancel(execution, scope="script"|"run")` 預設 script，沒有 finish_early。
Run scope 在 host loop 同一回合檢查與 set 正在取得資料的 token。
Analysis、Ask 或閒置時回 `cancelled=False, reason="no run in progress"`，不改 execution。
當前 Run 合作返回 `RunResult(partial_result=True)`，recipe 繼續分析、問答與保存。

Script scope 設根 token，當前與後續 Run 都會啟動並返回 partial。
Analyze、ask_writeback、session.ask 無結果時 raise 普通 `RecipeCancelled`，
不是 `asyncio.CancelledError`，三者的作者回傳型別都不是 Optional。
作者不 catch、不寫取消分支，finally 在同一 worker 照跑，同步操作仍可做。
終態為 `RecipeOutcome(status="cancelled", report=None, runs=[...])`。
RecipeCancelled 不記入 error，其他 owner、作者或 cleanup 例外仍保留診斷。
Completed 不代表所有輸出已保存。

Run 前的 `RecipeNeedsParameters` 使 execution 終態為 needs_parameters，
snapshot 列具名 missing paths、outcome=None，並釋放 admission。
RecipeOutcome 仍只有 completed／failed／cancelled，不把缺參包成失敗。

### Runtime 與回覆

`script_recipe_execution.py` 使用 production `experiment.script_host.ScriptHost`。
`recipe_runtime.py` 擁有 GUI binding、wire 解碼、合作取消與 operation source；
`recipe_host.py` 組合 scheduler、runtime 與 author session。
`recipe_execution.py` 管理注入 declarations、單一 recipe admission 與 execution 查詢。
舊 RecipeDefinition／generator handles 路徑已移除，尚未遷移的 workflow driver 不在 recipe 路徑使用。

Host loop 只在 execution worker。作者 task 不被 cancel，阻塞操作由 executor 合作 drain。
每個 operation 都有獨立 StopSignal child，已停止的 token 仍啟動 operation。
`set_error` 只記本 scope 第一個錯誤並向下停止，不改祖先或後續兄弟。
跨 thread 的 cancel／answer／control 經同一 loop 控制序列與 acknowledgement。
Acknowledgement 不是 operation drain 的證明。
`control(fn)` 的公開 memory-only callback ABI 見
[experiment README](../../experiment/README.md#shared-async-script-host)。

首次 recipe 呼叫最多等 300 秒，缺參、失敗、互動分析或問題可提前交付。
仍執行時回 execution，逾時不停止背景流程，也不自動重跑。
Client deadline 應超過 300 秒並預留 transport overhead。
Stdio server 同步處理請求，不保證首次等待期間同連線的控制請求立即處理。

`answer(execution, request_id, answer)` 必須帶目前問題的 request id，
answer 符合 pending question 的 Answer schema 與 evidence 規則。
錯誤、遲到、重複答案回 accepted／reason／pending_request_id，
不消耗問題、不記 answered。
問題沒有自動預設回答或 timeout。

Terminal summary 只回 status／report／runs。
Report 由作者定義，runs 記每次 Run 的 tab、adapter、run_id、operation status、
partial 與 Save 收據，不靠作者補取消報告。
非終態另帶 execution 與明確的 pending question／analysis。
`status(execution, detail="full")` 保留 detached Run 前 Form 條件、
各 stage 的 owner-native summary／invalid、最後明確捕捉的 writeback proposal、各次 writeback_receipts 與 error。
Scalar、uncertainty、unit 與元素 identity 都沿 owner summary，不重算或套 reporting 宣告。
Query 不重讀 GUI，也不自動存檔。

全域 status 的 entry 是 null 或 GUI entry.info 的 UUID、name、result_path、database_path，
只讀 metadata，不建立 context observation。
未 open entry 時 ready.missing 包含 entry。
全域 status 列 GUI operations 與本 MCP session 非終態 executions，
終態只給數量及按 ID 查詢提示。
`wait(op)` 只觀察 GUI operation，`wait(execution)` 等本地後續處理與當次 image 交付。
Execution ID 是 session-local，不是持久恢復機制；query 不建立新的寫回授權。

### 作者 await 邊界

所有 `zcu_lab/` async def 都使用同一規則，包含未被呼叫的 helpers。
只 await env／tab／session 的方法呼叫，或可靜態解析到 lab async 定義的 helper。
Workflow 可寫 `await helper(env, cfg)`，helper 可再 await 另一個 lab helper；
helper 內的 await 同樣受檢查。
工具拒絕 asyncio.*、create_task、gather、wait_for、TaskGroup，
明列原語的直接 Call 即使不 await 也拒絕。
其他 async with／async for 不另禁。
Quality gate 執行 `tools/check_script_awaits.py`，不 import lab。
這不是 type checker；assignment aliases、動態分派與包裝可繞過規則，作者仍須遵守序列執行契約。

## Workflow tools

Workflow 工具在同一 measure MCP session，與既有 recipe `answer` 分開命名。
所有命令都不以 GUI toggle 或目前選中的 tab 作為 target，沒有 workflow 模式切換 RPC。
名稱必須明填，不預選實驗室 workflow。GUI 掛載新開的 owner 時會選中新 tab，並切到 Workflow 模式。

| 工具 | 輸入與結果 |
| --- | --- |
| `workflow_open` | `name`，回 `tab_id/name/tab` snapshot，可同名多開。 |
| `workflow_close` | `tab_id`，回 `tab_id/closed`。Busy 回 false，不替 caller Stop。 |
| `workflow_status` | 可選 `run_id`。明填回 native status 與 tab 識別，省略/null 回 names 與全部 opened tabs。 |
| `workflow_start` | `tab_id/run_id`，可選 `plan/tunables` batches。回 pending `handle`，用既有 `wait(op=handle)` 觀察終局。 |
| `workflow_pause/resume/stop` | 明填 native `run_id`。Pause 保留 lease，Resume 重用原 run 與 handle，Stop 等 terminal delivery。 |
| `workflow_form` | `tab_id/form`，form 是 plan 或 tunables。可選 batch 只 preview，回 input/values/errors。 |
| `workflow_describe_form` | 同上另給 typed `path`，root 是 []。只加同一 candidate 的單層 metadata，不回整棵 schema。 |
| `workflow_edit_form` | `tab_id/form/batch`，修改 owner 的 shared draft，回更新後觀察。 |
| `workflow_tunables` | `run_id/batch`，經 owner Apply，回 native revision/values。空 batch 提交當前 draft。 |
| `workflow_records` | `run_id` 與可選 `since=0`，回 commit_seq 嚴格大於 since 的 native records。 |
| `workflow_figures` | `run_id`，只回 names，不 render。 |
| `workflow_figure` | `run_id/name`，回識別與 validated PNG image content，text 不保留 base64。 |
| `workflow_answer` | `run_id/answer`，answer 是符合 status 中 `pending_question.answer_schema` 的 JSON object。回 Start 的同一 pending handle，不宣稱已完成。 |

Batch 是 `[{"path": ["field", 0, "leaf"], "value": ...}]`，路徑是 canonical
欄位名與原始 list index。Form owner 決定 branch、expression、有效值與錯誤。
MCP 不計算 defaults、不做 lowering，也不另建 Engine、run registry 或 holder 規則。

Start 省略/null batch 使用 shared GUI draft。明填 batch，包括 []，使用 owner
當次 fresh captured defaults 加 edits。Form preview 的 [] 則讀當前 parsed draft，
不重算 defaults。Tunables Apply 的 [] 提交當前 draft，不把它重設為 defaults。

工具表是固定手寫，沿 live catalog 的 timeout、權限及 session operation mapping。
錯誤保留 code/message/reason/data，不重試、不回滾。Readonly 可查詢與 preview，
mutation 仍受 holder、busy 與 RUN lease 限制。Terminal records/figures 可繼續讀取。
Status 的 `pending_question` 是固定 `{question_type, question, answer_schema}`
envelope；只有 `question` 含 authored fields，沒有問題時為 null。Schema 與 JSON
轉換由 Engine/driver 擁有。Status 的 `last_exchange` 保留最近問題 envelope、接受的
answer 或 null，以及 waiting/submitted/answered/cancelled。它與 GUI 共用 headless
owner；submitted 不是已接受，錯型答案不記入 answer。錯欄位保留 paused 問題，
回 `invalid_params/answer_validation`。
Answer 與 execute 一樣會同步推進 generator，facet 將它排到 worker，保留同一
RUN lease、operation 與 terminal delivery。UI owner thread 不 inline 續跑，也不
另外呼叫 Resume。Stop 的 retained Ask drain 同樣在 worker，終局交付才釋放 RUN。

## 分析、互動與寫回

`tab_analyze` 在既有 Run／Load 上啟動 Primary 或 Post，沒有 recipe 身分。
它不補存 raw，raw_data 的 run_id／saved 沿 GUI owner 的當次觀察。
run_id 在保存前已存在，saved 只在 measurement_data 全部保存時為 true；
尚未捕捉到觀察時 raw_data 為 null。
這與 analysis artifact 保存分開，status 不重查目前 tab。

Standalone analysis execution 等原 GUI operation 完成，讀結果、保存 analysis artifacts，
再以原 opaque operation 讀一次 writeback preview。
沒有圖也讀候選，Read 失敗保留已有 result、當次 Save report 與 image，回 writeback_read error。
來源 superseded 不改讀新 pane，不 retry。
這個自動保存是 standalone 工具契約，不適用 recipe 的 `await tab.analyze()`；
recipe 作者另外明確 `tab.save()`。

`tab_interact` 不帶 payload 時讀 committed state、commands、info、preview_active 與圖，
不改焦點。帶 payload 時 GUI 驗證命令並跟隨 Analysis pane。
`done` 加入原 operation 的完成，不另建 analysis runner。
Recipe 的 done 回原 recipe 後續結果或問題，standalone 則完成自己的 execution。
純讀與 command 都要求 holder，owner-loop commit 依序生效，沒有自動 retry。

GUI 將 analysis summary 的非有限數字投影為 null，以 invalid 記路徑與原因。
MCP 保留同一份 owner-native summary，不重算 fit_quality、不套 recipe 的 summary 宣告，
也不設定寫回門檻。
有限 fit、不可估的 uncertainty 與 warning 可以同時保留。
缺值與明確 null 不互相替代，artifact 保存失敗也不抹掉已知分析結果。
Recipe 的 typed T 驗證不會補資料或更改這份 owner summary。

`apply_writeback(tab, items=None)` 按當前 preview 的穩定 target_name 選項目。
省略或 null 寫全部，[] 寫零項，不看 GUI 勾選，也不回答 recipe。
工具先讀 Primary 與既有 Post，再拒絕未知、重複或跨 stage 同名，
全部通過後才以當前 session ID 寫入。
Primary 先於 Post，首錯停止，不 retry 或 rollback。
收據只列 GUI-confirmed completed，保留 skipped、failed stage、
not_started 與失敗 stage 的不確定性。

Recipe 的 captured 問答與這個當前 draft 工具不同。
Recipe 只在 `accept_writeback(items)` 寫入已捕捉且仍有效的候選，
回答本身不授權寫入。
Destination.path 保留 owner actual_path，label 只作顯示。
元素 index／name 由 lab owner 明確提供，framework 不從 row offset 或 participants 猜 identity。
Full 查詢保留原生候選與目的地，不把 preview 當成授權。

保存收據保留全部 members、路徑、format、status 與 error。
Reserved 不代表已保存，run_id 存在也不代表任一輸出成功。
同 run 重存可覆寫 native 成員，Labber 另產新檔。
查詢不 stat 檔案、不補存，也不累積跨次成功路徑。
當次回覆可附記憶體中的 MCP image content，不落地暫存 PNG。
之後用 tab.get_figure 讀 GUI 當下圖，不保證等於先前回覆；status 不附圖或重新 render。

候選編輯用 tab.writeback_* RPC，module edit 與 tab Form 共用 typed path 陣列。
Library 用 context.ml_edit，child key 保持原 key；值查詢用 value.list／value.read。
MCP 的 context.aliases_set 套用整張 authored 表，context.set_parameter 寫目前 point 的具體路徑。
這些操作直接走 owner，不開 Inspect、不切頁，也不改既有 Form／Run 的 capture。
Aliases 在 Inspect 第三頁，dialog.screenshot 只拍目前頁，不接受 tab selector。
GUI 在每次 dispatch 檢查 holder、source 與 readiness，preview 不取代這些檢查。

## 連線與觀察

`connect` 可連既有 GUI 或依 launch 參數啟動。啟動腳本可連硬體，執行前須審閱腳本並取得授權。首次 label 預設 `agent`，mode 預設 `owner`；`readonly` 永不取得控制權。Owner 在未鎖定 Local 自動取得。GUI 鎖定或別的在線 holder 不阻止連線，回覆的 `control` 顯示 `{holder, locked, online, me}`。身份初始化在認證與 wire 檢查之後、catalog 之前。省略 label/mode 的重連保留原身份，顯式換身份會重開 socket。

`rpc_call` 提供 `control.status`、`control.acquire`、`control.release`。重複 connect 同一身份只查詢現在的控制狀態，不自動再 acquire。四種控制拒絕 `not_holder`、`control_locked`、`control_held`、`readonly_connection` 在 MCP `isError` 的文字 content 中保留 JSON `error`，包含原生 code、message、reason、data。GUI 負責每次 dispatch 的權限判定，MCP 不複製 holder 規則。

Control token 留在 session 供重新握手；顯式切換 port 不沿用前一 GUI token，tool log 遮蔽 token。認證失敗與 wire 不相容分別回報。Bridge 不會把仍活著的自有 GUI 當成可在另一空 port 再啟動的程序。

Session 將 GUI operation ID 映射為 opaque integer handle。明確重連或 EOF 後清除 catalog 與舊 handle 對應。GUI 可重用 wire ID，MCP session 不重用已外露 handle。GUI-origin operation 經 `status` 使用同一映射。每次 wire 操作前先確認連線，再解析 handle；不向新 GUI 重送舊 ID。

GUI 在 owner dispatch 檢查目前 holder，不維護每連線 seen baseline。MCP 不保存第二份 authority 或版本表，也不以隱藏讀取解鎖 mutation。只讀 method 可由 observer 使用，`tab.interact` 的純讀與 command 都要求 holder。控制身份與 domain readiness、busy、source-bound operation 及 authentication 分別檢查，見 [ADR-0070](../../../../docs/adr/0070-control-identity.md)。

接手既有或重啟後的 GUI 時，按操作需要讀 tab、SoC、context 與 device snapshots，核對實際狀態，不是建立寫入許可。Context 可能包含大型敏感資料，不可序列化時完整讀取失敗。摘要與 resource counters 不代替內容。

Cfg 使用 `tab.get_cfg` 的 Form input/values/errors。欄位能力用 `tab.describe_cfg` 逐層讀取，不下載整棵 schema。`tab.edit_cfg`、`tab.reset_cfg` 與 `tab.run_start` 不收 ref。Edit 作用於目前 input，Reset 重新捕捉來源及 defaults。Run 只接受目前有效值，並隔離執行值與 source provenance。後續編輯不改已接受 Run，轉移或斷線也不取消它。新 holder 可取消，observer 可 wait。

Admission 拒絕、斷線或 timeout 都不自動重送。傳輸失敗不證明 mutation 沒有副作用；先讀現況，再由 caller 決定下一步。

## 傳輸與關閉

Shared SocketTransport 以 shared framing 限制每個 frame 為 8 MiB UTF-8 bytes，不含換行，不分批。超大 request 在送出前拒絕，連線仍可用。超大 response 關閉連線並回 `message_too_large`，不能假定 mutation 未執行。重新連線會初始化身份並重載 catalog，舊 handle 不跨連線使用。

`tab_close` 與 RPC `app.shutdown` 都由 GUI 在同次 owner dispatch 檢查 active operations 及 unsaved artifacts。`discard_unsaved` 不能略過 busy。Shutdown 回覆的 `shutting_down` 與 PID 表示已接受正常關閉要求，不保證程序已退出；RPC 不等 process exit、不強制終止。

MCP server 結束時斷線並加入其 workers。它不關閉 GUI，也不保證停止硬體。MCP 不訂閱或佇列 GUI events；完成情況透過 snapshots、operation wait 與 execution 讀取。

## 驗證

`tests/mcp/measure/` 經 public tools/session 與 recording transport 驗證 recipe、分析、catalog、連線、holder admission 與 operation。GUI remote/service 測試驗證 socket 與 GUI-origin 路徑。離線選集只用 fake/mock，不啟動真儀器；fixture 與測試歸屬見 `tests/README.md`。

Entry new/open 只建立或開啟 entry，不自選 point，也沒有參數環境。Agent 先明確呼叫 point.use 或 point.new，才能 Run、WB 或目前 point 的 Inspect writes；未選取時沿既有 failed precondition 拒絕。WB 使用候選捕捉的 SavedPoint，選取其他同 entry point 不更換目的地。
