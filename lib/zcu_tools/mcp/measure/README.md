# `zcu_tools/mcp/measure/`

**Last updated:** 2026-10-24

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

## Point 檔案讀寫

目前 point 的參數、library 與 aliases 以磁碟檔案為權威。
GUI、writeback、recipe 與啟動腳本經 owner 寫入時，成功即落盤。
Agent 直接讀寫這些檔案，固定工具不提供參數／modules／aliases 特化讀寫。
`entry.info` 與 `context.reload` 是 RPC methods，經 `rpc_call` 呼叫，不另設固定工具。

1. 先用 `status()` 或 `rpc_call(method="entry.info")` 取得頂層 `files`。
   每筆為 `{document, path, revision}`，順序是 parameters、library、aliases。
   Path 是絕對路徑，revision 是已安裝 bytes 的 SHA-256。
   Aliases 尚無檔案時 revision 為 null；未選 point 時 files 為 []。
2. 依 path 直接讀檔。外部候選尚未通過驗證時，磁碟內容可能與已安裝版本不同。
   Status 不讀檔，也不把未驗證候選當成現值。
3. 修改前確認自己是 holder，保留該筆已安裝 revision，然後編輯對應檔案。
   檔案系統不執行 GUI 的 holder 檢查，取得控制權不能省略。
4. 呼叫 `rpc_call(method="context.reload", params={"path": path, "expected_revision": revision})`。
   expected_revision 必須明填，只有預期 aliases 缺檔時可填 null。
   Owner 比對已安裝版本，再讀取、驗證、安裝磁碟候選並發布事件。
   成功回 `{document, path, revision}`，revision 是新安裝版本。

驗證失敗會回路徑與原因，owner 保留舊記憶體，不改磁碟候選。
Revision 不符時重讀 files 與檔案，再決定是否修改，不自動覆寫或重試。
非 holder 的 reload 會拒絕，修正控制權後再由 caller 決定下一步。
Watcher 在 agent 持有時也可能先安裝合法候選；此時舊 revision 的 reload 會拒絕，
先核對已安裝版本即可，不重做寫入。
人持有時 watcher 只報待處理變更，Inspect 有草稿時提供「重載／保留」。
這些操作不改既有 Tab／Run 的固定 capture；Tab Refresh 保留 input 並捕捉新來源，
Reset 重新捕捉來源並重算 defaults。

Live catalog 的 `value.read`／`value.list`、`tab.parameter_read`、
`context.set_parameter`、`context.ml_*` 與 `context.aliases_set` 仍可經 `rpc_call` 使用。
這些 RPC 沿 owner 驗證與 holder admission，不是額外的固定工具。
Recipe 內部仍走 owner API，無須改成直接編輯檔案。
各檔獨立提交，不提供跨檔交易或自動合併。

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

Raw RPC 不提供高層工具的結果聚合或 PNG 解碼。例如 `device.setup` 回傳 handle，caller 用 `wait(op)` 等待後再讀 `device.snapshot`。`tab.save_data` 與 `tab.save_artifacts` 同步返回當次完整 Save report，不回 Save handle。Report 的 run_id 沿用 Run 時產生的同一個 ID，即使全部輸出失敗也不為 null。Caller 用各路徑的 status 與 error 判斷保存成功，不以 run_id 是否存在判斷。Report 保留全部 members，不用 await 或再次讀 snapshot。Readonly snapshot 的 confirmed run_id 仍只表示已有持久輸出成功。

## Recipe 與 execution

`recipe_api.py` 是作者的公開入口。`@recipe(name, description)` 宣告
`async def (session: RecipeSession, inputs: Inputs) -> Report`，
Inputs 與 Report 是作者選擇的 `BaseModel`，可分別繼承 `RecipeInputs` 與 `RecipeReport`。
框架推導 inputs schema，不另維護手寫 schema、SummaryParameter 或 SummaryEstimate 宣告。
Recipe inputs 只驗 JSON schema 與 pydantic，不渲染 GUI Form。
Adapter tab 的 Form 仍由 GUI owner 產生與驗證。

`session.open_tab(adapter, target=..., reuse=...)` 開 tab 或驗證 reuse，再 Reset 捕捉來源。
target 是 alias 到 concrete target 的 mapping，例如 `{"qubit": inputs.target}`。
省略、KEEP 或 null 的 mapping 項目保留全域 alias，只對這次 Reset 生效。
reuse 是 idle matching tab ID；省略、KEEP 或 null 開新 tab。
Recipe 不自動選重用 tab，也不自動關閉 tab；關閉用 `tab_close`。

`tab.read_parameter(path, missing_ok=False)` 讀 tab 固定 Open／Reset 的參數，例如 `$qubit.freq`。
缺葉拋 `GuiRpcError(reason="parameter_missing")`；missing_ok 只放行這種缺值。
明確 null 回 None，來源故障與格式錯誤仍拋出。
`session.device_unit(name)` 讀裝置快取的 flux 單位，不讀硬體或更新快取。
它保留 owner 的 A、V、native 或 none 字串。

`tab.read(path)` 使用 dotted string，例如 `"tones.2.drive"`，回 detached `FormField`。
Structured value 保留 owner 的 `YamlValue` mapping，需要 model 時由作者 TypeAdapter 解碼。
Value_status 區分 available、missing、unavailable；available null 是有效觀察，不證明整份 cfg 有效。

寫入路徑是非空 dotted string，例如 `"tones.2.sweep.start"`。
十進位段是完整 list 的原始位置，不按名稱或 participants 順序重編。
空段、負索引與 tuple 寫入路徑會被拒絕。
`tab.set(path, value)` 原樣替換欄位，不選來源或執行 converter。
`tab.select(path, option)` 選 owner 的精確 option label 或 captured module key。
`"Inline"` 或 `"Inline: const"` 等選項沿 owner 的同名同型繼承政策切換。
`tab.command(path, "to_inline")` 複製固定來源並保留 child keys。Owner 在入口判斷 selector，已 inline 時成功、不改草稿，也不發通知。這不依賴 Inline 選項宣告 command；GUI 只在 ref 選項列出 Convert to inline。
未知、歧義、readonly、來源與 ref 祖先錯誤立即拋出，MCP 不重建 Form 內部資料。

set／select 收到 KEEP 就跳過 mutation。None 原樣交 Form，表示清空／關閉 nullable 欄位。
非 nullable selector 收 None 會被 owner 拒絕。
每次操作可保留結構合法但尚未完整的草稿。
`tab.check_ready()` 唯讀驗證完整 Form 與 lowered cfg，不 Run、不捕捉新來源。
無效時拋 `RecipeNeedsParameters`，保留全部 owner path 與 reason。
先前編輯不回滾，Run admission 仍會再次驗證。

Lab 新增 `mux_onetone`、`onetone_range_find`、`twotone_flux_dep` 三個工具，
由注入的 `RECIPES` 產生 schema 與註冊，不在 framework 硬編工具表。
它們分別操作 `mux/onetone`、`onetone/range_find`、`twotone/flux_dep`。
用這些 adapter 名呼叫 `recipe_guide` 可讀各自的參數、分析與 writeback 說明。
三者都先 Run，再 Analyze，經問答接受選定候選後才寫入，最後保存。
Range find 的 `min_separation` 透過同一 tracked Analyze 的 `updates` 傳給 owner。
它使用 MHz，預設 1 MHz，必須有限且大於零，只影響 dip 選取，不改 acquisition grid。

作者用 `await tab.run()`、`await tab.analyze("primary"|"post", result=T)`、
`await tab.ask_writeback()` 串接流程。
Run 回 `RunResult`，不自動保存。
Analyze 用 TypeAdapter 將 owner summary 驗證為 T，不補欄位或修資料，也不自動保存或 preview。
`interactive=True` 明確開啟互動交接，公開 `PendingAnalysis(tab, stage, op)`。
Status／wait 提前回 `status="interactive"`，GUI 完成同一 operation 後原 await 繼續，不建第二個 runner。

`ask_writeback()` 捕捉 completed stages 的候選，再等待經 evidence 驗證的 `WritebackAnswer`。
回答不寫入。
只有 `tab.accept_writeback(decision.items)` 寫入選定 IDs 對應的 owner 當下 draft 值與目的地。
Unknown、重複或經作者修改的 captured 項目在第一筆寫入前拒絕。
Recipe 將選定 IDs 一次交給 `tab.writeback_accept`，不帶 `expected_revision`，preview 也不回 revision。
Owner 在同一提交核對 Primary／Post 全部選定 IDs。問答期間的 draft 編輯或替換不觸發 stale 拒絕，也不還原舊 proposal。
這沿用 GUI／MCP 的所有權模型，見 ADR-0075 的 2026-10-23 使用者裁定。
單 pane L1 `tab.writeback_write` 仍編輯並寫入當前 draft，不供 recipe captured acceptance 使用。
Primary 先於 Post，首錯停止，成功前綴保留，不 retry 或 rollback。
`WritebackReceipt.items` 是 owner 確認成功 ID 對應的原捕捉候選，不是寫入值快照。
實際寫入值與目的地取 owner 當下 draft。Receipt 以 pane 與 owner item ID 關聯 confirmed items，不拿 alias 名稱比 concrete target。
Proposal、receipt 與 full details 的 structured mappings 各自分離，不共用 retained acceptance evidence。
作者原地修改 proposal 的 nested value 後接受，會在任何 owner write 前拒絕。
作者可檢查收據或呼叫 `receipt.raise_for_error()`，不可將 decision.accepted 當作寫入成功。
Full details 的 writeback_receipts 保留每次接受的確認項目與錯誤，失敗後的 finally Save 仍執行。
`WritebackDestination.path` 是 owner 原樣的 opaque actual_path，
label 是 preview 當時 active context 的顯示標籤，不是 captured identity。

`tab.save()` 只同步呼叫一次 `tab.save_artifacts(artifacts="all")`，
保存 raw data、Labber 與全部已完成階段的 JSON／圖，每種輸出各一次。
RPC 在 GUI owner 同步完成，沒有 Save operation 或 handle 要 await。
個別輸出失敗沿 owner report 回報，不阻止其他輸出。
Transport 失敗會傳遞錯誤，不宣稱任何未收到的 raw 收據。
每次 Save 的完整 owner report 記在 runs，不累積跨次成功宣稱。
同 run 的 native 成員可覆寫，Labber 每次另產新檔。
每個 output 的 status／error 才是保存結果，不能只看 run_id。
通常在 return 前以 finally Save 一次；想先保住 raw 也可在 Run 後另存。
Run 未成功取得來源前不要進 Save finalizer，避免 Save 蓋掉原錯誤。

### 共用輸入與自訂步驟

`RecipeInputs` 提供 target、reuse_tab_id 與 overrides，實驗輸入模型加自己的欄位。
可省略的 Form 欄位使用 `T | None | Keep = KEEP`。
JSON schema 不列 Keep、不輸出 KEEP default，欄位可省略。
描述區分省略保留與 null 清空／關閉。數字仍拒絕非有限值。
target／reuse_tab_id 是 Open／Reset 控制，不是 Form 欄位，null 語意依上節。

`FieldEdit(path=..., value=...)` 只攜帶 dotted path 與原生值，供 overrides 使用。
JSON object 保留 expression tag、structured sweep 與巢狀內容，直接交 owner 驗證。
set 不接受以 object 替換 generated selector，選來源仍用 select。
overrides 省略為空 list，不接受 null。主體在實驗值之後依序呼叫 set，
重複路徑不去重，null 也會套用。全部操作完成後才 check_ready：

```python
from zcu_tools.mcp.measure.recipe_api import (
    KEEP, Keep, RecipeInputs, RecipeReport, RecipeSession, recipe,
)


class PulseInputs(RecipeInputs):
    gain: float | None | Keep = KEEP


@recipe("synthetic_pulse", "Set a pulse gain and acquire the current form")
async def synthetic_pulse(session: RecipeSession, inputs: PulseInputs) -> RecipeReport:
    tab = session.open_tab(
        "synthetic", target={"qubit": inputs.target}, reuse=inputs.reuse_tab_id,
    )
    if not isinstance(inputs.gain, Keep):
        tab.command("modules.pulse", "to_inline")
    tab.set("modules.pulse.gain", inputs.gain)
    for edit in inputs.overrides:
        tab.set(edit.path, edit.value)
    tab.check_ready()
    await tab.run()
    try:
        return RecipeReport()
    finally:
        tab.save()
```

來源選擇、編輯、驗證與 Run 都由主體明寫。分析、問答、寫回與保存仍是各自獨立的操作。
明填 scalar 前直接 command to_inline，再 set。已 inline 時 command 不改值。
單 kind module 與巢狀 waveform 同規則，不以選 Inline 或固定 const 改掉原 family。
全部 scalar 省略時不切來源，來源故障不 fallback。
Center／span 必須成對提供或省略，缺一個就回 needs_parameters 指名另一欄。
不從現有 endpoints 或校準推算，null 原樣交 owner、不參與算術。

`RecipeReport` 提供 writeback_applied、analysis_failure、fit_quality 與 diagnostics。
作者填入 owner 的事實，基底不決定 fit 是否可接受，也不自動寫回。
`recipe_api.FitQuality` 與 `QualityIssue` 直接匯出 analysis owner 的同一型別。
`await tab.analyze("primary", result=OwnerSummary)` 回傳的 result 使用 owner 型別，recipe 不另宣告鏡像。
`report_number(value)` 保留有限數的原始精度，None、NaN 與 infinity 轉為 None，不換單位或四捨五入。

### 取消與缺參

`cancel(op=...|execution=...|run_id=..., scope=..., expected=None)` 恰填一個 target。
Execution 指本地 recipe 或 standalone analysis，run_id 指 native workflow，op 指 GUI operation handle。
省略 scope 停整個目標，recipe／workflow 根分別可明填 recipe／workflow。細層選擇器只供 recipe／workflow 使用。

| Scope | 效果 |
| --- | --- |
| acquire | 只停一次取得，交付 partial，同 Schedule 的下一次 acquire 正常。 |
| schedule | 停該 Schedule 及其後續 acquire，下一個 operation 正常。 |
| iteration | 跳過目前這一點。作者釋放本趟後，下一趟 iteration 正常。 |
| run | 停 operation。Recipe 限進行中的 tab.run，workflow 可停任何 operation。 |
| operation | 選本地 operation，包括 recipe analysis，不取消父 scope。 |

Recipe 的 run／schedule／acquire 在沒有進行中 tab.run 時回
`cancelled=False, reason="no run in progress"`。Ask 不會被細層取消回答或關閉。
Iteration.cancelled 只退出目前 point，主 for 用 continue，point helper 用 return。
Point 在 try/finally 以同步 commit 保存一次 record，不在 finally await；根 ScriptCancelled 繼續向上傳播。

Status 的 active_path 是根到第一個 fork 的單一鏈。
Live_paths 是所有存活葉的完整路徑 list[str]，例如：

```json
{
  "active_path": "recipe:calibrate/operation#1",
  "live_paths": [
    "recipe:calibrate/operation#1/schedule#1/acquire#1",
    "recipe:calibrate/operation#1/schedule#2/acquire#1"
  ]
}
```

Selector 搜整棵存活子樹，支援 kind、kind#N、operation:<name> 與 live_paths 的完整路徑精確比對。
多筆命中回 cancelled=False、reason="ambiguous"，candidates 列完整候選路徑。
同 parent 的同名 operation 用 operation:<name>#N 解歧。不同 parent 各有 acquire#1 時，
acquire#1 仍 ambiguous，改用完整路徑。不以最新 child 或 callable 反射選對象。
Name 由 caller 提供，非 None 時必須非空且不含 #、/、:。

`scope="acquire#3"` 與 `scope="acquire", expected=3` 等價，兩者同填須一致。
序號必須為正整數，bool 不算整數。Root 沒有序號。合法但未存活回 scope not active，
序號不符回 sequence mismatch，均不設 token；壞語法回 invalid_params。

GUI root 的序號可在下一 Run 重用。跨 Run 精確取消要用
`cancel(op=17, scope="acquire", expected=3)`，op 必須來自觀察到的 Run。
細層 op target 必須仍是目前 tab.run handle，舊 op 回 scope not active，不改打新 Run。
Generic cancel(op=17) 維持整個 script Stop 語意。

Ack 保留 execution／workflow snapshot，另帶 cancelled、reason、active_path 與 candidates。
Cancelled=True 表示 owner 已接受命令，不證明 operation 或 cleanup 已 drain，終局仍讀 wait/status。
Workflow 根取消且 expected=None 時，Stop owner lock 先寫 StopRequested／manifest 並 latch，再由 host.cancel 原子設 root，持久化不進 control callback。
這個分支的 accepted ack 表示 owner 已接受並記錄 Stop；host 若仍活則同時設 token。
Host 已退出但 owner 尚未 finalization 時，接受 Stop 仍可使終態為 cancelled。
其他 selector 直接由 host 核對。Terminal workflow 對合法 scope 回 scope not active，錯誤語法仍回 invalid_params。

根 recipe／workflow 取消在下一個 await 丟共用 ScriptCancelled，普通 Exception 子類，
不是 asyncio.CancelledError。進行中的 Run 先合作 drain、保留 run_id／partial／completion facts，再讓 await raise。
作者 finally 在同一 worker 照跑，同步 Save 可保留已取得資料；不在 finally 開新的 await。
Recipe 終態為 `RecipeOutcome(status="cancelled", report=None, runs=[...])`。
ScriptCancelled 不記入 error，真實 owner、作者或 cleanup 例外仍保留診斷。
Completed 不代表所有輸出已保存。

Analyze 的預期擬合失敗回 `AnalysisResult(result=None, status="failed", reason=...)`。
只有 analyze work 明確丟 AnalysisFailed 才轉成這個值。任意程式錯誤、record 或 writeback 失敗仍 raise。
作者看 status 後自行保存、跳過、重試或停止，不需要為預期 fit 失敗 raise。
成功 result 按作者提供的 T 驗證，缺欄位與型別不合仍 raise。
已分類的數值路徑見 [fitting README](../../analysis/fitting/README.md)。

Run 前的 `RecipeNeedsParameters` 使 execution 終態為 needs_parameters，
snapshot 列具名 missing paths、outcome=None，並釋放 admission。
Cfg preflight 額外回 `parameter_errors: [{path, message}]`，保留 owner 的全部 typed paths 與理由，包含未參與的 tone。
有作者說明的必填空白 errors message 包含欄位與父欄位的 label、unit 及 description。
回覆時分開報告 readout slot found 與 form valid：slot lookup 只證明來源存在，`status.ready.can_run` 只檢查環境就緒；目標表單是否有效要另讀 `tab.get_cfg.errors`，recipe preflight 則讀 `parameter_errors`，不新增重複的 readiness 欄位。
Missing 的可讀路徑如 `tones[5].sweep.stop`，parameter_errors 的同一路徑為 `["tones", 5, "sweep", "stop"]`。
`await tab.run()` 也先讀完整 Form errors，有錯就回 needs_parameters，不送 run_start。
RecipeOutcome 仍只有 completed／failed／cancelled，不把缺參包成失敗。

### Runtime 與回覆

`script_recipe_execution.py` 使用 production `experiment.script_host.ScriptHost`。
Recipe 的本地樹與 GUI Run 樹分屬兩個 process。Schedule／acquire 經
`operation.cancel_scope` 以目前 tab.run 的 operation handle 代理，GUI 原子核對自己的樹。
Status 每次用 `operation.status_scope` 取得尾段，接在本地 path 後，不重列 GUI run root。
拒絕回覆的 cancelled/reason 保留 GUI 事實，active_path、live_paths 與候選 candidates 使用相同合成規則。
這不是跨 process 原子快照，晚到請求靠 handle 與序號拒絕，不取消下一個 Run。
GUI `result_state.partial_result` 交付 Run 的完成事實，即使沒有 native metadata 也保留。
Recipe RunResult 讀這一欄，不因 GUI operation 仍為 finished 而遺失內層取消。
`recipe_runtime.py` 擁有 GUI binding、wire 解碼、合作取消與 operation source；
`recipe_host.py` 組合 scheduler、runtime 與 author session。
`recipe_execution.py` 管理注入 declarations、單一 recipe admission 與 execution 查詢。
Recipe 不建隱式 iteration，路徑直接由 recipe:<name> 掛 operation#N。

Host loop 只在 execution worker。作者 task 不被 cancel，阻塞操作由 executor 合作 drain。
每個 operation 都有顯式 Scope child，底層讀 scope.stop。根已停止時不再開始新的 await operation。
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

Terminal recipe summary 回 execution／status／report／runs。Status 另帶 active_path 與結構化 live_paths。
Failed 另帶 `failure: {phase, reason}`，full detail 也保留同一 failure 與原 error。
Phase 在失敗動作處記錄：作者／Run／開 tab 為 operation，分析結果讀取與解碼為 result_read，
Save 為 image_save，preview 為 figure_read，writeback 候選讀取為 writeback_read。
Post 的 `result_read` 失敗另附 `recovery`，保留 owner 的唯讀 RPC、原生參數、stage 與當時 MCP `op`。
先用 `tab.snapshot` 核對 `post_analysis_state.source_operation_id` 與 `recovery.params.operation_id`。
相符後以 `rpc_call(method=recovery.rpc, params=recovery.params)` 讀回，不重跑分析。
不相符表示結果已被替換，回報 superseded，不取用新結果代替舊結果。
即使 snapshot 後發生替換，source-bound RPC 仍以 `result_superseded` 拒絕。
Reason 是 `str(exception)`，不附格式化 traceback。作者 catch 後的新錯誤不沿用舊 phase，
成功的 finally Save 也不覆蓋原失敗的 phase。
Report 由作者定義，runs 記每次 Run 的 tab、adapter、run_id、operation status、
partial 與 Save 收據，不靠作者補取消報告。
非終態另帶 execution 與明確的 pending question／analysis。
`status(execution, detail="full")` 保留 detached Run 前 Form 條件、
各 stage 的 owner-native summary／invalid、最後明確捕捉的 writeback proposal、各次 writeback_receipts 與 error。
Scalar、uncertainty、unit 與元素 identity 都沿 owner summary，不重算或套 reporting 宣告。
Status 的 active_path 會讀 GUI 當前 scope，其餘 captured detail 不重讀 GUI，也不自動存檔。

全域 status 的 entry 是 null 或 GUI entry.info 的 UUID、name、result_path、database_path，
只讀 metadata，不建立 context observation。
`ready.can_run` 只表示環境 readiness。它檢查 entry、active context、SoC 與至少一張 tab 是否存在，缺項列在 `ready.missing`。未 open entry 時 missing 包含 `entry`。它不驗證任何 tab 的表單，也不保證 holder、busy 或硬體 guard 會允許 Run。

即使 `ready.can_run=true` 且 `ready.missing=[]`，表單仍可能缺必填值。Run 前用 `rpc_call(method="tab.get_cfg", params={"tab_id": "<tab id>"})` 讀目標 tab 的 `input`、`values` 與 `errors`。依 errors 的 path 補值，再讀一次確認；不要用環境 readiness 取代表單 validity。`tab.describe_cfg` 用來逐層查欄位能力。兩個入口另回全表 `diagnostics`，不受目前 tone picker 或 describe path 限制。每列有 canonical `path`、目前的 `message`、原始 `name`／`index`、`seed_source`、`seed_status` 與 `uncalibrated`。`message=null` 表示該列沒有目前錯誤。`seed_status` 的 missing／available／literal 描述 Open／Reset 的初值來源，之後補值不改寫這項來源事實。`uncalibrated` 只讀作者宣告，true 表示未校準候選，false 表示 measured，未知為 null。缺值不等於未校準。Describe 的局部 `errors` 仍只含 Form preview errors，全表 diagnostics 也包含 lowered cfg errors。Run admission 仍會檢查目前表單與動態條件。

全域 status 列 GUI operations 與本 MCP session 非終態 executions。
`terminal_count` 計算所有本地終態 execution，包含 standalone analysis。
`recent_terminal_executions` 只列最近 10 筆終態 recipe，依 finished_at 新至舊排列。
每筆含 execution、recipe、status、finished_at；finished_at 是發布終態的 UTC ISO-8601 時間，
包含 completed、failed、cancelled 與 needs_parameters。索引不含 standalone analysis。
超出最近 10 筆的 recipe 仍可按 ID 查詢；全域回覆保留 query_hint。
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
| `workflow_pause/resume/stop` | 明填 native `run_id`。Pause 保留 lease，Resume 重用原 run 與 handle，Stop 維持 script scope，等 terminal delivery。 |
| `cancel` | 明填 `run_id`，可選 scope 與 expected，經 `workflow.cancel` 核對 native active_path，不另建 workflow execution。 |
| `workflow_form` | `tab_id/form`，form 是 plan 或 tunables。可選 batch 只 preview，回 input/values/errors。 |
| `workflow_describe_form` | 同上另給 typed `path`，root 是 []。只加同一 candidate 的單層 metadata，不回整棵 schema。 |
| `workflow_edit_form` | `tab_id/form/batch`，修改 owner 的 shared draft，回更新後觀察。 |
| `workflow_tunables` | `run_id/batch`，經 owner Apply，回 native revision/values。空 batch 提交當前 draft。 |
| `workflow_records` | `run_id` 與可選 `since=0`，回 commit_seq 嚴格大於 since 的 native records。 |
| `workflow_figures` | `run_id`，只回 names，不 render。 |
| `workflow_figure` | `run_id/name`，回識別與 validated PNG image content，text 不保留 base64。 |
| `workflow_answer` | `run_id/request_id/answer`，request_id 必須是原題 `pending_question.request_id`，answer 符合其 answer_schema。回 Start 的同一 pending handle，不宣稱已完成。 |

Batch 是 `[{"path": ["field", 0, "leaf"], "value": ...}]`，路徑是 canonical
欄位名與原始 list index。Form owner 決定 branch、expression、有效值與錯誤。
MCP 不計算 defaults、不做 lowering，也不另建 Engine、run registry 或 holder 規則。

Start 省略/null batch 使用 shared GUI draft。明填 batch，包括 []，使用 owner
當次 fresh captured defaults 加 edits。Form preview 的 [] 則讀當前 parsed draft，
不重算 defaults。Tunables Apply 的 [] 提交當前 draft，不把它重設為 defaults。

工具表是固定手寫，沿 live catalog 的 timeout、權限及 session operation mapping。
錯誤保留 code/message/reason/data，不重試、不回滾。Readonly 可查詢與 preview，
mutation 仍受 holder、busy 與 RUN lease 限制。Terminal records/figures 可繼續讀取。
Status 的 `pending_question` 是固定
`{request_id, question_type, question, answer_schema}` envelope。只有 question 含
作者欄位，沒有問題時為 null。原題 request_id 必須隨 answer 回傳。Engine 先比對
ID，再驗 schema。錯 ID 或無 pending 問題回 `precondition_failed/question_mismatch`，
data 帶目前 pending_request_id 或 null，不改 lifecycle、不記 accepted answer。
錯欄位回 `invalid_params/answer_validation`，問題保留供重試。

`last_exchange` 保留最近問題、已接受的 answer 或 null，以及
waiting/submitted/answered/cancelled 狀態。它與 GUI 共用 headless owner。
Submitted 不表示已接受，錯型答案不記入 answer。整個 coroutine 留在同一 worker，
Pause、Ask、Resume 與 Answer 都不新增 worker segment。Stop 在該 worker 交 None
並執行清理，terminal owner delivery 才釋放 RUN。

Figures 與 GUI pane 使用相同投影。Workflow 圖名維持 `<name>`，目前 operation 圖名為 `operation/<name>`。新 attempt 整組替換 operation 圖，terminal 保留最新一組。Workflow 層登記圖名時拒絕 `operation/` 前綴。

Records 依 commit_seq 查詢，包含配置 directory、committed_at、實際讀過的 revisions、
record_type 與 operations。配置目錄順序不等於 commit 順序。Operation facts 包含
kind、iteration 內 attempt、started_at/finished_at、run 的呼叫前 device_snapshot、
outcome、reason 與 stopped，另有一基 iteration。這與作者 Completed.facts 使用同一
OperationFacts 結構。這些是 framework 事實，不代表作者已保存所有原始量測檔案。
Running 或 paused 都可 Apply；revision 在下一次 tun.current 讀取生效，不追改當前
operation 的 cfg。Pause 打斷的 stale partial 在 Resume 後重試原 request，完整結果
只等 Resume 交付。Status 仍用公開讀取／polling，不新增 lifecycle subscribe。

## 分析、互動與寫回

`tab_analyze` 在既有 Run／Load 上啟動 Primary 或 Post，沒有 recipe 身分。
它不補存 raw，raw_data 的 run_id／saved 沿 GUI owner 的當次觀察。
run_id 在保存前已存在，saved 只在 measurement_data 全部保存時為 true；
尚未捕捉到觀察時 raw_data 為 null。
這與 analysis artifact 保存分開，status 不重查目前 tab。

Operation-bound analysis 回覆附 `presentation`，記錄來源 `run_id`、
`stage`（primary／post）、目前 pane 的 `figure_names` 與 `figures_replaced`。
Standalone summary 放在 `analysis.<stage>.presentation`，full 放在
`result.presentation`；recipe full 放在 `details[].analyses[].presentation`。
`figures_replaced=true` 表示同 Run、同 pane 原有非空圖集合已被這次分析替換，
舊圖不能再從該 pane 取得。它不表示任何 Save 檔案已覆寫。
需要比較模型時，先另存舊圖或保留當次 MCP image，再提交新分析。
新 Run／Load 清除這個 pane 歷史；Primary 更新也使舊 Post 失效。
缺來源 metadata 時 run_id 為 null，不拿目前工作點補歷史。
Mux qubit summary 保留完整原始位置與 Q 名，例如 Q3／Q5 專掃仍為
`position=2,name=Q3` 與 `position=4,name=Q5`，不把參與順序重編成 0／1。
圖名仍由實驗定義，例如 `fit_2`；framework 不從圖名推測 Q 身分。

Standalone analysis execution 等原 GUI operation 完成，讀結果、保存 analysis artifacts，
再以原 opaque operation 讀一次 writeback preview。
沒有圖也讀候選，Read 失敗保留已有 result、當次 Save report 與 image，回 writeback_read error。
來源 superseded 不改讀新 pane，不 retry。
Post start receipt 的 `result_read_timeout_seconds` 來自 owner，以 5 秒加每個 measured tone 5 秒計算，上限 60 秒。
Standalone 與 recipe 使用同一欄位，缺欄時用 5 秒。GUI Post read handler 的 ceiling 是 60 秒。
Post 結果讀取失敗時，execution 回覆保留同一份 `recovery`；依上方 Runtime 的 snapshot 與 source-bound RPC 步驟讀回。
這個自動保存是 standalone 工具契約，不適用 recipe 的 `await tab.analyze()`；
recipe 作者另外明確 `tab.save()`。

`RecipeTab.analyze(stage, result=T, updates=...)` 接受 `Mapping[str, YamlValue | Keep]`。
Recipe 直接傳 dict，例如 `updates={"dual_exp": inputs.dual_exp, "skip": inputs.skip}`。
Key 是 adapter 的具名分析參數，value 使用該欄位的原生單位，不求值 Form Expr 或換算單位。
RecipeTab 送 owner 前略過 KEEP 項，明填 null 原樣傳給 owner。
省略、空 mapping 或全 KEEP 都保留 owner 參數。
Owner 在同一個 tracked Primary／Post operation 驗證並套用更新，不先開另一個分析 operation。
`interactive=True` 仍交付 PendingAnalysis，由 `tab_interact` 的 done 完成原 operation，恢復原 recipe await。
`answer` 只回答 recipe 明確提出的問題，不控制 picker，也不直接寫回。
未知參數、型別不符與 owner admission 錯誤直接回報，不重試。

`tab_interact` 不帶 payload 時讀 committed state、commands、info、preview_active 與圖，
不改焦點。帶 payload 時 GUI 驗證命令並跟隨 Analysis pane。
`done` 加入原 operation 的完成，不另建 analysis runner。
Recipe 的 done 跳過剛完成 operation 的舊 interactive 快照，回原 recipe 的下一個 handoff 或終局；standalone 則完成自己的 execution。
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
每個實際提交的 pane 都先向 owner 取得一次性 ID。`submissions` 以 primary／post 對應 ID，
`submission_id` 是最後嘗試的 pane，`recovery` 帶 `tab.writeback_receipt` 與精確 query params。
寫入逾時時 status 是 unknown，保留 `failed_stage_may_have_partial_writes`，
並要求先查狀態，再決定是否重送。不能把 handler 逾時當成寫入失敗。
查詢回完成列的 item ID、實際值、concrete target 與固定 entry_id／point。
Receipt 只保留在 GUI 生命週期內，查詢不重新讀取當前 draft 或目的地。
Recipe `accept_writeback` 的 receipt 也保留 submission_id／recovery，逾時回 unknown。
它的 items 仍是 confirmed IDs 對應的原捕捉候選，不取代 owner 的實際值 receipt。

Recipe 的 captured 問答與這個當前 draft 工具不同。
Recipe 只在 `accept_writeback(items)` 以已捕捉候選的 IDs 寫入當下 draft，
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
參數、Library 與 aliases 的日常操作依上方「Point 檔案讀寫」直接讀檔、編輯並 reload。
需要 owner 命令時，既有 context.ml_edit、context.aliases_set、context.set_parameter
與 value.list／value.read 仍經 rpc_call 可達，不開 Inspect、不切頁，也不改既有 Form／Run 的 capture。
Aliases 在 Inspect 第三頁。
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
