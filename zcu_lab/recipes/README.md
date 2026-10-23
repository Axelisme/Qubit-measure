# MCP recipes

**讀者：** lab-author、agent。查找 recipe 宣告、操作順序與作者契約。

**Last updated:** 2026-10-24

Recipe 是操作 GUI tab 的 async 函式。Adapter 擁有 cfg、表單、量測與分析。
Recipe 選來源、設參數，再決定 Run、Analyze 與 writeback 問答的順序。

## 宣告與註冊

[__init__.py](__init__.py) 的 `RECIPES` 明列 29 個宣告，不掃描目錄。
每個模組用 `@recipe(name, description)` 裝飾
`async def (session: RecipeSession, inputs: Inputs) -> Report`。
每個 recipe 的 `Inputs` 繼承 `RecipeInputs`，只加實驗欄位，逐欄寫單位、預設與 null 政策。
`RecipeInputs` 提供 target、reuse_tab_id 與 overrides，不提供通用 library／inline 輸入。
可省略的實驗欄位宣告為 `T | None | Keep = KEEP`。省略保留 Form 值，明填 null 原樣交 owner。
KEEP 是 Python 省略標記，不是 JSON 值。Schema 隱藏 Keep 分支與 KEEP default，欄位不列 required。
Mux 的 participants 使用完整 tones 清單的原始 index，逐 tone 範圍清單保留原始順序，見下方 mux guide。
框架從這個 model 產生 MCP JSON schema，不另維護手寫 schema。
Recipe inputs 只作 JSON schema／pydantic 驗證，不渲染 GUI Form。
每個 recipe 的 `Report` 繼承 `RecipeReport`，只加實驗的回報量。
基底提供 writeback_applied、analysis_failure、fit_quality 與 diagnostics，作者明填 owner 事實。
各模組的 `DEFINITION` 指向裝飾後的宣告。組合根把 `RECIPES` 注入框架。
Import 不開 GUI、不註冊 adapter，也不執行量測。新增宣告後須重啟 MCP server。

掃描端點、flux 範圍與裝置單位檢查直接寫在 recipe 主體，不另設選源 helper。
數字回報使用 lib 的 `report_number`，不改單位或精度。
`mux_helpers.py` 只共用 tone 範圍輸入與報告欄位，不組裝 Form edits。
`mux_flux_helpers.py` 共用四個 flux recipe 的輸入、報告與逐 Q 分析 updates。
Updates helper 只把完整 tones 的原始位置與 Q 名對應到分析參數，不改量測設定或選來源。
每個 Q 都明送 `flux_half`／`flux_int`，未指定的線送 null，讓 owner 使用該 Q 的 Run capture。
這是避免沿用上一個 Q 顯式線的 reset 指令，不是把所有省略欄位轉成 null。

## 作者可呼叫的操作

`session.open_tab(adapter, target={"qubit": inputs.target}, reuse=inputs.reuse_tab_id)`
開啟或 Reset tab。Range find 的 mapping key 是 `resonator`，mux 是 `mux`。
Target mapping 的省略／KEEP／null 項目沿用全域 alias，只對這次 capture 生效。
Reuse 是 idle matching tab ID，省略／KEEP／null 開新 tab。不改全域 aliases 或 GUI focus。
`session.context.read(source_path)` 讀 session 值，不需要先開 tab。
`tab.read_parameter(path, missing_ok=False)` 讀 tab 固定捕捉的參數。
`session.device_unit(name)` 讀裝置快取的 flux 單位，不碰硬體或刷新快取。
`tab.read(path)` 回 `FormField`，仍用 typed tuple，例如 `("tones", 2, "drive")`。

寫入使用非空 dotted string，例如 `"tones.2.drive.gain"`，索引指完整清單的原始位置。
`tab.set(path, value)` 原樣替換欄位，不選來源、不自動 convert。
`tab.select(path, option)` 使用 owner 的精確 option label 或 captured module key。
單 kind 的 inline label 是 `"Inline"`，多 kind 是 `"Inline: <kind>"`。
`tab.command(path, "to_inline")` 複製固定來源，保留原 module／waveform family。
三者同步完成。Set／select 收到 KEEP 不發 mutation，None 原樣交 owner。
Nullable selector 用 None 關閉，非 nullable 由 owner 拒絕，不用字串 `"off"`。

明填 scalar 前先讀 selector.mode，只有 ref 才 command to_inline，已 inline 直接 set。
巢狀 waveform 也按同一規則，不以固定 const 或 select Inline 作可寫前置。
全部 scalar 省略時不切來源。來源壞掉不 fallback，讓 owner 回報路徑與原因。
Center／span 必須成對提供或成對省略，缺一個就回 needs_parameters 指名另一欄。
兩者省略不 set，不從現有 endpoints 或校準推算。Null 不參與算術，原樣交 owner。
完整 tone 範圍清單必須與完整 tones 等長，缺少或多出的原始位置都要回報。

`FieldEdit` 只含 dotted path 與 value。Value 與 set 同契約，object 與巢狀 list 保留內容，
Expr 與具名 native model 也可用。Owner 驗證欄位 schema，selector 選項仍用 select。
主體先設實驗值，再逐筆套 overrides，最後驗整份 Form／cfg：

```python
for edit in inputs.overrides:
    tab.set(edit.path, edit.value)
tab.check_ready()
```

Overrides 省略為空 list，不接受 null。重複路徑不去重，也不過濾 null。
每次操作允許尚未 ready 的合法草稿，結構、來源與 readonly 錯誤立即拒絕。
`check_ready` 唯讀、不 Run、不刷新 capture，缺值拋 `RecipeNeedsParameters` 並列全部 path／reason。
先前成功編輯不回滾。Run admission 仍會再次驗證。
Lookback 不提供 reset／init_pulse 輸入，也不存取這兩個隱藏 selector。
Analysis updates 只濾 KEEP，明填 null 原樣交分析 owner。

所有 lab async def 統一只 await `env.*`／`tab.*`／`session.*` 呼叫，
或靜態解析到 `zcu_lab` 內定義的 helper，定義可用 def 或 async def。
Recipe 通常用 `tab.*` 與 `session.ask(...)`。
Workflow 可寫 `await helper(env, cfg)`，helper 可再呼叫 helper，自己的 async def 本文也同樣受檢查。
不要 await 外部 coroutine。在 async def 內，`asyncio.*`、`create_task`、`gather`、
`wait_for`、`TaskGroup` 的直接呼叫即使不 await 也拒絕，包含 async with TaskGroup()。
其他 async with／async for 不另禁。Runtime 擁有 worker loop 與 operation 生命週期。
Quality gate 的 `tools/check_script_awaits.py` 全掃 async def，不識別 decorator，
不分 entry／helper，也不做可達性分析。
這不是 type checker；assignment aliases、動態分派與包裝可繞過規則，作者仍須遵守契約。
通常的順序如下：

```python
await tab.run()
try:
    analysis = await tab.analyze("primary", result=FitSummary)
    if analysis.result is None:
        return Report(
            value=None,
            writeback_applied=False,
            analysis_failure=analysis.reason,
        )
    decision = await tab.ask_writeback()
    writeback_applied = False
    if decision.accepted:
        receipt = tab.accept_writeback(decision.items)
        receipt.raise_for_error()
        writeback_applied = bool(receipt.items)
    return Report(
        value=analysis.result.value,
        writeback_applied=writeback_applied,
        analysis_failure=None,
    )
finally:
    tab.save()
```

`FitSummary` 應直接 import 分析 owner 的具名 summary，不在 recipe 重宣告鏡像。
例如 twotone/freq 匯出 `TwotoneSpectrumSummary`，figure-only MIST 直接用 owner 的空結果型別。
框架用 `TypeAdapter` 解碼 summary，欄位缺失或型別不合會失敗，不補值。
可照著操作的完整骨架見 [Recipe 撰寫](../v2/measure-authoring.md#recipe-撰寫)。
`report_number` 只把不可估計或非有限的 fit scalar 轉成 null，不改擬合或 writeback 政策。

預期擬合失敗回 `AnalysisResult(result=None, status="failed", reason=...)`。
範例的 Report 宣告 nullable value、writeback_applied 與 nullable analysis_failure。
有分析的 recipe 都在每次 Analyze 後檢查 result。
失敗時回報原因，不詢問或接受 writeback，也不跑相依的 Post。
成功時 analysis_failure 為 null。Primary 成功而 Post 失敗時，報告保留 Primary 的估計。
Mux 頂層分析失敗回空 tones；單 tone 失敗仍由 owner 保留該 tone 的原因，其他 tone 繼續。
Owner 的程式錯誤、record 與 postprocessing 錯誤仍向上傳播，不以空報告掩蓋。

`tab.save()` 同步保存 raw data 與全部已完成階段的 artifacts。
Analyze 不會自動保存。各 recipe 在已有 Run 結果後進入 `try/finally`，收尾只 save 一次。
缺參數或 Run 啟動失敗不進入這個 finalizer，避免無來源的 Save 蓋掉原錯誤。
每個輸出的成功與失敗由 `runs` 中的 Save 收據回報，完成不代表所有格式都保存成功。

## 問答與取消

`ask_writeback()` 捕捉當次候選，再等待回答。回答本身不寫入校準。
只有 `accept_writeback(decision.items)` 以選定候選 IDs 寫入 owner 當下 draft 的值與目的地。
略過時不寫入。Draft 的初始目的地來自 Run capture，後續 retarget 編輯以 draft 為準，不重新套用目前 aliases。
接受時 owner 一次核對 Primary／Post 全部選定 IDs，unknown／duplicate 在第一筆寫入前拒絕。
問答期間的 draft 編輯與替換會被接受，不還原舊 proposal，見 ADR-0075 的 2026-10-23 使用者裁定。
`accept_writeback` 回 `WritebackReceipt`，失敗收據保留已確認項目與錯誤。
Receipt.items 是成功 ID 對應的原捕捉候選，不是寫入值快照。實際值與目的地取 owner 當下 draft。
作者呼叫 `receipt.raise_for_error()` 將失敗交給 execution，`finally` 仍 Save。
報告的 `writeback_applied` 只依完成收據中的已確認項目，不依回答的 accepted。

Run scope cancel 只停止正在取得資料的 Run，回傳 partial，recipe 繼續後續分析。
取消 recipe 根則讓下一個 call／ask await raise `ScriptCancelled`。
進行中的 Run 先合作 drain，runtime 保留 run_id、partial 與完成事實，再 raise。
這是普通例外，不是 `asyncio.CancelledError`。
Analyze、ask_writeback、session.ask 都回非 Optional 物件，不用 None 表示根取消。
AnalysisResult.result 的 null 表示該分析沒有成功結果，與根取消的例外不同。
作者不 catch 根取消。已進入 try 的 `finally` 仍在原 worker 同步 Save。
本目錄的 `await tab.run()` 保持在 try 外，根取消在此 raise 時不進入 Save finalizer。
Drained partial 留在 tab，交由 agent 決定是否另行保存。
Runtime 回覆 `status="cancelled"`、`report=null` 與各次 `runs`。
其他錯誤仍保留失敗診斷，不重跑 acquisition。

## 操作文件與驗收

用 `recipe_guide(adapter="twotone/t1")` 依 adapter 名查 operator guide。
Recipe 可以串多個 adapter，宣告不另設單一 adapter mapping。
十二個 mux 的輸入、原始 tone 位置與報告見 [mux guide](mux-guide.md)。

新增流程：

| MCP tool | Adapter | 掃描與回報 |
| --- | --- | --- |
| `mux_onetone` | `mux/onetone` | `participants` 收完整 readout 的原始整數 index，依清單順序執行，不以名稱查位置。`df:{start,stop}` 是共同 MHz offset，`n` 是共用點數。回覆每 tone 的 index、name、freq、kappa、fit_quality 與 identity 診斷。 |
| `onetone_range_find` | `onetone/range_find` | 必填絕對 MHz `freq:{start,stop}` 與 `n`，可選 `resonators` 只作 Analyze 命名，不進 acquisition cfg。省略保留分析名單，null 原樣交分析 owner，空清單做無名探索。只有明填非空名單才詢問寫回。回覆每 dip 的可選 name、freq、depth 與 `identity_diagnostic`。`min_separation` 是有限正數 MHz，預設 1 MHz，經同一 Analyze operation 更新 owner 的凹陷間距，不改 acquisition。 |
| `twotone_flux_dep` | `twotone/flux_dep` | 必填 drive 的 `center_mhz`、`span_mhz`、`freq_points` 及 native flux device、unit、端點、點數。互動 picker 完成後詢問寫回 qubit flux 校準。 |

三者先 Run、Analyze，再依各 recipe 的條件詢問寫回，只接受回答選定的候選，最後 Save。
四個 `mux_{onetone,twotone}_flux_{global,local}` recipe 共用 flat cfg 路徑，
逐 participant 用原始 tone index 開啟 native picker，再分別詢問選定寫回。
`df/n`、`sweeps/n` 與 `flux/flux_n`、`flux_sweeps/flux_n` 的形狀及預選線範例見
[mux flux recipe guide](mux-guide.md#mux-flux-maps)。
`writeback_applied` 只在寫入收據確認成功項目後為 true。略過不寫入。
Framework 的執行、問答與取消契約見
[recipe 與 execution](../../lib/zcu_tools/mcp/measure/README.md#recipe-與-execution)。
新增 adapter 與流程的入口見 [實驗教學](../tutorial/03-experiment.md)
與 [recipe 教學](../tutorial/04-recipe.md)。

Lab 定義以直接 schema、guide、adapter forms-open 與離線 smoke 審閱。
Framework pytest 使用合成 recipe，不 import `zcu_lab`，也不固定這裡的實驗預設值。
