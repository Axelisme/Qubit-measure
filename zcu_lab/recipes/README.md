# MCP recipes

**讀者：** lab-author、agent。查找 recipe 宣告、操作順序與作者契約。

**Last updated:** 2026-10-23

Recipe 是操作 GUI tab 的 async 函式。Adapter 擁有 cfg、表單、量測與分析。
Recipe 選來源、設參數，再決定 Run、Analyze 與 writeback 問答的順序。

## 宣告與註冊

[__init__.py](__init__.py) 的 `RECIPES` 明列 29 個宣告，不掃描目錄。
每個模組用 `@recipe(name, description)` 裝飾
`async def (session: RecipeSession, inputs: Inputs) -> Report`。
`Inputs` 繼承 `FormModel`，逐欄寫單位、預設與 null 政策。
框架從這個 model 產生 MCP JSON schema，不另維護手寫 schema。
Recipe inputs 只作 JSON schema／pydantic 驗證，不渲染 GUI Form。
`Report` 是作者選擇的具名結果 model，包含此流程要回報的量與 writeback 決定。
各模組的 `DEFINITION` 指向裝飾後的宣告。組合根把 `RECIPES` 注入框架。
Import 不開 GUI、不註冊 adapter，也不執行量測。新增宣告後須重啟 MCP server。

`helpers.py` 擁有 frequency、flux、校準來源與數值回報的 lab 政策。
`mux_helpers.py` 擁有 mux 的 library、inline 輸入與 tone report 共用欄位。
它們不是獨立 recipe。參數推導不放進 framework。

## 作者可呼叫的操作

`session.open_tab(adapter, aliases=..., reuse=...)` 開啟或 Reset 一個 tab。
Alias override 只供這次固定 capture，不更改全域 aliases 或 GUI focus。
`session.context.read(source_path)` 讀 session 的值，不需要先開 tab。
`tab.read(path)` 則讀該 tab 已捕捉的 Form 來源。

Form 路徑是 tuple，字串指欄位，整數指完整清單中的原始位置。
`tab.set`、`tab.select`、`tab.apply` 與 `tab.call` 同步完成。
`set(path, None)` 保留 GUI input，不代表停用。
準備模組用字串 `"off"` 明確停用，null 會被 inputs model 拒絕。
缺少必要校準時，在 Run 前 raise `RecipeNeedsParameters`。
`tab.apply` 一次提交完整有序 batch。Mux recipe 在最後一批傳 `require_valid=True`，
owner 驗證完整候選 cfg 後才安裝。缺值回 needs_parameters 與全部 indexed path／理由，
本批欄位與 Run 都不執行。先前開 tab、Reset、library 與 inline 操作不回滾。
Lookback Form 隱藏 reset 與 init_pulse，兩個 recipe 輸入只接受 `"off"`，不存取其 selector。

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

`FitSummary` 是作者宣告的 `BaseModel`，欄位對應 adapter 的 owner summary。
框架用 `TypeAdapter` 驗證 summary，不把 GUI 內部 analysis dataclass 填回來。
欄位缺失或型別不合會失敗，不補值。沒有 scalar 的 figure-only 分析可用空的具名 view。
`report_number` 只把不可估計或非有限的 fit scalar 轉成 null，不改擬合或 writeback 政策。

預期擬合失敗回 `AnalysisResult(result=None, status="failed", reason=...)`。
範例的 Report 宣告 nullable value、writeback_applied 與 nullable analysis_failure。
有分析的 24 支 recipe 都在每次 Analyze 後檢查 result。
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
| `mux_onetone` | `mux/onetone` | `participants` 收 resonator 名稱，依固定 `$mux.resonators` 映射原始 index。`df:{start,stop}` 是共同 MHz offset，`n` 是共用點數。回覆每 tone 的 index、name、freq、kappa、fit_quality 與 identity 診斷。 |
| `onetone_range_find` | `onetone/range_find` | 必填絕對 MHz `freq:{start,stop}` 與 `n`，可選 `resonators` 名單。回覆每 dip 的 name、freq、depth 與 `identity_diagnostic`。`min_separation` 是有限正數 MHz，預設 1 MHz，經同一 Analyze operation 更新 owner 的凹陷間距，不改 acquisition。 |
| `twotone_flux_dep` | `twotone/flux_dep` | 必填 drive 的 `center_mhz`、`span_mhz`、`freq_points` 及 native flux device、unit、端點、點數。互動 picker 完成後詢問寫回 qubit flux 校準。 |

三者依序 Run、Analyze、ask_writeback，只接受回答選定的候選，最後 Save。
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
