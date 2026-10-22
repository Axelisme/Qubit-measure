# MCP recipes

**讀者：** lab-author、agent。查找 recipe 宣告、操作順序與作者契約。

**Last updated:** 2026-10-22

Recipe 是操作 GUI tab 的 async 函式。Adapter 擁有 cfg、表單、量測與分析。
Recipe 選來源、設參數，再決定 Run、Analyze 與 writeback 問答的順序。

## 宣告與註冊

[__init__.py](__init__.py) 的 `RECIPES` 明列 22 個宣告，不掃描目錄。
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

所有 lab async def 統一只 await `env.*`／`tab.*`／`session.*` 呼叫，
或靜態解析到 `zcu_lab` async 定義的 helper。Recipe 通常用 `tab.*` 與 `session.ask(...)`。
Workflow 可寫 `await helper(env, cfg)`，helper 可再呼叫 helper，自己的本文也同樣受檢查。
不要 await 外部 coroutine。`asyncio.*`、`create_task`、`gather`、
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
    decision = await tab.ask_writeback()
    if decision.accepted:
        tab.accept_writeback(decision.items)
    return Report(value=analysis.result.value)
finally:
    tab.save()
```

`FitSummary` 是作者宣告的 `BaseModel`，欄位對應 adapter 的 owner summary。
框架用 `TypeAdapter` 驗證 summary，不把 GUI 內部 analysis dataclass 填回來。
欄位缺失或型別不合會失敗，不補值。沒有 scalar 的 figure-only 分析可用空的具名 view。
`report_number` 只把不可估計或非有限的 fit scalar 轉成 null，不改擬合或 writeback 政策。

`tab.save()` 同步保存 raw data 與全部已完成階段的 artifacts。
Analyze 不會自動保存。各 recipe 在已有 Run 結果後進入 `try/finally`，收尾只 save 一次。
缺參數或 Run 啟動失敗不進入這個 finalizer，避免無來源的 Save 蓋掉原錯誤。
每個輸出的成功與失敗由 `runs` 中的 Save 收據回報，完成不代表所有格式都保存成功。

## 問答與取消

`ask_writeback()` 捕捉當次候選，再等待回答。回答本身不寫入校準。
只有 `accept_writeback(decision.items)` 寫入被選擇且未變更的候選。
略過時不寫入。原始目的地來自 Run 的 capture，不改用目前 aliases。

Run scope cancel 只停止正在取得資料的 Run，回傳 partial，recipe 繼續後續分析。
Script scope cancel 使沒有結果的 Analyze 或 Ask raise `RecipeCancelled`。
這是普通例外，不是 `asyncio.CancelledError`。
Analyze、ask_writeback、session.ask 都回非 Optional 結果，不用 None 表示取消。
作者不 catch、不寫取消分支，已有 Run 的 `finally` 仍保存可用結果。
Runtime 回覆 `status="cancelled"`、`report=null` 與各次 `runs`。
其他錯誤仍保留失敗診斷，不重跑 acquisition。

## 操作文件與驗收

用 `recipe_guide(adapter="twotone/t1")` 依 adapter 名查 operator guide。
Recipe 可以串多個 adapter，宣告不另設單一 adapter mapping。
七個 mux 的輸入、原始 tone 位置與報告見 [mux guide](mux-guide.md)。
Framework 的執行、問答與取消契約見
[recipe 與 execution](../../lib/zcu_tools/mcp/measure/README.md#recipe-與-execution)。
新增 adapter 與流程的入口見 [實驗教學](../tutorial/03-experiment.md)
與 [recipe 教學](../tutorial/04-recipe.md)。

Lab 定義以直接 schema、guide、adapter forms-open 與離線 smoke 審閱。
Framework pytest 使用合成 recipe，不 import `zcu_lab`，也不固定這裡的實驗預設值。
