# 為實驗加入 MCP recipe

**讀者：** lab-author、agent。為已註冊的實驗新增 async recipe，串接 Run、保存與 preview。

**Last updated:** 2026-10-22

Recipe 把 GUI tab 操作串成一個 MCP 流程，量測仍由 adapter 執行。
這篇接續 [實驗與表單](03-experiment.md)，為 `twotone/power_map` 加入 `power_map` recipe。
先確認 adapter 已列在 GUI New Tab，且 `guide.md` 能讀取。
Recipe 不負責註冊 adapter，也不從 recipe 名稱推測 adapter。

## 1. 選擇與實驗能力相符的範本

有 fit 與 writeback 的實驗可以從 [t1.py](../recipes/t1.py) 開始。
本篇的 power map 沒有 analysis，改參考
[onetone_spectrum_over_power.py](../recipes/onetone_spectrum_over_power.py) 的 Run、Save、preview 流程。
不要把 T1 的 Analyze／writeback 複製到 `AnalysisMode.NONE` 的 adapter。
GUI cfg、Form 初值與量測序列仍由原實驗擁有，不在 recipe 再建一份。

## 2. 宣告 inputs 與 async 函式

建立 `zcu_lab/recipes/power_map.py`，放入以下範例。
它使用新 tab 的 Form 初值，取得資料後 preview，最後同步保存 raw 與已完成的 artifacts。
範例不補校準，缺少必要輸入時 GUI 會拒絕 Run。
執行前先在 entry／library 備好來源，並核對量測範圍。

```python
from pydantic import BaseModel, Field

from zcu_tools.gui.form import FormModel
from zcu_tools.mcp.measure.recipe_api import RecipeSession, recipe


class PowerMapInputs(FormModel):
    """Choose the qubit for this tab's fixed Reset capture."""

    target: str | None = Field(
        None,
        min_length=1,
        description="Qubit alias for this Reset; null keeps the GUI aliases.",
    )


class PowerMapReport(BaseModel):
    """Report acquisition progress; Save receipts are returned separately in runs."""

    map_acquired: bool = Field(
        description="Whether acquisition reached preview; not a Save success flag."
    )


@recipe("power_map", "Map qubit drive gain and frequency, then save and preview.")
async def power_map(
    session: RecipeSession, inputs: PowerMapInputs
) -> PowerMapReport:
    """Acquire the tab's frequency/gain map without fitting or writeback.

    Inputs select only this Reset's qubit alias. Form defaults supply the scan
    and pulses. Invalid calibration or cfg fails Run. Cooperative Run stop
    retains partial data; finally saves the available Run on this worker.
    """
    # Capture the requested aliases without changing the global table.
    if inputs.target is not None and not inputs.target.strip():
        raise ValueError("target must be a non-empty string or null")
    tab = session.open_tab(
        "twotone/power_map",
        aliases=None if inputs.target is None else {"qubit": inputs.target},
    )

    # Only enter the Save finalizer once a Run source exists.
    await tab.run()
    try:
        tab.preview("run")
        return PowerMapReport(map_acquired=True)
    finally:
        tab.save()


DEFINITION = power_map
```

`RecipeSession` 由 runtime 傳入，不是 MCP input。
`@recipe` 從第二個參數的 `FormModel` 推導 MCP JSON schema，並驗證呼叫輸入。
Inputs 不渲染 GUI Form，不另寫 input_schema。
Recipe 可以開多個 adapter，因此宣告不設 adapter_name。
Report 是作者選擇的具名 model，不需要 SummaryParameter、SummaryEstimate 或空的 summary 宣告。
框架另外回 `RecipeOutcome(status, report, runs)`，runs 記每次 Run 與 Save 收據。
`map_acquired=True` 只表示程式走到該處，不表示所有格式都保存成功。

`target` 只覆寫這個 tab 在 Reset 時捕捉的 qubit alias，不改 GUI 全域 aliases。
其他角色沿用該 entry 的 alias 關係。
Author API 與來源語意見 [recipe 與 execution](../../lib/zcu_tools/mcp/measure/README.md#recipe-與-execution)。

## 3. 有分析時明確串接問答與保存

FIT adapter 的 Run 後可使用以下流程。
`FitSummary` 與 `Report` 由作者按 owner summary 宣告，框架以 TypeAdapter 驗證，不補欄位。

```python
await tab.run()
try:
    analysis = await tab.analyze("primary", result=FitSummary)
    decision = await tab.ask_writeback()
    if decision.accepted:
        receipt = tab.accept_writeback(decision.items)
        receipt.raise_for_error()
    return Report(value=analysis.result.value)
finally:
    tab.save()
```

Analyze 不自動保存。`tab.save()` 同步保存 raw 與全部已完成階段的 artifacts，沒有 Save handle 要 await。
如果要先保住 raw，可以在 Run 後再額外 Save 一次。
不要在 Run 取得來源前進入 Save finalizer，避免保存失敗蓋掉原來的 Run 錯誤。
回答本身不寫回，只有明確 accept_writeback 才寫已捕捉的項目。
它回傳收據，`receipt.raise_for_error()` 在失敗時保留 owner 診斷並終止作者流程。
不要以 decision.accepted 宣稱已寫入。Owner 先核對兩個 stages 的捕捉 revision，任何 draft 改動或替換都在第一筆寫入前拒絕。

Run scope cancel 使 Run 回 partial，之後步驟仍繼續。
Script scope cancel 讓無結果的 Analyze／ask_writeback／session.ask raise 普通 `RecipeCancelled`。
三個 await 都回非 Optional 結果，不用 None 表示取消，也不加 require-value helper。
作者不 catch、不寫取消分支，finally 仍在同一 worker 保存已有結果。
Runtime 回 cancelled、report=null 與 runs，這個普通例外不記成 error。
作者 task 不會收到 `asyncio.CancelledError`，真正的 owner 或 cleanup 例外仍保留診斷。

所有 lab async def 的 await 都遵守同一條規則，只允許 env／tab／session 的呼叫，
或靜態解析到 zcu_lab 內定義的 helper，定義可用 def 或 async def，helper 可以再呼叫 helper。
在 async def 內，明列 asyncio.*、create_task、gather、wait_for、TaskGroup 的直接呼叫即使不 await 也拒絕。
其他 async with／async for 不另禁。
Quality gate 全掃 async def，不分 decorator、entry、helper 或可達性。
AST 檢查不追蹤 assignment aliases、動態分派或包裝，不是 type checker。
不要用這些繞過手法新增並行工作，runtime 擁有 operation 與 worker loop。

## 4. 把宣告加入明確清單

在 `zcu_lab/recipes/__init__.py` 加入 import：

```python
from .power_map import DEFINITION as POWER_MAP
```

再在 `RECIPES` tuple 加入 `POWER_MAP,`。
只新增檔案不會出現 MCP tool，import 也不自動註冊或開始量測。
[mcp_servers/measure.py](../../mcp_servers/measure.py) 把清單注入 session 與 tool table。
重啟 MCP server 重新載入清單。
GUI 的 Reload experiments 只更新 adapter catalog，不更新另一個 process 的 recipe registry。
若也新增 adapter，先在 GUI 重載 catalog 或重啟 GUI，再連線。

## 5. 用 adapter 查 operator guide

連到已有新 adapter 的 GUI 後，呼叫：

```text
recipe_guide(adapter="twotone/power_map")
```

工具回 adapter 與 guide，不推測 recipe 到 adapter 的 mapping。
`guide.markdown` 是 adapter 的 guide.md 全文，resource_dir 是 GUI host 上該目錄的絕對路徑。
未宣告 guide 時 markdown 回預設文字，resource_dir 為 null。
跨機 client 不一定能存取該目錄。
工具不讀圖、不執行量測，也不把 Python docstring 當作 guide。

用 MCP tools/list 確認 power_map 工具已註冊，再查 adapter guide 核對操作步驟。
Unknown tool 時檢查 RECIPES 並重啟 MCP server。
Adapter 查詢失敗時檢查 GUI catalog 與重載狀態。
看到 guide 只證明文件能讀取，不代表校準齊全或已允許操作硬體。

Recipe 定義以直接 schema 審閱、guide 與離線 smoke 驗收，不為這份 lab source 新增 pytest。
Framework tests 用合成 recipe 驗證執行契約，不 import zcu_lab。
