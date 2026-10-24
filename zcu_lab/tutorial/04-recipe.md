# 為實驗加入 MCP recipe

**讀者：** lab-author、agent。為已註冊的實驗新增 async recipe，串接 Run、保存與 preview。

**Last updated:** 2026-10-24

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
範例不補校準，check_ready 先回報缺值與路徑，阻止 Run。
執行前先在 entry／library 備好來源，並核對量測範圍。

```python
from pydantic import Field

from zcu_tools.mcp.measure.recipe_api import (
    RecipeInputs, RecipeReport, RecipeSession, recipe,
)


class PowerMapInputs(RecipeInputs):
    """Use the captured Form with optional target, reuse and ordered overrides."""


class PowerMapReport(RecipeReport):
    """Report acquisition progress; Save receipts are returned separately in runs."""

    map_acquired: bool = Field(
        description="Whether acquisition reached preview; not a Save success flag."
    )


@recipe("power_map", "Map qubit drive gain and frequency, then save and preview.")
async def power_map(
    session: RecipeSession, inputs: PowerMapInputs
) -> PowerMapReport:
    """Acquire the tab's frequency/gain map without fitting or writeback.

    Target selects this Reset's qubit alias; overrides edit the captured Form.
    Form defaults supply the scan and pulses. check_ready reports missing cfg
    values before Run. A child Run stop returns partial data, then finally saves
    it on this worker. Root cancellation drains
    the Run and raises before this Save finalizer; the partial stays in the tab.
    """
    # Capture the requested target without changing the global aliases.
    tab = session.open_tab(
        "twotone/power_map",
        target={"qubit": inputs.target},
        reuse=inputs.reuse_tab_id,
    )
    for edit in inputs.overrides:
        tab.set(edit.path, edit.value)
    tab.check_ready()

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
`@recipe` 從第二個參數的 inputs model 推導 MCP JSON schema，並驗證呼叫輸入。
Inputs 不渲染 GUI Form，不另寫 input_schema。
Recipe 可以開多個 adapter，因此宣告不設 adapter_name。
Report 是作者選擇的具名 model，不需要 SummaryParameter、SummaryEstimate 或空的 summary 宣告。
框架另外回 `RecipeOutcome(status, report, runs)`，runs 記每次 Run 與 Save 收據。
`map_acquired=True` 只表示程式走到該處，不表示所有格式都保存成功。

`target` 只覆寫這個 tab 在 Reset 時捕捉的 qubit alias，不改 GUI 全域 aliases。
其他角色沿用該 entry 的 alias 關係。省略／KEEP／null target 都保留全域 alias。
`reuse_tab_id` 指定 idle matching tab，省略／KEEP／null 則開新 tab。

加入可省略的 Form 欄位時，從 recipe_api import `KEEP, Keep`，宣告 `T | None | Keep = KEEP`。
省略保留 Form 值，明填 null 原樣交 owner。KEEP 不是 JSON 值，schema 不列 Keep 或 KEEP default。
主體先明寫選 module 與逐欄 set，再依序 set overrides，最後 check_ready。
例如 `tab.set("freq.n", inputs.points)`，寫入與 FieldEdit 用 dotted string。
Overrides 只含 path／value，省略是空 list，不接受 null，不去重或過濾 null。
`tab.read("qub_pulse")` 用 dotted string，回 FormField。
明填 scalar 前直接 command to_inline，再 set。已 inline 時 command 不改值。
單 kind module 與巢狀 waveform 同規則，保留原 family，不強選 const。
全部 scalar 省略時不切來源，也不因來源無效而 fallback。
Center／span 必須成對提供或省略，不從 endpoints 或校準補齊，null 不參與算術。
具名參數讀取用 `tab.read_parameter`，裝置快取單位用 `session.device_unit`。
完整逐欄範例見 [Recipe 撰寫](../v2/measure-authoring.md#recipe-撰寫)。
Author API 與來源語意見 [recipe 與 execution](../../lib/zcu_tools/mcp/measure/README.md#recipe-與-execution)。

## 3. 有分析時明確串接問答與保存

FIT adapter 的 Run 後可使用以下流程。Analysis updates 只濾 KEEP，明填 null 原樣交分析 owner。
`FitSummary` 直接 import 分析 owner 的具名 summary，不另宣告鏡像。`Report` 由作者宣告。
框架以 TypeAdapter 驗證 owner summary，不補欄位。
此例的 Report 包含 nullable value、writeback_applied 與 nullable analysis_failure。

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

預期擬合失敗回 `AnalysisResult(result=None, status="failed", reason=...)`。
每次 Analyze 後先檢查 result，再解參照估計或開始下一個分析。
失敗時回報 analysis_failure，不詢問或接受 writeback，也不跑相依的 Post。
Post 失敗時保留已成功的 Primary 估計。程式錯誤仍 raise，不轉成分析失敗值。

Analyze 不自動保存。`tab.save()` 同步保存 raw 與全部已完成階段的 artifacts，沒有 Save handle 要 await。
如果要先保住 raw，可以在 Run 後再額外 Save 一次。
不要在 Run 取得來源前進入 Save finalizer，避免保存失敗蓋掉原來的 Run 錯誤。
回答本身不寫回，只有明確 accept_writeback 才寫已捕捉的項目。
它回傳收據，`receipt.raise_for_error()` 在失敗時保留 owner 診斷並終止作者流程。
不要以 decision.accepted 宣稱已寫入。Owner 一次核對兩個 stages 的選定 IDs，unknown／duplicate 在第一筆寫入前拒絕。
問答期間 draft 編輯或替換不觸發 stale 拒絕，寫入採 owner 當下 draft 的值與目的地，不還原舊 proposal。

Run scope cancel 使 Run 回 partial，之後步驟仍繼續。
取消 recipe 根則讓下一個 call／ask await raise 普通 `ScriptCancelled`。
進行中的 Run 先合作 drain，保留 run_id、partial 與完成事實，再 raise。
Analyze／ask_writeback／session.ask 回非 Optional 物件，不用 None 表示根取消。
AnalysisResult.result 為 null 是分析失敗值，與根取消的例外不同。
作者不 catch 根取消，已進入 try 的 finally 仍在同一 worker 同步保存已有結果。
範例的 `await tab.run()` 在 try 外，根取消在此 raise 不會進入 Save finalizer。
Drained partial 留在 tab，交由 agent 決定是否另行保存。
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
