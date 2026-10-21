# 為實驗加入 MCP recipe

Recipe 把 GUI 操作串成一個 MCP 流程，量測仍由 adapter 執行。
這篇接續 [實驗與表單](03-experiment.md)，為 `twotone/power_map` 加入 `power_map` recipe。
先確認 adapter 已列在 GUI New Tab，且 `guide.md` 能讀取。
Recipe 不負責註冊 adapter，也不從 recipe 名稱推測 adapter。

## 1. 選擇與實驗能力相符的範本

有 fit 與 writeback 的實驗可以從 [t1.py](../recipes/t1.py) 開始。
本篇的 power map 沒有 analysis，改參考
[onetone_spectrum_over_power.py](../recipes/onetone_spectrum_over_power.py) 的 Run、Save、preview 流程。
不要把 T1 的 Analyze／writeback 步驟複製到 `AnalysisMode.NONE` 的 adapter。

Recipe 本體應直接列出操作順序，見 [recipe 操作流程](../v2/experiment-authoring.md#2-在-recipe-函式本體寫出操作流程)。
GUI cfg、Form 初值與量測序列仍由原實驗擁有，不在 recipe 再建一份。

## 2. 寫 generator 與 definition

建立 `zcu_lab/recipes/power_map.py`，放入以下完整範例。
它使用新 tab 的 Form 初值，執行量測後保存 raw IQ 並擷取 Run preview。
範例不補校準；缺少必要輸入時，GUI 會拒絕 Run。
執行前先在 entry／library 備好來源，並核對量測範圍。

```python
from zcu_tools.mcp.measure.recipe import (
    RecipeDefinition,
    RecipeGenerator,
    RecipeSession,
)


def power_map(
    session: RecipeSession, *, target: str | None = None
) -> RecipeGenerator:
    """Map qubit IQ versus drive gain and frequency, then save raw IQ.

    target selects the qubit for this tab's fixed capture; None uses global
    aliases. Form defaults supply the scan and pulses. Missing calibration
    or invalid cfg stops Run. Cancelled acquisition skips save and preview.
    This recipe performs no fit or writeback.
    """
    # Prepare a tab with the requested fixed alias capture.
    if target is not None and not target.strip():
        raise ValueError("target must be a non-empty string or null")
    tab = session.open_tab(
        "twotone/power_map",
        aliases=None if target is None else {"qubit": target},
    )

    # Measure with the tab inputs; cancellation skips persistence.
    result = yield from tab.run_step()
    if result.status == "cancelled" or result.run is None:
        return

    # Save raw IQ and capture the Run preview without fitting.
    result.run.save_raw()
    result.run.preview()


DEFINITION = RecipeDefinition(
    name="power_map",
    description="Map qubit drive gain and frequency; save raw IQ and preview.",
    input_schema={
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "target": {"type": ["string", "null"], "minLength": 1},
        },
    },
    run=power_map,
    adapter_name="twotone/power_map",
    summary_parameters=(),
    summary_estimates=(),
)
```

`RecipeSession` 由 driver 傳入，不是 MCP input。
`input_schema` 只描述 callable 的 keyword inputs，本例只有選填 `target`。
`name` 是唯一的 MCP tool 名稱，`adapter_name` 是上一篇註冊的 GUI key，兩者可以不同。
`run` 指向 generator callable，不是執行後的 generator instance。
Registry 會檢查 schema 與 callback 是否一致。

`target` 只覆寫這個 tab 在 Reset 時捕捉的 `qubit` alias，不改 GUI 全域 aliases。
其他角色仍沿用該 entry 的 alias 關係。
取消時直接返回，不能把取消當作一次成功保存。
本例沒有 scalar fit，兩個 summary 宣告先留空。
需要在 MCP 摘要列出設定時，用 `SummaryParameter` 宣告 cfg path 與單位；
需要 fit estimates 時才加 `SummaryEstimate`。
Author API 與來源語意見 [recipe 與 execution](../../lib/zcu_tools/mcp/measure/README.md#recipe-與-execution)。

若實驗有 FIT，參照 T1 在保存 raw 後呼叫 `yield from run.analyze_step("primary")`。
需要 writeback 時，再用 `yield from ask(run.propose_writeback())` 等待回答，
只在 accepted 後呼叫 `tab.accept()`。回答本身不會寫回。
具體契約見 [分析、互動與寫回](../../lib/zcu_tools/mcp/measure/README.md#分析互動與寫回)。

## 3. 把 definition 加入明確清單

在 `zcu_lab/recipes/__init__.py` 加入 import：

```python
from .power_map import DEFINITION as POWER_MAP
```

再在 `RECIPES` tuple 加入 `POWER_MAP,`。
只新增檔案不會出現 MCP tool，import recipe 也不會自動註冊或開始量測。
目前的 MCP 組合根 [mcp_servers/measure.py](../../mcp_servers/measure.py)
把這份清單注入 session 與 tool table。

重啟 MCP server，讓它重新載入清單。
GUI 的 Reload experiments 只更新 adapter catalog，不會更新另一個 process 的 recipe registry。
若你也剛新增 adapter，先在 GUI 重載 catalog 或重啟 GUI，再連線。
工具與清單的對應見 [固定工具與 RPC](../../lib/zcu_tools/mcp/measure/README.md#固定工具與-rpc)。

## 4. 用 recipe_guide 核對 operator 文件

連到已有新 adapter 的 GUI 後，呼叫：

```text
recipe_guide(recipe="power_map")
```

工具回覆 `recipe="power_map"`、`adapter="twotone/power_map"` 與 `guide`。
`guide.markdown` 是 adapter 的 `guide.md` 全文，
`guide.resource_dir` 是 GUI host 上該檔案目錄的絕對路徑。
沒有宣告 guide 時，`guide.markdown` 仍回傳「尚未撰寫」預設文字，只有 `guide.resource_dir` 為 null。
跨機 client 不一定能存取 GUI host 上的資源目錄。
工具不執行量測、不讀圖片，也不把 Python recipe docstring 當成 guide。
Guide 查詢契約見 [setup 工具](../../lib/zcu_tools/mcp/measure/README.md#setup-工具)。

`recipe_guide` 不列出全部 recipes。用 MCP client 的 tools/list 確認有 `power_map`，
再用具名查詢核對它的 adapter mapping 與 operator 文件。
若回覆 unknown recipe，檢查 `RECIPES` 並重啟 MCP server。
若 adapter 查詢失敗，檢查 GUI 的 definitions 與重載狀態。
看到 guide 只證明註冊與文件能讀取，不代表校準齊全或已允許執行硬體量測。
