# MCP recipes

**讀者：** lab-author、agent。查找 recipe 宣告、操作順序與作者契約。

**Last updated:** 2026-11-01，作者 await 與測試邊界改連 owner

Recipe 是操作 GUI tab 的 async 函式。Core 擁有 cfg、量測與分析計算，Adapter 接上表單、RunRecord 與分析來源。
Recipe 選來源、設參數，再決定 Run、Analyze 與 writeback 問答的順序。

## 宣告與註冊

[__init__.py](__init__.py) 的 `RECIPES` 明列宣告，不掃描目錄。
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
Recipe 的 half／integer 輸入與報告是原生偏壓擬合點，不是 component 欄位。
Component v3 使用 `{offset, period}` 與 `flux_dev`。量測 writeback 分開提 signed period 與預設勾選的 offset，note 列其他儀器的 Run snapshot 值與扣除貢獻。
缺席儀器不貢獻，period 保留。獨立 app Export params 只寫 energies 與選定 period，一律保留 component offset。

## 作者可呼叫的操作

`session.open_tab(adapter, target={"qubit": inputs.target}, reuse=inputs.reuse_tab_id)`
開啟或 Reset tab。Range find 的 mapping key 是 `resonator`，mux 是 `mux`。
Target mapping 的省略／KEEP／null 項目沿用全域 alias，只對這次 capture 生效。
Reuse 是 idle matching tab ID，省略／KEEP／null 開新 tab。不改全域 aliases 或 GUI focus。
`tab.read_parameter(path, missing_ok=False)` 讀 tab 固定捕捉的參數。
`session.device_unit(name)` 讀裝置快取的 flux 單位，不碰硬體或刷新快取。
`tab.read(path)` 回 `FormField`，使用 dotted string，例如 `"tones.2.drive"`。
Module 與掃描欄位位於頂層，例如 `"readout"`、`"length"`，不加 modules／sweep 包裝。
T2Ramsey／T2Echo 使用 `"detune_ratio"`；CPMG 的共用時間窗使用
`"min_length_us"`、`"max_length_us"`、`"length_points"`，不加 extras 前綴。
這些是 adapter 的 Form input 路徑，adapter.run 再轉成保存於 RunRecord 的實驗 cfg。

寫入使用非空 dotted string，例如 `"tones.2.drive.gain"`，索引指完整清單的原始位置。
`tab.set(path, value)` 原樣替換欄位，不選來源、不自動 convert。
`tab.select(path, option)` 使用 owner 的精確 option label 或 captured module key。
單 kind 的 inline label 是 `"Inline"`，多 kind 是 `"Inline: <kind>"`。
`tab.command(path, "to_inline")` 複製固定來源，保留原 module／waveform family。已 inline 時成功且不改草稿。
三者同步完成。Set／select 收到 KEEP 不發 mutation，None 原樣交 owner。
Nullable selector 用 None 關閉，非 nullable 由 owner 拒絕，不用字串 `"off"`。

明填 scalar 前直接 command to_inline，再 set。已 inline 時 command 不改值。
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
Analysis updates 直接傳 dict，例如 `updates={"dual_exp": inputs.dual_exp, "skip": inputs.skip}`。
`tab.analyze` 送 owner 前略過 KEEP 項，明填 null 原樣交分析 owner。
全 KEEP 與不帶 updates 相同，保留 owner 的分析參數。

Lab async 函式的 await 與序列執行限制遵循 [ADR-0075](../../docs/adr/0075-agent-interface-layers.md#async-協作與控制命令)，檢查命令見 [await guardrail](../../tools/README.md#lab-script-await-guardrail)。
通常的順序如下：

```python
await tab.run()
try:
    analysis = await tab.analyze("primary", result=FitSummary)
    if analysis.result is None:
        # Descendant analysis cancellation, not a fitting exception.
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

Analyze 回傳就發布結果，拋錯就把原錯誤交給作者，不依例外型別或原因分類。
Fitting 的 `AnalysisFailed` 也會向上傳播；作者如需處理，從
`zcu_tools.analysis.failure` import，並自行決定是否捕捉，未捕捉就使 recipe failed。
Record 與 postprocessing 的錯誤同樣向上傳播，不以空報告掩蓋。
逐 tone 的失敗隔離與 summary 內容由分析 owner 決定，recipe 不替 owner 分類。

範例的 `AnalysisResult.result=None` 分支只處理子層分析取消且沒有結果的情況，
此時 status="failed"，reason 說明取消原因；它不是 fitting 例外的交付方式。
Report 的 nullable value、writeback_applied 與 nullable analysis_failure 由作者宣告，
此範例用 analysis_failure 保存取消原因，成功時為 null。
在解參照 result 或開始下一個分析前先檢查 result。
取消而無結果時不詢問或接受 writeback，也不跑相依 Post；先前成功的估計仍可由作者保留。

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
`tab.analyze` 與 `tab.ask_writeback` 都回非 Optional 物件，不用 None 表示根取消。
子層分析取消且無結果時，AnalysisResult.result 為 null；fitting 例外與根取消都不走這個值路徑。
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
| `onetone_range_find` | `onetone/range_find` | 必填絕對 MHz `freq:{start,stop}` 與 `n`。可選 `resonators` 提供 picker 標籤與初始頻率升序配對，不進 acquisition cfg。省略保留分析名單，null 原樣交 owner，空清單做無名探索。單一 Interactive 入口使用靜態初始猜測，預設不平滑、prominence 倍率 5、FWHM 上限 25 MHz，以 depth / FWHM 排序。`min_separation` 為有限正數 MHz，初值 1 MHz，與名單經同一 Analyze updates 傳入。Recipe 等待 agent 經 `tab_interact` 調整或確認並送 done，才詢問 writeback，不自動送 done。回覆最終 name、freq、depth、fwhm、score 與 identity／writeback 診斷。手動非候選點的 metrics 為 null。 |
| `stirap` | `twotone/stirap` | 必填 `scan_mode`：delay、amplitude 或 detuning。第一軸 `start/stop/n` 分別是 Stokes lead（us）、共同 gain scale（無單位）或 Pump detuning（MHz）；detuning 另用 `stokes_start_mhz/stokes_stop_mhz/stokes_n` 掃 Stokes 軸。操作者提供兩個 carrier，分析只用 Run 保存的 centroid3 readout。完整 Run、Analyze、Save，不提出 writeback。 |
| `raman_gf` | `twotone/raman_gf` | `carrier_mhz` 由操作者提供，`waveform_asset` 指定已建立的 arbitrary envelope。`scan_mode` 沿用 frequency、area、frequency_area；detuning 使用 two-photon MHz，正值使單 carrier 增加一半 detuning。分析只用 Run 保存的 centroid3 readout，完整 Run、Analyze、Save，不提出 writeback。 |
| `twotone_flux_dep` | `twotone/flux_dep` | 必填 drive 的 `center_mhz`、`span_mhz`、`freq_points` 及 native flux device、unit、端點、點數。互動 picker 完成後詢問寫回 qubit flux 校準。 |

有 writeback proposal 的流程先 Run、Analyze，再依各 recipe 的條件詢問寫回，只接受回答選定的候選，最後 Save。
STIRAP 與 Raman 沒有 writeback proposal，不詢問寫回。兩者回報 saved confusion matrix 的 condition number 與 minimum singular value；analysis summary 只保留 training confusion matrix 與這兩個診斷，逐點 g/e/f raw/corrected fractions 只畫在圖中。分析失敗仍 Save g/e/f counts。三態 reference 與 STIRAP 各 mode 不另加工具。
四個 `mux_{onetone,twotone}_flux_{global,local}` recipe 共用 flat cfg 路徑，
逐 participant 用原始 tone index 開啟 native picker，再分別詢問選定寫回。
`df/n`、`sweeps/n` 與 `flux/flux_n`、`flux_sweeps/flux_n` 的形狀及預選線範例見
[mux flux recipe guide](mux-guide.md#mux-flux-maps)。
`writeback_applied` 只在寫入收據確認成功項目後為 true。略過不寫入。
Framework 的執行、問答與取消契約見
[recipe 與 execution](../../lib/zcu_tools/mcp/measure/README.md#recipe-與-execution)。
新增 adapter 與流程的入口見 [實驗教學](../tutorial/03-experiment.md)
與 [recipe 教學](../tutorial/04-recipe.md)。

Lab 定義的測試邊界與驗收方式遵循 [ADR-0072](../../docs/adr/0072-module-boundaries.md#分層)。
