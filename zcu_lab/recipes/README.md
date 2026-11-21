# MCP recipes

**讀者：** lab-author、agent。查找 recipe 宣告、操作順序與作者契約。

**Last updated:** 2026-11-20，two-qubit CZ amplitude、local Z、constructive phase、conditional phase 與 paired RB

Recipe 是操作 GUI tab 的 async 函式。Core 擁有 cfg、量測與分析計算，Adapter 接上表單、RunRecord 與分析來源。
Recipe 選來源、設參數，再決定 Run、Analyze 與 writeback 問答的順序。

## 宣告與註冊

[__init__.py](__init__.py) 的 `RECIPES` 明列 50 個 recipe，不掃描目錄。
每個模組用 `@recipe(name, description)` 裝飾
`async def (session: RecipeSession, inputs: Inputs) -> Report`。
每個 recipe 的 `Inputs` 繼承 `RecipeInputs`。實驗欄位逐欄寫單位、預設與 null 政策。
`RecipeInputs` 不宣告欄位。接受 Open／Reset 控制的 recipe 自己宣告 `target` 與 `reuse_tab_id`。
單目標用 `target: SingleTarget`，多角色可宣告自己的 target model。重用既有 tab 用 `reuse_tab_id: ReuseTabId`。
接受最終欄位操作的 recipe 自己宣告 `overrides: Overrides`。這三個共用型別從 `recipe_api` 匯入。
可省略的實驗欄位宣告為 `T | None | Keep = KEEP`。省略保留 Form 值，明填 null 原樣交 owner。
KEEP 是 Python 省略標記，不是 JSON 值。Schema 隱藏 Keep 分支與 KEEP default，欄位不列 required。
Mux 的 participants 使用完整 tones 清單的原始 index，逐 tone 範圍清單保留原始順序，見下方 mux guide。
框架從這個 model 產生 MCP JSON schema，不另維護手寫 schema。
Recipe inputs 只作 JSON schema／pydantic 驗證，不渲染 GUI Form。
每個 recipe 的 `Report` 繼承 `RecipeReport`，只加實驗的回報量。
基底提供 `writeback_applied`、`analysis_failure`、`fit_quality` 與 `diagnostics`。
單目標的 `fit_quality` 轉出 owner 已計算的主擬合診斷。Mux 把這個欄位放在各 tone，頂層不做合併。
沒有可轉出的診斷就回 null，不新增品質門檻或改變 writeback。Mux Ramsey 也轉出各 tone 的 `sampling_quality`。
GE 的 `fit_quality` 是 joint 擬合診斷，另用 `ground_fit_quality`、`excited_fit_quality` 回報各 histogram 的擬合。Mux GE 在 Primary 的各 tone 回報這三份診斷。
各模組的 `DEFINITION` 指向裝飾後的宣告。組合根把 `RECIPES` 注入框架。
Import 不開 GUI、不註冊 adapter，也不執行量測。新增宣告後須重啟 MCP server。

掃描端點、flux 範圍與裝置單位檢查直接寫在 recipe 主體，不另設選源 helper。
數字回報使用 lib 的 `report_number`，不改單位或精度。
`mux_helpers.py` 只共用 tone 範圍輸入與報告欄位，不組裝 Form edits。
[`two_qubit_helpers.py`](two_qubit_helpers.py) 共用四角色 target、選定候選的寫回與收據處理，並說明如何轉交 owner 報告。Recipe 主體決定何時詢問寫回。
[`two_qubit_rb_helpers.py`](two_qubit_rb_helpers.py) 共用兩個 paired RB recipe 的輸入與 CZ 誤差估計欄位。
`ro_optimize_chain_helpers.py` 共用三個 readout-chain ro_optimize recipe 的報告欄位，並複製最佳 setting 的 score、QND、M 與模型近似誤差尺度。
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
`tab.command(path, name)` 執行該欄位宣告的 command。`to_inline` 複製固定來源，保留原 module／waveform family。已 inline 時成功且不改草稿。
三者同步完成。Set／select 收到 KEEP 不發 mutation，None 原樣交 owner。
Nullable selector 用 None 關閉，非 nullable 由 owner 拒絕，不用字串 `"off"`。

明填 scalar 前直接 command to_inline，再 set。已 inline 時 command 不改值。
巢狀 waveform 也按同一規則，不以固定 const 或 select Inline 作可寫前置。
全部 scalar 省略時不切來源。來源壞掉不 fallback，讓 owner 回報路徑與原因。
Center／span 必須成對提供或成對省略，缺一個就回 needs_parameters 指名另一欄。
兩者省略不 set，不從現有 endpoints 或校準推算。Null 不參與算術，原樣交 owner。
完整 tone 範圍清單必須與完整 tones 等長，缺少或多出的原始位置都要回報。

從 `recipe_api` 匯入 `Overrides` 並在 Inputs 宣告 `overrides: Overrides`。省略時是空列表，不接受 null。
每筆填 dotted logical `path`，另填恰好一個操作：

- `value` 呼叫 `tab.set`。巢狀物件、清單、Expr 與具名 native model 原樣交 owner 驗證。
- `select` 呼叫 `tab.select`。可填精確選項標籤或 captured library key，null 清空可為空的 selector。
- `command` 呼叫 `tab.command`。可填欄位宣告的指令，例如 `to_inline`。

主體先設實驗值，再套 overrides，最後驗整份 Form／cfg：

```python
tab.apply_overrides(inputs.overrides)
tab.check_ready()
```

`apply_overrides` 按列表順序逐筆執行，重複路徑不去重。失敗時錯誤含序號及路徑，例如 `overrides[1] (drive.gain)`。
後面的操作不執行，先前成功的操作不回滾。
`check_ready` 唯讀，不 Run 或刷新 capture。缺參時拋 `RecipeNeedsParameters`，列出全部路徑與原因。
Run admission 仍會再次驗證。
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
Two-qubit 的角色、herald reset、寫回與 RB 波形覆寫見 [two-qubit guide](two-qubit-guide.md)。

三個 thermal-chain ro_optimize recipe 以 K 等待熱平衡，再無 drive 連讀 16 次。
`relax_delay` 設 K，單位 us。K 後熱態不同於 readout-driven steady state。
Score 是受限二態模型的資訊 proxy，乘上 M 相關增量懲罰。M 不等於可辨識的 leakage，純二態再布居也會受罰。
只取 score 最大的 setting，同分保留 acquisition order，誤差不參與排名。
最佳 setting 在邊界時，擴大該側範圍後重測。
Live 與 analysis 都顯示 score、QND、M 與 |ρ| 三個 panel，紅線標最佳點。
QND 軸固定 0..1，但回報不裁切。Score 軸只按 score 值決定，不被大誤差棒撐大。
誤差是模型條件式近似尺度，不是一般 one-sigma；score 傳播忽略 M、主軸及共享 covariance 的不確定度。
稀少遠端第三態、相關雜訊及 IQ 重疊可能誤導 score 或隱藏 leakage。實機 K R^16 尚未驗證此效用。
各實驗 guide 說明 clipping 斜率、熱態對比與偵測限制。Saved record 保留 count 加完整 107 sums，不只存 M aggregate。

新增流程：

| MCP tool | Adapter | 掃描與回報 |
| --- | --- | --- |
| `mux_onetone` | `mux/onetone` | `participants` 收完整 readout 的原始整數 index，依清單順序執行，不以名稱查位置。`df:{start,stop}` 是共同 MHz offset，`n` 是共用點數。回覆每 tone 的 index、name、freq、kappa、fit_quality 與 identity 診斷。 |
| `onetone_range_find` | `onetone/range_find` | 必填絕對 MHz `freq:{start,stop}`、`n` 與非空 `resonators` 名單。名稱數是預期峰數，初始依頻率升序配對，不進 acquisition cfg。傳入實際元件名稱，例如 `["R1"]`；未辨識的峰可用暫用名稱。GUI Analysis 預設為 `["R1"]`，recipe 仍須明填名單，不沿用此預設。單一 Interactive 入口預設不平滑、自動依名稱數求 prominence factor、FWHM 上限 25 MHz，以 depth / FWHM 排序。`min_separation` 為有限正數 MHz，初值 1 MHz，與名單經同一 Analyze updates 傳入。Recipe 等待 agent 經 `tab_interact` 調整或確認並送 done，才詢問 writeback，不自動送 done。每個具名 dip 都提議 `<name>.freq`，不依 capture 篩選，Unassigned 不提案。接受後在作用中文件覆寫既有 freq，或建立只指定 freq 的 resonator，其餘使用 model defaults。名稱打錯會建立多餘元件，可在接受前取消該候選。回覆最終 name、freq、depth、fwhm、score、實際 `prominence_factor` 與 `identity_diagnostic`。手動非候選點的 metrics 為 null。 |
| `stirap_amplitude` | `twotone/stirap/amplitude` | `pump_start/pump_stop/pump_n` 與 `stokes_start/stokes_stop/stokes_n` 指定兩條獨立 gain 軸，兩組 `n → expts`。支援 signed gain 與零值；任一有效端點超出 `[-1, 1]` 時停止，不送出 Run。同一 program 用兩個硬體 sweep 取得二維 g/e/f counts。Pump/Stokes 各選 gauss、cosine、drag 或 arb，`pump_waveform_ref/stokes_waveform_ref` 選 library key 或 inline Form 選項，overrides 設波形參數。載波用 MHz，固定 `stokes_lead_us` 正值表示 Stokes 先。Run、Analyze、Save，不提出 writeback。 |
| `stirap_delay` | `twotone/stirap/delay` | `start_us/stop_us/n` 掃描 Stokes lead，正值 Stokes 先、負值 Pump 先。兩條各自選 gauss、cosine、drag 或 arb 與各自時長。必填量得的 `pump_freq_mhz/stokes_freq_mhz` 與 signed gains。完整 Run、Analyze、Save，分析只用保存的 centroid3，不提出 writeback。 |
| `stirap_detuning` | `twotone/stirap/detuning` | Pump 與 Stokes 各自選 gauss、cosine、drag 或 arb。`pump_start_mhz/pump_stop_mhz/pump_n` 掃內層 Pump，`stokes_start_mhz/stokes_stop_mhz/stokes_n` 掃外層 Stokes。兩軸都是加到量得載波的 MHz offset。`pump_waveform_ref/stokes_waveform_ref` 選 library key 或 `Inline: cosine` 等 Form 選項，overrides 設所選波形參數。完整 Run、Analyze、Save，不提出 writeback。 |
| `raman_gf` | `twotone/raman_gf` | `carrier_mhz` 由操作者提供；drive 接受任何受支援波形，`waveform_asset` 可指定已建立的 arbitrary envelope（sqrt(sin) 形狀需 arb）。`scan_mode` 使用 frequency、gain、frequency_gain；detuning 使用 two-photon MHz，正值使單 carrier 增加一半 detuning，gain 直接使用 DAC full-scale fraction（a.u.）。分析只用 Run 保存的 centroid3 readout，完整 Run、Analyze、Save，不提出 writeback。 |
| `ro_optimize_chain_power` | `twotone/ro_optimize/chain_power` | 可省略 `gain_start`／`gain_stop`（a.u.）、`points`、`relax_delay`（us）、`reps`、`rounds`。每 shot 等待 K 後連讀 16 次，無 drive。回報最佳 `gain`、`score`、`score_error`、`qnd`、`qnd_error`、`correlation`、`amplification`、`mist`（M 診斷）。完整 Run、Analyze、Save，不提出 writeback。 |
| `ro_optimize_chain_freq` | `twotone/ro_optimize/chain_freq` | 同上，掃描端點為 `freq_start_mhz`／`freq_stop_mhz`（MHz），回報 `freq_mhz`。 |
| `ro_optimize_chain_length` | [`twotone/ro_optimize/chain_length`](../v2/twotone/ro_optimize/chain_length/guide.md) | 同上，`length_start_us`／`length_stop_us` 設 ADC window 掃描端點（us）。Run 依 trigger offset、time of flight 與各 window 設 pulse 長度，回報 `length_us`。 |
| `twotone_chain` | `twotone/chain` | 必填 drive 的 `center_mhz`、`span_mhz`、`points`。每點 48 次無 reset 的（驅動、讀出），`reps` 是每輪快掃次數。分析以 `twotone/freq` 的峰擬合擬合離背景的 sigma 曲線，`expected_center_mhz` 選峰。回報中心、近似寬度與 5% 誤報門檻，完整 Run、Analyze、Save，不提出 writeback。 |
| `chain_amp_rabi` | `chain/amp_rabi` | 掃 drive gain，`gain_range`、`points`。每個 rep 是一個 cycle：K（`relax_delay`）後 (Rabi pulse, 讀, 讀) × 8，`reps` 是每輪每點 cycle 數。分析以 `twotone/rabi/amp_rabi` 擬合第一次讀出取起點，再聯合擬合 65 個數。回報 π gain、Rabi 頻率與 Wald 誤差、R、三項檢查，完整 Run、Analyze、Save，不提出 writeback。 |
| `chain_len_rabi` | `chain/len_rabi` | 掃 const／flat_top drive 的長度，`max_length_us`、`points`。Cycle 與 `reps` 同上。起點取自 `twotone/rabi/len_rabi` 的第一讀擬合。回報 π 長度、Rabi 頻率（MHz）與 Wald 誤差、R、三項檢查，不提出 writeback。 |
| `t1_cov` | `twotone/t1_cov` | 用單 tone pulse readout 讀取 π 與掃描等待前後的 IQ。`use_reset` 選讀取前的脈衝 reset，`points` 設 Expts。回報狀態改變機率擬合的 T1 與誤差，詢問 `$qubit.t1` 寫回並保存二階矩總和。 |
| `chain_t1` | `chain/t1` | 掃 π 後的等待，`max_delay_us`、`points`。每個 block 是 (π, 等待, 讀, 讀)。起點取自第一讀的單指數擬合，聯合擬合用 rate-equation 模型。回報 T1（振幅較大的 eigen time）、secondary time 與 Wald 誤差、三項檢查，不提出 writeback。 |
| `chain_t2ramsey` | `chain/t2ramsey` | 掃 Ramsey 等待，`max_delay_us`、`points`、`detune_mhz`（第二個 π/2 的相位斜率）。每個 block 是 (π/2, 等待, 反相 π/2, 讀, 讀)。起點取自 `twotone/t2ramsey` 的第一讀擬合。回報 T2、fringe 頻率、第三態衰減時間與 Wald 誤差、R、三項檢查，不提出 writeback。 |
| `chain_t2echo` | `chain/t2echo` | 掃總 echo 等待，`max_delay_us`（總長）、`points`。每個 block 是 (π/2, x/2, π, x/2, π/2, 讀, 讀)。起點取自 `twotone/t2echo` 的第一讀擬合。回報 T2 echo、第三態衰減時間與 Wald 誤差、三項檢查，不提出 writeback。 |
| `twotone_flux_dep` | `twotone/flux_dep` | 必填 drive 的 `center_mhz`、`span_mhz`、`freq_points` 及 native flux device、unit、端點、點數。互動 picker 完成後詢問寫回 qubit flux 校準。 |
| [`two_qubit_cz_amplitude`](two-qubit-guide.md#two_qubit_cz_amplitude) | `two_qubit/cz_amplitude` | `scale_range`、`scale_points` 掃兩路共同 gain 倍率。回報第一個 return scale 與擬合診斷。有候選才問 Primary 寫回；完成 Run 後保存。 |
| [`two_qubit_local_z`](two-qubit-guide.md#two_qubit_local_z) | `two_qubit/local_z` | 掃 `gate_count`，分別回報 F1、F2 的相位擬合。兩條擬合都給出 correction 才詢問寫回校正對。 |
| [`two_qubit_constructive_phase`](two-qubit-guide.md#two_qubit_constructive_phase) | `two_qubit/constructive_phase` | 掃 F2 相對相位和兩路共同脈衝長度。回報最大擬合速率的相位與干涉擬合診斷。有可辨識相位才詢問 F2 脈衝和 slot 寫回。 |
| [`two_qubit_conditional_phase`](two-qubit-guide.md#two_qubit_conditional_phase) | `two_qubit/conditional_phase` | 用 `ramsey_role` 選相位量測的 qubit，並設定 `gate_count`、`frequency_range` 與 `frequency_points`。回報唯一的 π 交點，或 null 與原因。有交點才問 Primary 頻率寫回；完成 Run 後保存。 |
| [`two_qubit_conditional_phase_heralded`](two-qubit-guide.md#two_qubit_conditional_phase_heralded) | `two_qubit/conditional_phase_heralded` | 掃描輸入同上。每發先做 mux herald，用 joint gg/ee 斜率找唯一的 π 交點；沒有交點不問寫回。完成 Run 後保存。 |
| [`two_qubit_rb`](two-qubit-guide.md#two_qubit_rb) | `two_qubit/rb` | `depths` 和 `seeds` 設定成對序列。回報 CZ 誤差、信賴區間和兩臂擬合診斷。保存，不詢問寫回。 |
| [`two_qubit_rb_heralded`](two-qubit-guide.md#two_qubit_rb_heralded) | `two_qubit/rb_heralded` | 每 shot 先做 herald readout。回報 joint gg/ee 估計和四組擬合診斷。保存，不詢問寫回。 |

有 writeback proposal 的流程先 Run、Analyze，再依各 recipe 的條件詢問寫回，只接受回答選定的候選，最後 Save。
CPMG report 回傳 `t2_1_us`、`t2_1_err_us`、`gamma`、`gamma_err` 與 `fit_rows`。可加權列少於三列時四個擬合值為 null，仍保存圖與原始資料；分析失敗時 `fit_rows=0` 並附原因。
STIRAP 與 Raman 沒有 writeback proposal，不詢問寫回。
它們回報 saved confusion matrix 的 condition number 與 minimum singular value。
STIRAP 另回 `f_population_estimate`，即本次網格內最大有限的 corrected f population。同值取最早取數的點。
它不裁切到 [0, 1]，不是 fidelity，也不代表全域最佳脈衝。逐點 g/e/f fractions 仍只畫在圖中。

| STIRAP recipe | 峰值座標 |
| --- | --- |
| `stirap_delay` | `stokes_lead_us` |
| `stirap_detuning` | `pump_detuning_mhz`、`stokes_detuning_mhz`，相對量得載波的 offset |
| `stirap_amplitude` | `pump_gain`、`stokes_gain`，a.u. |

沒有有限值或 Analyze 無結果時，回報的峰值與座標都是 null。
Analyze 拋錯仍保留原錯誤，且成功 Run 後仍 Save g/e/f counts。三態 reference 不另加工具。
`stirap_delay`、`stirap_amplitude` 與 `stirap_detuning` 使用各自獨立的 adapter 與表單。
舊合併實驗的 run 不做遷移，也不支援載入。

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
