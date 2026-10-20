# Mux length Rabi

## GUI setup

Open `mux/len_rabi` from **New Tab** after binding `$mux` to a MuxGroup.
The capture must contain every qubit and its matching resonator in readout order.
Readout references `$mux.module.readout` when present, preserving the library's tones and scalar settings.
The stored module must use the mux family with the full group tone count.
Without a slot, Readout starts inline from the blank mux template with group calibration tones; complete its required fields before running.

The Tones picker chooses which element to edit, not which qubit runs.
Name is read-only. Ordinary edits cannot add, remove, rename or reorder tones.
Set Participants to a JSON name list, for example `["Q3", "Q1"]`.
That order becomes execution/data rows for original positions 2/0.
The picker can show Q2 while Q3/Q1 run.
Empty, duplicate and unknown participants remain editable invalid drafts and block Run.

Every tone needs a valid drive, including nonparticipants.
Errors in another picker element remain visible on the Tones row.
Select that element to fix them; hiding it or removing its name from Participants
does not bypass validation. Reps, rounds and recovery delay are common controls.
Recovery initially uses five times the largest captured T1, with the existing
missing-T1 default. A stored invalid T1 leaves recovery blank for correction.

Length Rabi accepts only const and flat_top drives.
For flat_top, total length must exceed both ramps.
Switch an arb source to a supported waveform explicitly; the Form does not
stretch, crop or replace an asset to make it valid.

## GUI results and saved sources

Run draws a combined `measurement` figure.
Each line keeps its own actual x coordinates, even when scan ranges differ.
The legend uses original index and source cfg name in participants order.
Per-tone native `measurement_i` figures remain available.
Analyze reads the selected Run/Load source, not the current Form.
Use the Analysis figure picker to inspect `fit_i`.
Here `i` is an original tone position, never a participants row number.

Use Data to save measurement data and analysis artifacts.
Load Data selects a native file or a saved run.
Loading replaces the analysis source without changing current Form inputs.
A later Form edit does not alter the historical pulse, binding or destination.
Failed or partial rows do not produce calibration candidates.
Read each Writeback preview destination, value and fit diagnostics.
Deselect unwanted items before **Apply Selected**.
Preview and Analyze do not write; applying one item does not apply other candidates.

## Form drafts

Form persistence contains exactly `data` and `selections`.
Selections use typed paths with integer list positions.
Invalid leaf text and participants drafts survive save/restore.
Restoration reports dropped fields and does not run converters or refresh capture.
If the saved input cannot be read, the host reports it and uses fresh defaults.
Reset explicitly captures current parameters and rebuilds initial values.

A raw `FormState.write` replaces the complete value at its path and clears selections
at or below it. To change one leaf, use its complete path, for example
`("tones", 0, "drive", "module", "gain")` for an inline drive.
Do not send a partial tone or pulse dictionary expecting a merge.
Writing does not run a converter; selecting another module kind does.
Recipe/MCP logical field paths omit the inline wrapper, for example
`("tones", 0, "drive", "gain")`. The integer is the full original tone position,
not a row in Participants and not a name key.

## Core reference

`MuxLenRabiExp.run(config, *, context)` 同時掃描選定 qubits 的 drive length，以 mux 取得各自的平均 IQ。入口不依賴 GUI，不建立 Form、recipe、catalog 或 writeback proposal。

## 輸入與長度

`MuxLenRabiCfg.modules` 包含共同 reset 及完整 `MuxReadoutCfg`。`tones: list[MuxToneCfg]` 長度等於完整 readout tone 數，順序就是原始 tone index。每個元素有固定的 `name`、必填 `drive` 與 `sweep`。所有 drive 都必須有效，使用不同 generator，不能占用 readout generator。完整 qubit/resonator 綁定由呼叫端驗證，core 只接 numeric cfg。

每個 tone 的 sweep 的 start/stop 是正的微秒長度。範圍可不同，點數必須相同。第 p 點使用所有 active tones 各自的第 p 個 length，不展開 Cartesian product。未參與 tone 的掃描點數也必須相同。

本批支援 const 與 flat_top。Const length 是整個 pulse 的長度。Flat_top length 是包含 raise_waveform 的總 requested length，waveform owner 從中扣除 ramp，決定 plateau。Core 不改 ramp、gain、frequency 或 phase。Arb 長度由 asset 決定，既有 owner 沒有公開 length sweep 操作，因此 cfg 與 Run 入口對全部元素拒絕 arb，未參與元素也不例外。Core 不伸縮或裁切 asset。

`prepare_length_scan(cfg, soccfg=...)` 複製 cfg，按 participants 順序查出原始 indices。它以公開 `sweep2array(..., "time", gen_ch=...)` 產生每個 drive 的量化 requested length 軸，交給 pulse 的 `set_param("length", ...)`。每條軸保留自己的硬體格點，不排序、去重或縮減點數。量化後非正或相鄰點重疊會拒絕。這些座標不是 compiled hardware trace，flat_top ramp 與 native clock rounding 可讓總播放時間差數個 ticks。Ramp 與最少 cycle 等硬體限制由既有 waveform/compiler owner 拒絕，錯誤直接傳出。

## 排程與結果

Run 重用 `acquire_mux_scan`，每點 compile 一次。共同 reset 後，既有 Join 讓獨立 drive branches 同時開始。短 branch 做完後等待當點最長 branch，之後才 mux readout。不以全 sweep 的最大長度 padding，不倒推起點或補齊末端，也不改 pre/post delay。非零 pre_delay 仍由各 pulse owner 處理。Reps、rounds、initial_delay 與 relax_delay 都來自共同 cfg。

`MuxSweepResult.tone_indices` 是 int64 `(K,)` 原始 indices。`coordinates` 是 float64 `(K,P)` 的量化 requested 總長度，記憶體單位為 us。`signals` 是 complex128 `(K,P)` 的 ADC-length-normalized 平均 IQ，單位 a.u.。`measurement_i` 以各自座標顯示 magnitude，i 是原始 index。重複 tone label 不會覆蓋圖。

```python
from zcu_tools.experiment.records import RunRecord
from zcu_lab.v2.mux.len_rabi import MuxLenRabiExp

core = MuxLenRabiExp()
result = core.run(cfg, context=context)
context.cancel_signal.raise_if_error()
source = RunRecord(cfg=cfg, result=result)
outcomes = core.analyze(source, core.Options(), plots=analysis_plots)
```

RunContext 提供相容的 soc/soccfg、caller-owned Plots 與每次新建的 StopSignal。Caller 負責 finish/release Plots 及裝置清理。Run 不保存檔案或上次狀態，也不修改輸入 cfg。

取消時，已完成點與目前點已完成 rounds 的平均會保留，未取得點維持 NaN。先呼叫 `raise_if_error()` 檢查 acquisition failure，再以 `is_set()` 區分取消與完整結果。Failure 保留原始 cause，不因已有 partial data 而報成功。Cfg、量化與 build 錯誤直接傳出。

## 分析與保存

分析只讀 explicit RunRecord，允許 cfg=None，不從目前表單補資料。每個原始 index 回傳 `MuxLenRabiFit` 或有原因的 `MuxFitFailure`，並建立 `fit_i`。每條曲線使用自己的 length 座標，移除非有限 IQ 後至少需要六點。空資料與不足點數是逐 tone failure，source shape/dtype 錯誤則直接 raise。

Options 沿用單 qubit Len Rabi。`decay=True` 預設擬合衰減 cosine，可關閉。`fit_phase=False` 預設固定 0°，由 signed amplitude 涵蓋兩個方向；啟用後擬合自由 phase。Fitter 擁有 π/π/2 length 與 covariance 誤差傳播，不另外實作模型。Length 及其 one-sigma errors 是 us，Rabi frequency 及其 error 是 MHz。Fit quality 使用本次有限觀測、模型與 covariance，不更動校準政策。數值 fit 成功不表示小截斷模擬或低品質曲線足以校準。

`save_run(source, path, metadata=...)` 保存同次完整 cfg、原始 indices、每條 length 軸與 complex IQ。Native 的 length 座標使用 seconds。`load_run(path)` 還原同次來源，不讀目前表單、library 或 waveform bytes。Cfg 保留原 drive pulse/ramp 供後續 proposal 使用，但 arb path 不會把 asset bytes 封存進 native 檔案。Caller 根據停止狀態填寫 complete、partial 或 stopped，不能把 NaN partial 報為完整成功。

## Adapter source and proposals

`MuxLenRabiAdapter` in `gui.py` uses the explicit Run/Load record and its capture.
Its native generated Form is registered as `mux/len_rabi`.
The registered MCP recipe is `mux_len_rabi`. See the
[mux recipe guide](../../../recipes/mux-guide.md) for typed overrides, origins,
source-bound reporting and explicit writeback decisions.

Summary rows retain original tone `position`, source `name` and each fit or failure
reason. Q3/Q1 produces rows for indices 2/0, not renumbered 0/1. Fit-quality
errors identify the corresponding summary row. Current Form edits do not change
these results or repair missing source names.

A successful, fully finite tone proposes `<qubit>.ext.rabi_freq` in MHz. With
source cfg, it also copies the measured const/flat_top pulse and proposes fitted
pi/pi2 lengths to `<qubit>.pulses.pi_amp` and `<qubit>.pulses.pi2_amp`.
Separate candidates point `module.x180` and `module.x90` to those library keys.
The `ext.pi_product` candidate uses the source gain times fitted pi length.
No proposal stretches or crops an arb asset, or replaces the source pulse with
current library contents. Missing capture yields no proposals; inconsistent
captured bindings raise. Failed or partial rows yield none.

Review the source shape and flat_top ramp, destinations, quality and uncertainty.
Analyze and proposal creation do not write. Apply each intended candidate
explicitly, including a slot update when wanted.
