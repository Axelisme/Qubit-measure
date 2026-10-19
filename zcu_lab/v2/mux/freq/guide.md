# Mux qubit frequency

## GUI setup

Open `mux/freq` from **New Tab** after binding `$mux` to a MuxGroup.
The capture must contain every qubit and its matching resonator in readout order.
An existing readout slot must use the mux family with that full tone count.
If no readout is stored, fill the blank mux readout before running.

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

`MuxFreqExp.run(config, *, context)` 同時驅動選定 qubits，以 mux 讀出平均 IQ。這是無 GUI 的 core 入口。它不建立 Form、recipe、catalog 或 writeback proposal。

## 輸入與排程

`participants: list[str]` 是非空、不重複的參與名稱清單，名稱必須存在於完整 tones。完整 Q1／Q2／Q3 搭配 participants Q3／Q1 時，執行與資料原始 indices 為 `[2, 0]`，不排序或重新編號。未參與元素保留設定並使用同一驗證規則。

`MuxFreqCfg.modules` 包含共同 reset 與完整 `MuxReadoutCfg`。`tones` 長度等於完整 readout tone 數，順序就是原始 tone index。每個 `MuxToneCfg` 有固定的 `name`、必填 `drive` 及 `sweep`。所有 drive 都必須有效，使用不同 generator，且不能占用 readout generator。

每個 tone 的 sweep 的 start/stop 是 MHz。範圍可以不同，點數必須相同。第 p 點使用所有 active tones 各自的第 p 個頻率，不作 Cartesian product。Core 在 Run 前驗證 cfg，依 participants 名稱解析原始位置，不讀 MuxGroup。呼叫端負責完整 qubit/resonator 綁定。

每點建立一個 program，依序執行共同 reset、Join 內的獨立 drive pulses、共同 mux readout。Join 讓 branches 同時開始，短 pulse 完成後等待長 pulse。Core 不補齊末端，不改 drive pre/post delay。Reps、rounds、initial_delay 與 relax_delay 來自同一份 cfg。Program 負責 native 量化與平均。

## 執行與取消

呼叫端提供 RunContext，內有相容的 soc/soccfg、Plots 與每次新建的 StopSignal。

```python
from zcu_tools.experiment.records import RunRecord
from zcu_lab.v2.mux.freq import MuxFreqExp

core = MuxFreqExp()
result = core.run(cfg, context=context)
context.cancel_signal.raise_if_error()
source = RunRecord(cfg=cfg, result=result)
outcomes = core.analyze(source, core.Options(), plots=analysis_plots)
```

先檢查 error，再檢查 `context.cancel_signal.is_set()` 區分完整與取消結果。Run 不保存檔案。Caller 負責 finish/release Plots 和裝置清理。

`result.tone_indices` 是 active 原始 indices，形狀 `(K,)`。`coordinates` 是各 drive 自己的量化 MHz，`signals` 是 ADC-length-normalized complex IQ，兩者形狀都是 `(K,P)`。`measurement_i` 用第 i 個原始 tone 的座標畫 magnitude。Label 重複不會覆蓋圖。

停止後，完成點與目前點已完成 rounds 的平均會保留。未取得的點仍是 NaN。Acquisition 錯誤留在共用 StopSignal，`raise_if_error()` 會拋出；build 與 cfg 錯誤直接傳出。有資料不代表操作完整成功。

## 分析與保存

分析沿用單 qubit lor/sinc fitter，每條曲線至少要有六個有效點。每個原始 index 回傳 `MuxFreqFit` 或帶原因的 `MuxFitFailure`，並建立 `fit_i`。Frequency、FWHM 及誤差單位都是 MHz。Fit quality 只描述該次模型，不保證可校準。

`save_run(source, path, metadata=...)` 保存同次完整 cfg、原始 indices、獨立座標與 complex IQ。Native 的 frequency 座標使用 Hz。`load_run(path)` 還原 explicit source，不讀目前表單。保存時由 caller 根據 StopSignal 填寫 complete、partial 或 stopped，不把 NaN partial 報為完整。

## Adapter source and proposals

`MuxFreqAdapter` in `gui.py` connects the core to explicit Run/Load sources.
Its native generated Form is registered as `mux/freq`.
The registered MCP recipe is `mux_freq`. See the
[mux recipe guide](../../../recipes/mux-guide.md) for typed overrides, origins,
source-bound reporting and explicit writeback decisions.

Its summary lists each acquired tone's original `position`, source `name`, fit
values and fit-quality diagnostics, or a failure reason. Rows follow
participants order. With Q3/Q1, row 0 describes index 2, not index 0.
The name is unavailable when source cfg is absent; the current Form does not
supply a replacement.

A successful, fully finite tone proposes that captured qubit's `freq` and
`kappa` in MHz, plus a frequency sample at the captured bias. The destination
comes only from the same Run/Load capture's complete qubit binding, never from
readout labels or current aliases. Missing capture produces no proposal;
inconsistent captured bindings raise. Failed or partially acquired rows produce
no proposals. Analyze and proposal creation do not write. Review the destination,
fit quality and uncertainty before explicitly applying individual candidates.
