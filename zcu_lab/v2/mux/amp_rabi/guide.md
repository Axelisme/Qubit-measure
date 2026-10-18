# Mux amplitude Rabi

## GUI setup

Open `mux/amp_rabi` from **New Tab** after binding `$mux` to a MuxGroup.
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

`MuxAmpRabiExp.run(config, *, context)` 同時掃描選定 qubits 的 drive gain，以 mux 讀出平均 IQ。這是無 GUI 的 core 入口。它不建立 Form、recipe、catalog 或 writeback proposal。

## 輸入與排程

`participants: list[str]` 是非空、不重複的參與名稱清單，名稱必須存在於完整 tones。完整 Q1／Q2／Q3 搭配 participants Q3／Q1 時，執行與資料原始 indices 為 `[2, 0]`，不排序或重新編號。未參與元素保留設定並使用同一驗證規則。

`MuxAmpRabiCfg.modules` 包含共同 reset 與完整 `MuxReadoutCfg`。完整 `tones` 每個元素包含固定的 `name`、必填 `drive` 及 `sweep`，順序就是原始 tone index。所有 drive 都必須有效，使用獨立 generator，不能占用 readout generator。

每個 tone 的 sweep 的 start/stop 是 normalized gain，必須在 `[-1,1]`。範圍可以不同，點數必須相同。第 p 點同時使用每個 active tone 的第 p 個 gain，不作 Cartesian product。未參與 tone 同樣驗證，掃描點數也必須相同。Core 驗證 numeric cfg，完整 qubit/resonator 綁定由呼叫端負責。

每點 compile 一次，依序執行共同 reset、Join 內的獨立 drive pulses、共同 mux readout。Branches 同時開始，短 pulse 完成後等待長 pulse。Core 不做末端 padding，不改 drive pre/post delay。所有 qubits 共用 reps、rounds、initial_delay 及 relax_delay。Native conversion 留在 program。

## 執行與結果

```python
from zcu_tools.experiment.records import RunRecord
from zcu_lab.v2.mux.amp_rabi import MuxAmpRabiExp

core = MuxAmpRabiExp()
result = core.run(cfg, context=context)
context.cancel_signal.raise_if_error()
source = RunRecord(cfg=cfg, result=result)
outcomes = core.analyze(source, core.Options(), plots=analysis_plots)
```

RunContext 提供相容的 soc/soccfg、caller-owned Plots 與每次新建的 StopSignal。Caller 負責 finish/release 與裝置清理。Run 不保存檔案或上次狀態。

`result.tone_indices` 是 active 原始 indices，形狀 `(K,)`。`coordinates` 是各 drive 自己的量化 normalized gains，`signals` 是 ADC-length-normalized complex IQ。兩者都是 `(K,P)`。`measurement_i` 用該原始 tone 的座標畫 magnitude，不因 label 重複覆蓋圖。

Program 平均 reps/rounds，completed-round hook 保留目前點已完成 rounds 的平均。取消後，未取得點維持 NaN。Caller 先呼叫 `raise_if_error()` 檢查 failure，再用 `cancel_signal.is_set()` 區分取消與完整結果。Acquisition 錯誤不因 partial data 而成為成功。Cfg 與 build 錯誤直接傳出。

## 分析與保存

分析沿用單 qubit undamped Rabi fitter 與 zero-gain phase 政策。Options.skip 先移除前幾個 zip 點，每條曲線再保留有限 IQ，至少需要六點。每個原始 index 回傳 `MuxAmpRabiFit` 或帶原因的 `MuxFitFailure`。Fit 圖命名 `fit_i`。π/π/2 gain 與誤差都是 normalized gain。Fit quality 不改 fitter 或校準政策，數值 fit 成功不代表資料足以校準。

`save_run(source, path, metadata=...)` 保存同次完整 cfg、原始 indices、獨立 gain 座標與 complex IQ。`load_run(path)` 還原 explicit source，不以目前 cfg 補歷史資料。Caller 根據 StopSignal 填寫 complete、partial 或 stopped，不把未取得的 NaN 報為完整成功。

## Adapter source and proposals

`MuxAmpRabiAdapter` in `gui.py` uses the explicit Run/Load record and its capture.
Its native generated Form is registered as `mux/amp_rabi`.
A recipe entry is not available yet.

Summary rows retain the original tone `position` and source `name`, including failed
fits and their reasons. Rows follow participants order, so Q3/Q1 gives indices
2/0. Fit-quality errors identify the corresponding summary row. Current Form
edits do not alter these results or supply missing source names.

For each successful, fully finite tone, the adapter copies that source cfg's
pulse and proposes pi/pi2 gain updates to `<qubit>.pulses.pi_amp` and
`<qubit>.pulses.pi2_amp`. Separate candidates point `module.x180` and
`module.x90` at those library keys. It also proposes `ext.pi_product` from the
fitted pi gain times the source pulse's scalar length. It does not fetch a
replacement pulse from the current library. Without source cfg it cannot build
these pulse proposals. Missing capture yields no proposals; inconsistent captured
bindings raise. Failed or partial rows yield none.

Inspect pulse shape, destination, fit quality and uncertainty in the proposed
items. Analyze and proposal creation do not write; applying one item does not
implicitly apply the module-slot or other candidates.
